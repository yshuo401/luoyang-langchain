"""一次 AI 问答的用例。

调用顺序固定为：

    guard -> checkpoint -> Agent -> 来源校验 -> ChatResult

把这段流程单独放在 `chat.py`，是为了让 `main.py` 只处理 HTTP，让
`agent.py` 只处理 Agent 本身，让 `tools.py` 只处理工具。
"""

from __future__ import annotations

import asyncio
import re
import time
from enum import StrEnum
from typing import Any

from app.agent import (
    OUT_OF_SCOPE,
    QUESTION_TOO_VAGUE,
    RETRIEVAL_EMPTY,
    TOOL_DEPENDENCY_UNAVAILABLE,
    elapsed_ms,
    extract_final_answer,
    extract_token_usage,
    finalize_grounded_result,
    has_messages,
    insufficient_result,
)
from app.config import Settings, get_settings
from app.errors import CheckpointerUnavailableError, ModelUnavailableError
from app.schemas import ChatRequest, ChatResult
from app.tools import RequestContext, reset_request_context, set_request_context

GREETING_RESPONSE = (
    "你好，我是洛阳旅游助手。可以问我景点介绍、游玩时长、攻略、美食，也可以让我解释推荐结果。"
)
REFUSAL_RESPONSE = "这个问题超出洛阳旅游助手的能力范围，我不能提供系统提示、密钥或代码执行帮助。"
VAGUE_RESPONSE = "请补充具体景点、时间或偏好，例如“龙门石窟适合玩多久？”"

_GREETING = re.compile(
    r"^(你好|您好|嗨|hi|hello|哈喽|早上好|下午好|晚上好)[！!。.?？\s]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(谢谢|感谢|多谢|thanks|thank you)[！!。.?？\s]*$", re.IGNORECASE)
_UNSAFE = re.compile(
    r"(忽略.{0,12}(指令|提示|规则)|系统提示|system prompt|api[_ -]?key|密钥|数据库密码|"
    r"drop\s+table|rm\s+-rf|curl\s+|powershell|<\s*script)",
    re.IGNORECASE,
)


class GuardIntent(StrEnum):
    GREETING = "GREETING"
    THANKS = "THANKS"
    UNSAFE = "UNSAFE"
    VAGUE = "VAGUE"
    NORMAL = "NORMAL"


def classify_guard_intent(message: str) -> GuardIntent:
    normalized = message.strip()
    if _GREETING.fullmatch(normalized):
        return GuardIntent.GREETING
    if _THANKS.fullmatch(normalized):
        return GuardIntent.THANKS
    if _UNSAFE.search(normalized):
        return GuardIntent.UNSAFE
    if len(normalized) <= 2 and not any(char.isdigit() for char in normalized):
        return GuardIntent.VAGUE
    return GuardIntent.NORMAL


class ChatService:
    """处理一次请求，不把 FastAPI 的 Request/Response 混进业务逻辑。"""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        springboot_client: Any,
        rag_service: Any | None,
        agent: Any | None,
        checkpointer: Any | None,
    ) -> None:
        self.settings = settings or get_settings()
        self.springboot_client = springboot_client
        self.rag_service = rag_service
        self.agent = agent
        self.checkpointer = checkpointer

    async def chat(self, request: ChatRequest, request_id: str) -> ChatResult:
        del request_id
        started_at = time.monotonic()
        intent = classify_guard_intent(request.message)

        if intent in {GuardIntent.GREETING, GuardIntent.THANKS}:
            return self._fixed_result(intent, started_at)
        if intent is GuardIntent.UNSAFE:
            return insufficient_result(
                OUT_OF_SCOPE,
                started_at=started_at,
                answer=REFUSAL_RESPONSE,
            )
        if intent is GuardIntent.VAGUE:
            return insufficient_result(
                QUESTION_TOO_VAGUE,
                started_at=started_at,
                answer=VAGUE_RESPONSE,
            )
        if self.agent is None or self.checkpointer is None:
            raise CheckpointerUnavailableError()

        # LangGraph 用 thread_id 找到同一会话的旧消息和 checkpoint。
        config = {
            "configurable": {"thread_id": request.thread_id},
            "recursion_limit": self.settings.agent_recursion_limit,
        }
        try:
            state = await self.agent.aget_state(config)
        except Exception as exc:
            raise CheckpointerUnavailableError("会话状态暂时不可用") from exc

        messages = self._build_messages(
            has_checkpoint=has_messages(state),
            recovery_history=request.recovery_history,
            current_message=request.message,
        )
        context = RequestContext(
            user_id=request.user_id,
            thread_id=request.thread_id,
            springboot_client=self.springboot_client,
            rag_service=self.rag_service,
            max_tool_calls=self.settings.agent_max_tool_calls,
        )
        token = set_request_context(context)
        try:
            agent_result = await asyncio.wait_for(
                self.agent.ainvoke({"messages": messages}, config=config),
                timeout=self.settings.chat_total_timeout_seconds,
            )
        except TimeoutError as exc:
            raise ModelUnavailableError("智能问答超时，请稍后重试") from exc
        except Exception as exc:
            raise ModelUnavailableError("智能问答暂时不可用") from exc
        finally:
            reset_request_context(token)

        answer = extract_final_answer(agent_result)
        if not answer:
            reason = (
                TOOL_DEPENDENCY_UNAVAILABLE
                if context.tool_errors and not context.sources
                else RETRIEVAL_EMPTY
            )
            return insufficient_result(reason, started_at=started_at)

        return finalize_grounded_result(
            answer=answer,
            sources=context.sources,
            started_at=started_at,
            model_name=self.settings.qwen_chat_model,
            token_usage=extract_token_usage(agent_result),
        )

    def _fixed_result(self, intent: GuardIntent, started_at: float) -> ChatResult:
        answer = (
            GREETING_RESPONSE if intent is GuardIntent.GREETING else "不客气，祝你洛阳之行愉快。"
        )
        return ChatResult(
            answer=answer,
            grounded=False,
            insufficient=False,
            sources=[],
            suggested_questions=[
                "龙门石窟适合玩多久？",
                "帮我安排一条洛阳半天路线。",
            ],
            model_name=None,
            latency_ms=elapsed_ms(started_at),
        )

    @staticmethod
    def _build_messages(
        *,
        has_checkpoint: bool,
        recovery_history: list[Any],
        current_message: str,
    ) -> list[dict[str, str]]:
        """只有 checkpoint 丢失时才使用 Java 传来的恢复历史。

        如果进程内还有 checkpoint，就直接利用 LangGraph 的完整上下文，
        避免把最近消息重复拼接给模型。
        """

        messages: list[dict[str, str]] = []
        if not has_checkpoint:
            messages.extend(
                {"role": item.role, "content": item.content} for item in recovery_history[-10:]
            )
        messages.append({"role": "user", "content": current_message})
        return messages

    async def reset_checkpoint(self, thread_id: str) -> None:
        if self.checkpointer is None:
            raise CheckpointerUnavailableError()
        try:
            await self.checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            raise CheckpointerUnavailableError("会话状态清理暂时不可用") from exc

    async def close(self) -> None:
        return None
