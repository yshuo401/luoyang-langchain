"""Chat use case: guard -> checkpoint -> Agent -> evidence validation -> cache."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from app.agents.state import RequestContext, reset_request_context, set_request_context
from app.agents.tourism_agent import extract_final_answer, extract_token_usage, has_messages
from app.chains.grounding import (
    elapsed_ms,
    finalize_grounded_result,
    insufficient_result,
)
from app.core.config import Settings, get_settings
from app.core.constants import InsufficientReason
from app.core.errors import (
    CheckpointerUnavailableError,
    ConflictError,
    ModelUnavailableError,
)
from app.core.logging import get_logger
from app.memory.chat_state import ChatStateStore, RedisChatStateStore
from app.prompts.guard import (
    GREETING_RESPONSE,
    REFUSAL_RESPONSE,
    VAGUE_RESPONSE,
    GuardIntent,
    classify_guard_intent,
)
from app.schemas.chat import ChatRequest, ChatResult

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        state_store: ChatStateStore,
        springboot_client: Any,
        rag_service: Any | None,
        agent: Any | None,
        checkpointer: Any | None,
    ) -> None:
        self._settings = settings
        self._state_store = state_store
        self._springboot_client = springboot_client
        self._rag_service = rag_service
        self._agent = agent
        self._checkpointer = checkpointer

    @classmethod
    def from_settings(
        cls,
        *,
        springboot_client: Any,
        rag_service: Any,
        agent: Any | None,
        checkpointer: Any | None,
        settings: Settings | None = None,
    ) -> ChatService:
        config = settings or get_settings()
        return cls(
            settings=config,
            state_store=RedisChatStateStore(config),
            springboot_client=springboot_client,
            rag_service=rag_service,
            agent=agent,
            checkpointer=checkpointer,
        )

    async def chat(self, request: ChatRequest, request_id: str) -> ChatResult:
        del request_id
        started_at = time.monotonic()
        guard_intent = classify_guard_intent(request.message)
        if guard_intent in {GuardIntent.GREETING, GuardIntent.THANKS}:
            return self._fixed_result(guard_intent, started_at)
        if guard_intent is GuardIntent.UNSAFE:
            return insufficient_result(
                InsufficientReason.OUT_OF_SCOPE,
                started_at=started_at,
                answer=REFUSAL_RESPONSE,
            )
        if guard_intent is GuardIntent.VAGUE:
            return insufficient_result(
                InsufficientReason.QUESTION_TOO_VAGUE,
                started_at=started_at,
                answer=VAGUE_RESPONSE,
            )
        if self._agent is None or self._checkpointer is None:
            raise CheckpointerUnavailableError()

        result_key = f"ai:chat:result:{request.thread_id}:{request.client_message_id}"
        lock_key = f"ai:chat:inflight:{request.thread_id}:{request.client_message_id}"
        try:
            cached = await self._state_store.get_json(result_key)
        except Exception as exc:
            raise CheckpointerUnavailableError("会话缓存暂时不可用") from exc
        if cached is not None:
            return ChatResult.model_validate(cached)

        try:
            acquired = await self._state_store.set_if_absent(
                lock_key,
                request.request_id,
                ttl_seconds=self._settings.chat_inflight_ttl_seconds,
            )
        except Exception as exc:
            raise CheckpointerUnavailableError("会话锁暂时不可用") from exc
        if not acquired:
            raise ConflictError("该消息正在处理中", code=40903, retryable=True)

        try:
            config = {
                "configurable": {"thread_id": request.thread_id},
                "recursion_limit": self._settings.agent_recursion_limit,
            }
            try:
                state = await self._agent.aget_state(config)
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
                springboot_client=self._springboot_client,
                rag_service=self._rag_service,
                max_tool_calls=self._settings.agent_max_tool_calls,
            )
            token = set_request_context(context)
            try:
                agent_result = await asyncio.wait_for(
                    self._agent.ainvoke(
                        {"messages": messages},
                        config=config,
                    ),
                    timeout=self._settings.chat_total_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ModelUnavailableError("智能问答超时，请稍后重试") from exc
            except Exception as exc:
                logger.exception(
                    "agent_invocation_failed",
                    extra={"event": "agent_invocation_failed", "errorCode": "AGENT_ERROR"},
                )
                raise ModelUnavailableError("智能问答暂时不可用") from exc
            finally:
                reset_request_context(token)

            answer = extract_final_answer(agent_result)
            if not answer:
                reason = (
                    InsufficientReason.TOOL_DEPENDENCY_UNAVAILABLE
                    if context.tool_errors and not context.evidence.sources
                    else InsufficientReason.RETRIEVAL_EMPTY
                )
                return insufficient_result(reason, started_at=started_at)
            token_usage = extract_token_usage(agent_result)
            result = finalize_grounded_result(
                answer=answer,
                evidence=context.evidence,
                started_at=started_at,
                model_name=self._settings.qwen_chat_model,
                token_usage=token_usage,
            )
            await self._state_store.set_json(
                result_key,
                result.model_dump(by_alias=True),
                ttl_seconds=self._settings.chat_result_cache_ttl_seconds,
            )
            logger.info(
                "ai_chat_completed",
                extra={
                    "event": "ai_chat_completed",
                    "latencyMs": elapsed_ms(started_at),
                    "resultCount": len(result.sources),
                },
            )
            return result
        finally:
            try:
                await self._state_store.delete(lock_key)
            except Exception:
                logger.exception("chat_lock_release_failed")

    def _fixed_result(self, intent: GuardIntent, started_at: float) -> ChatResult:
        answer = (
            GREETING_RESPONSE
            if intent is GuardIntent.GREETING
            else "不客气，祝你洛阳之行愉快。"
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
        messages: list[dict[str, str]] = []
        if not has_checkpoint:
            messages.extend(
                {"role": item.role, "content": item.content}
                for item in recovery_history[-10:]
            )
        messages.append({"role": "user", "content": current_message})
        return messages

    @staticmethod
    def hash_user_id(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]

    async def reset_checkpoint(self, thread_id: str) -> None:
        if self._checkpointer is None:
            raise CheckpointerUnavailableError()
        try:
            await self._checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            raise CheckpointerUnavailableError("会话状态清理暂时不可用") from exc

    async def close(self) -> None:
        await self._state_store.close()


