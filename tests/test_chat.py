from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from app.chat import ChatService
from app.config import get_settings
from app.schemas import ChatRequest, SourceItem
from app.tools import current_request_context
from langchain_core.messages import AIMessage


class FakeAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[list[dict[str, str]]] = []
        self.has_checkpoint = False

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(
            values={"messages": [AIMessage(content="旧回答")]} if self.has_checkpoint else {}
        )

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        del config
        self.calls += 1
        self.inputs.append(payload["messages"])
        context = current_request_context()
        context.add_source(
            SourceItem(
                sourceType="SPOT",
                sourceId="1001",
                title="龙门石窟",
                snippet="世界文化遗产",
                deepLink="/pages/spot/detail?id=1001",
            )
        )
        self.has_checkpoint = True
        return {"messages": [AIMessage(content="龙门石窟适合游玩半天。[S1]")]}


def _request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "userId": "10001",
            "message": "龙门石窟适合玩多久？",
            "clientMessageId": "client-message-001",
            "requestId": "request-001",
            "recoveryHistory": [{"role": "user", "content": "上一轮在问洛阳"}],
        }
    )


@pytest.mark.asyncio
async def test_chat_recovers_history_then_reuses_checkpoint() -> None:
    agent = FakeAgent()
    service = ChatService(
        settings=get_settings(),
        springboot_client=object(),
        rag_service=None,
        agent=agent,
        checkpointer=object(),
    )
    first = await service.chat(_request(), "request-001")
    second = await service.chat(_request(), "request-001")
    assert first.grounded is True
    assert second.answer == first.answer
    assert agent.calls == 2
    assert agent.inputs[0][0]["content"] == "上一轮在问洛阳"
    assert len(agent.inputs[1]) == 1


@pytest.mark.asyncio
async def test_greeting_does_not_require_agent() -> None:
    service = ChatService(
        settings=get_settings(),
        springboot_client=object(),
        rag_service=None,
        agent=None,
        checkpointer=None,
    )
    request = _request().model_copy(update={"message": "你好"})
    result = await service.chat(request, "request-greeting")
    assert result.insufficient is False
    assert "洛阳旅游助手" in result.answer
