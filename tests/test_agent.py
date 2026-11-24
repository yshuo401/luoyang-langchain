from __future__ import annotations

import time
from typing import Any

import pytest
from app.agent import (
    create_memory_checkpointer,
    create_tourism_agent,
    finalize_grounded_result,
    parse_citation_numbers,
)
from app.schemas import SourceItem
from app.tools import RequestContext, reset_request_context, set_request_context
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


class ToolCapableFakeChatModel(FakeListChatModel):
    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        del tools, tool_choice, kwargs
        return self


class ScriptedToolChatModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        del tools, tool_choice, kwargs
        return self


class FakeSpotClient:
    async def search_spots(self, **_: Any) -> dict[str, Any]:
        return {
            "records": [
                {
                    "id": "1001",
                    "name": "龙门石窟",
                    "summary": "世界文化遗产",
                    "deepLink": "/pages/spot/detail?id=1001",
                }
            ]
        }


@pytest.mark.asyncio
async def test_agent_executes_local_tool_and_registers_source() -> None:
    model = ScriptedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_scenic_spots",
                        "args": {"keyword": "龙门石窟"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="龙门石窟是世界文化遗产。[S1]"),
        ]
    )
    agent = create_tourism_agent(
        chat_model=model,
        checkpointer=create_memory_checkpointer(),
    )
    context = RequestContext(
        user_id="10001",
        thread_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        springboot_client=FakeSpotClient(),
        rag_service=None,
        max_tool_calls=6,
    )
    token = set_request_context(context)
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "龙门石窟有什么介绍？"}]},
            config={"configurable": {"thread_id": context.thread_id}},
        )
    finally:
        reset_request_context(token)

    assert result["messages"][-1].content == "龙门石窟是世界文化遗产。[S1]"
    assert context.tool_call_count == 1
    assert context.sources[0].source_type == "SPOT"


@pytest.mark.asyncio
async def test_thread_id_keeps_two_turns() -> None:
    checkpointer = create_memory_checkpointer()
    agent = create_tourism_agent(
        chat_model=ToolCapableFakeChatModel(responses=["第一轮回答", "第二轮回答"]),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}}
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "先介绍龙门石窟"}]}, config=config
    )
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "附近还有什么？"}]}, config=config
    )
    state = await agent.aget_state(config)
    contents = [message.content for message in state.values["messages"]]
    assert "先介绍龙门石窟" in contents
    assert "附近还有什么？" in contents


def test_citation_and_insufficient_validation() -> None:
    assert parse_citation_numbers("结论[S1]，补充[S2]。") == [1, 2]
    result = finalize_grounded_result(
        answer="不存在的来源[S99]",
        sources=[SourceItem(sourceType="SPOT", sourceId="1", title="景点", rankNo=1)],
        started_at=time.monotonic(),
        model_name="qwen3.8-flash",
    )
    assert result.insufficient is True
    assert result.insufficient_reason == "CITATION_INVALID"
    assert result.sources == []
