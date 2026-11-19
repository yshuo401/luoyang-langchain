from __future__ import annotations

from typing import Any

import pytest
from app.agents.tourism_agent import create_tourism_agent
from app.memory.checkpointer import create_memory_checkpointer
from langchain_core.language_models.fake_chat_models import FakeListChatModel
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


@pytest.mark.asyncio
async def test_agent_uses_thread_id_for_multi_turn_memory() -> None:
    checkpointer = create_memory_checkpointer()
    agent = create_tourism_agent(
        chat_model=ToolCapableFakeChatModel(responses=["第一轮回答", "第二轮回答"]),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}}
    first = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "先介绍龙门石窟"}]},
        config=config,
    )
    assert first["messages"][-1].content == "第一轮回答"
    second = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "那它附近还有什么？"}]},
        config=config,
    )
    assert second["messages"][-1].content == "第二轮回答"
    state = await agent.aget_state(config)
    contents = [message.content for message in state.values["messages"]]
    assert "先介绍龙门石窟" in contents
    assert "那它附近还有什么？" in contents


@pytest.mark.asyncio
async def test_deleted_thread_does_not_restore_old_state() -> None:
    checkpointer = create_memory_checkpointer()
    agent = create_tourism_agent(
        chat_model=ToolCapableFakeChatModel(responses=["旧回答", "新会话回答"]),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}}
    await agent.ainvoke({"messages": [{"role": "user", "content": "旧问题"}]}, config=config)
    await checkpointer.adelete_thread(config["configurable"]["thread_id"])
    state = await agent.aget_state(config)
    assert not state.values.get("messages")
    await agent.ainvoke({"messages": [{"role": "user", "content": "新问题"}]}, config=config)
    state = await agent.aget_state(config)
    contents = [message.content for message in state.values["messages"]]
    assert "旧问题" not in contents
