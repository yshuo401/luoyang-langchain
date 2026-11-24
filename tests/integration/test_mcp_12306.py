from __future__ import annotations

import os

import pytest
from app.config import get_settings
from app.tools import load_mcp_tools

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MCP_TESTS") != "1",
    reason="设置 RUN_MCP_TESTS=1 后才启动 12306 MCP",
)


@pytest.mark.asyncio
async def test_12306_mcp_tools_are_loadable() -> None:
    settings = get_settings().model_copy(update={"mcp_12306_enabled": True})
    tools = await load_mcp_tools(settings)
    names = {tool.name for tool in tools}
    assert "get-tickets" in names
