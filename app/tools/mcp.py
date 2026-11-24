"""MCP 12306 工具加载与包装。

MCP 是可选增强。默认关闭；即使启动失败，本地工具和 RAG 仍可工作。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import Settings, get_settings
from app.tools.context import begin_tool


def _mcp_text(result: Any) -> str:
    """把 MCP 返回的 content blocks 合并成普通字符串。"""

    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, list):
        parts = [_mcp_text(item) for item in result]
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        if "content" in result:
            return _mcp_text(result["content"])
        return json.dumps(result, ensure_ascii=False)
    return str(result or "")


def _wrap_mcp_tool(mcp_tool: BaseTool, settings: Settings) -> BaseTool:
    """给 MCP 工具增加工具预算、超时和来源登记。"""

    async def invoke(**kwargs: Any) -> dict[str, Any]:
        context, error = begin_tool(mcp_tool.name)
        if error is not None:
            return error
        assert context is not None
        try:
            result = await asyncio.wait_for(
                mcp_tool.ainvoke(kwargs),
                timeout=settings.mcp_tool_timeout_seconds,
            )
            content = _mcp_text(result).strip() or "12306 MCP 没有返回内容"
            marker = context.add_mcp_result(mcp_tool.name, content)
            return {
                "ok": True,
                "source": "12306-MCP",
                "marker": f"[{marker}]",
                "data": content[:4000],
                "notice": "实时数据仅供参考，请以 12306 官方页面为准。",
            }
        except TimeoutError:
            context.record_error("MCP_TIMEOUT")
            return {
                "ok": False,
                "errorCode": "MCP_TIMEOUT",
                "message": "12306 查询超时",
                "retryable": True,
            }
        except Exception as exc:
            context.record_error("MCP_TOOL_ERROR")
            return {
                "ok": False,
                "errorCode": "MCP_TOOL_ERROR",
                "message": f"12306 查询失败：{type(exc).__name__}",
                "retryable": True,
            }

    return StructuredTool.from_function(
        coroutine=invoke,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
    )


async def load_mcp_tools(settings: Settings | None = None) -> list[BaseTool]:
    """按配置加载 12306 MCP；默认关闭，失败由 runtime 记录日志。"""

    config = settings or get_settings()
    if not config.mcp_12306_enabled:
        return []

    client = MultiServerMCPClient(
        {
            "rail12306": {
                "transport": "stdio",
                "command": config.mcp_12306_command,
                "args": ["-y", "12306-mcp"],
            }
        }
    )
    tools = await client.get_tools()
    return [_wrap_mcp_tool(item, config) for item in tools]
