"""Shared helpers for the static Agent tool whitelist."""

from __future__ import annotations

from typing import Any

from app.agents.state import RequestContext, current_request_context
from app.core.errors import SpringBootError


def begin_tool(name: str) -> tuple[RequestContext | None, dict[str, Any] | None]:
    context = current_request_context()
    if not context.reserve_tool_call(name):
        return None, {
            "ok": False,
            "errorCode": "TOOL_BUDGET_EXCEEDED",
            "message": "本轮工具调用次数已达上限",
            "retryable": False,
        }
    return context, None


def tool_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, SpringBootError):
        return {
            "ok": False,
            "errorCode": exc.error_code,
            "message": exc.message,
            "retryable": exc.retryable,
        }
    return {
        "ok": False,
        "errorCode": "TOOL_EXECUTION_ERROR",
        "message": "数据查询暂时不可用",
        "retryable": True,
    }


def records_from_response(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        records = response.get("records", [])
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def register_records(
    context: RequestContext,
    source_type: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        marker = context.evidence.register_business_item(source_type, record)  # type: ignore[arg-type]
        enriched.append({**record, "marker": f"[{marker}]"})
    return enriched


def text(value: Any, *, max_chars: int = 1200) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def safe_deep_link(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.startswith("/pages/") and "://" not in value else None


def success(items: list[dict[str, Any]], *, truncated: bool = False) -> dict[str, Any]:
    return {
        "ok": True,
        "data": items,
        "meta": {
            "count": len(items),
            "truncated": truncated,
        },
    }

