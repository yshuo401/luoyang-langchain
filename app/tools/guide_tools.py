"""Travel-guide tools backed by Spring Boot internal APIs."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools._common import (
    begin_tool,
    records_from_response,
    register_records,
    safe_deep_link,
    success,
    text,
    tool_error,
)


class GuideSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="路线、攻略或景点关键词")
    limit: int = Field(default=5, ge=1, le=10)


@tool("search_travel_guides", args_schema=GuideSearchInput)
async def search_travel_guides(keyword: str, limit: int = 5) -> dict[str, Any]:
    """搜索已发布洛阳攻略和路线，返回行程摘要与深链。"""

    context, error = begin_tool("search_travel_guides")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_guides(keyword=keyword, limit=limit)
        records = records_from_response(response)
        if not records:
            result = success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "GUIDE", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "title": text(item.get("title"), max_chars=120),
                "summary": text(item.get("summary"), max_chars=500),
                "durationDays": item.get("durationDays"),
                "routeSummary": text(item.get("routeSummary"), max_chars=800),
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        if "errorCode" in result:
            context.record_error(str(result["errorCode"]))
        return result
