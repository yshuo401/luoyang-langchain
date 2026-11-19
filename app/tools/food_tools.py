"""Luoyang food tools backed by Spring Boot internal APIs."""

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


class FoodSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="美食名称或关键词")
    category_id: str | None = Field(default=None, max_length=64, description="可选分类 ID")
    limit: int = Field(default=5, ge=1, le=10)


@tool("search_foods", args_schema=FoodSearchInput)
async def search_foods(
    keyword: str,
    category_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """搜索洛阳美食，返回名称、摘要、推荐景点与深链。"""

    context, error = begin_tool("search_foods")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_foods(
            keyword=keyword,
            category_id=category_id,
            limit=limit,
        )
        records = records_from_response(response)
        if not records:
            result = success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "FOOD", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": text(item.get("name"), max_chars=100),
                "summary": text(item.get("summary"), max_chars=500),
                "categoryName": text(item.get("categoryName"), max_chars=100),
                "recommendedSpots": item.get("recommendedSpots", [])[:10]
                if isinstance(item.get("recommendedSpots"), list)
                else [],
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
