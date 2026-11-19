"""Deterministic recommendation tool; Java owns candidates and ordering."""

from __future__ import annotations

from typing import Any, Literal

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


class RecommendationInput(BaseModel):
    scene: Literal["HOME", "HALF_DAY", "ONE_DAY", "NEARBY"] = Field(
        description="推荐场景"
    )
    limit: int = Field(default=5, ge=1, le=10)
    context: str | None = Field(default=None, max_length=500, description="用户补充偏好")


@tool("recommend_spots", args_schema=RecommendationInput)
async def recommend_spots(
    scene: str,
    limit: int = 5,
    context: str | None = None,
) -> dict[str, Any]:
    """获取 Java 确定性推荐候选，AI 只能解释候选顺序和理由。"""

    request_context, error = begin_tool("recommend_spots")
    if error is not None:
        return error
    assert request_context is not None
    try:
        response = await request_context.springboot_client.recommend(
            user_id=request_context.user_id,
            scene=scene,
            limit=limit,
            context=context,
        )
        records = records_from_response(response)
        if not records:
            result = success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(request_context, "RECOMMENDATION", records[:limit])
        items = [
            {
                "id": str(item.get("id") or item.get("spotId") or ""),
                "name": text(item.get("name") or item.get("title"), max_chars=100),
                "reason": text(item.get("reason") or item.get("summary"), max_chars=500),
                "rank": index + 1,
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for index, item in enumerate(enriched)
        ]
        return success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        if "errorCode" in result:
            request_context.record_error(str(result["errorCode"]))
        return result
