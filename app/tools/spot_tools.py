"""Scenic-spot tools backed by Spring Boot internal APIs."""

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


class SpotSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="景点名称或关键词")
    category_id: str | None = Field(default=None, max_length=64, description="可选分类 ID")
    limit: int = Field(default=5, ge=1, le=10, description="最多返回条数")


class SpotDetailInput(BaseModel):
    spot_id: str = Field(min_length=1, max_length=64, description="景点 ID")


@tool("search_scenic_spots", args_schema=SpotSearchInput)
async def search_scenic_spots(
    keyword: str,
    category_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """搜索当前已发布景点，返回名称、摘要、地址、开放时间摘要和深链。"""

    context, error = begin_tool("search_scenic_spots")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_spots(
            keyword=keyword,
            category_id=category_id,
            limit=limit,
        )
        records = records_from_response(response)
        if not records:
            result = success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "SPOT", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": text(item.get("name"), max_chars=100),
                "summary": text(item.get("summary"), max_chars=500),
                "categoryName": text(item.get("categoryName"), max_chars=100),
                "address": text(item.get("address"), max_chars=200),
                "openTimeDesc": text(item.get("openTimeDesc"), max_chars=200),
                "tags": item.get("tags", [])[:10] if isinstance(item.get("tags"), list) else [],
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


@tool("get_spot_detail", args_schema=SpotDetailInput)
async def get_spot_detail(spot_id: str) -> dict[str, Any]:
    """获取单个已发布景点的摘要、开放时间、票务提示和深链。"""

    context, error = begin_tool("get_spot_detail")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.get_spot(spot_id)
        if not isinstance(response, dict):
            return {
                "ok": False,
                "errorCode": "SPOT_NOT_FOUND",
                "message": "未找到该景点",
                "retryable": False,
            }
        marker = context.evidence.register_business_item("SPOT", response)
        item = {
            "id": str(response.get("id", "")),
            "name": text(response.get("name"), max_chars=100),
            "summary": text(response.get("summary"), max_chars=500),
            "descriptionText": text(response.get("descriptionText"), max_chars=1200),
            "openTimeDesc": text(response.get("openTimeDesc"), max_chars=200),
            "ticketInfo": text(response.get("ticketInfo"), max_chars=300),
            "address": text(response.get("address"), max_chars=200),
            "playDurationMinutes": response.get("playDurationMinutes"),
            "tags": response.get("tags", [])[:10] if isinstance(response.get("tags"), list) else [],
            "deepLink": safe_deep_link(response.get("deepLink")),
            "marker": f"[{marker}]",
        }
        return success([item])
    except Exception as exc:
        result = tool_error(exc)
        if "errorCode" in result:
            context.record_error(str(result["errorCode"]))
        return result
