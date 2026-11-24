"""Java 业务数据工具。

这些工具只负责把 Spring Boot 返回的数据整理成 Agent 可读的结构，并登记
来源。它们不直接访问 MySQL，也不直接访问 Milvus。

包含五个工具：

- search_scenic_spots
- get_spot_detail
- search_travel_guides
- search_foods
- recommend_spots
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools.context import (
    begin_tool,
    compact_text,
    records_from_response,
    register_records,
    safe_deep_link,
    success_response,
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
            result = success_response([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "SPOT", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": compact_text(item.get("name"), max_chars=100),
                "summary": compact_text(item.get("summary"), max_chars=500),
                "categoryName": compact_text(item.get("categoryName"), max_chars=100),
                "address": compact_text(item.get("address"), max_chars=200),
                "openTimeDesc": compact_text(item.get("openTimeDesc"), max_chars=200),
                "tags": item.get("tags", [])[:10] if isinstance(item.get("tags"), list) else [],
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return success_response(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
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
        marker = context.add_business_item("SPOT", response)
        item = {
            "id": str(response.get("id", "")),
            "name": compact_text(response.get("name"), max_chars=100),
            "summary": compact_text(response.get("summary"), max_chars=500),
            "descriptionText": compact_text(response.get("descriptionText"), max_chars=1200),
            "openTimeDesc": compact_text(response.get("openTimeDesc"), max_chars=200),
            "ticketInfo": compact_text(response.get("ticketInfo"), max_chars=300),
            "address": compact_text(response.get("address"), max_chars=200),
            "playDurationMinutes": response.get("playDurationMinutes"),
            "tags": response.get("tags", [])[:10] if isinstance(response.get("tags"), list) else [],
            "deepLink": safe_deep_link(response.get("deepLink")),
            "marker": f"[{marker}]",
        }
        return success_response([item])
    except Exception as exc:
        result = tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


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
            result = success_response([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "GUIDE", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "title": compact_text(item.get("title"), max_chars=120),
                "summary": compact_text(item.get("summary"), max_chars=500),
                "durationDays": item.get("durationDays"),
                "routeSummary": compact_text(item.get("routeSummary"), max_chars=800),
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return success_response(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


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
            result = success_response([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(context, "FOOD", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": compact_text(item.get("name"), max_chars=100),
                "summary": compact_text(item.get("summary"), max_chars=500),
                "categoryName": compact_text(item.get("categoryName"), max_chars=100),
                "recommendedSpots": item.get("recommendedSpots", [])[:10]
                if isinstance(item.get("recommendedSpots"), list)
                else [],
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return success_response(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


class RecommendationInput(BaseModel):
    scene: Literal["HOME", "HALF_DAY", "ONE_DAY", "NEARBY"] = Field(description="推荐场景")
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
            result = success_response([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = register_records(request_context, "RECOMMENDATION", records[:limit])
        items = [
            {
                "id": str(item.get("id") or item.get("spotId") or ""),
                "name": compact_text(item.get("name") or item.get("title"), max_chars=100),
                "reason": compact_text(item.get("reason") or item.get("summary"), max_chars=500),
                "rank": index + 1,
                "deepLink": safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for index, item in enumerate(enriched)
        ]
        return success_response(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = tool_error(exc)
        request_context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result
