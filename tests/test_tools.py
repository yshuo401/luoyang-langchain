from __future__ import annotations

from typing import Any

import pytest
from app.tools import (
    RequestContext,
    reset_request_context,
    retrieve_tourism_knowledge,
    search_scenic_spots,
    set_request_context,
)
from langchain_core.documents import Document


class FakeSpringBootClient:
    async def search_spots(self, **_: Any) -> dict[str, Any]:
        return {
            "records": [
                {
                    "id": 1001,
                    "name": "龙门石窟",
                    "summary": "世界文化遗产",
                    "categoryName": "历史人文",
                    "address": "洛阳市洛龙区",
                    "openTimeDesc": "08:00-18:00",
                    "tags": ["世界遗产"],
                    "deepLink": "/pages/spot/detail?id=1001",
                }
            ]
        }


class FakeRAGService:
    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        del query, top_k
        return [
            Document(
                page_content="建议游玩三到四小时。",
                metadata={
                    "document_id": "18",
                    "vector_id": "vec-18-0",
                    "title": "龙门石窟讲解",
                    "_normalized_score": 0.88,
                },
            )
        ]


def _context(*, max_tool_calls: int = 6) -> RequestContext:
    return RequestContext(
        user_id="10001",
        thread_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        springboot_client=FakeSpringBootClient(),
        rag_service=FakeRAGService(),
        max_tool_calls=max_tool_calls,
    )


@pytest.mark.asyncio
async def test_spot_tool_registers_source_and_returns_marker() -> None:
    context = _context()
    token = set_request_context(context)
    try:
        result = await search_scenic_spots.ainvoke({"keyword": "龙门石窟", "limit": 5})
    finally:
        reset_request_context(token)

    assert result["ok"] is True
    assert result["data"][0]["marker"] == "[S1]"
    assert context.sources[0].source_type == "SPOT"


@pytest.mark.asyncio
async def test_knowledge_tool_registers_retrieved_documents() -> None:
    context = _context()
    token = set_request_context(context)
    try:
        result = await retrieve_tourism_knowledge.ainvoke({"query": "龙门石窟游玩多久", "top_k": 5})
    finally:
        reset_request_context(token)

    assert result["ok"] is True
    assert result["data"][0]["marker"] == "[S1]"
    assert context.sources[0].score == 0.88


@pytest.mark.asyncio
async def test_tool_budget_is_enforced() -> None:
    context = _context(max_tool_calls=1)
    token = set_request_context(context)
    try:
        first = await search_scenic_spots.ainvoke({"keyword": "龙门石窟"})
        second = await search_scenic_spots.ainvoke({"keyword": "白马寺"})
    finally:
        reset_request_context(token)

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["errorCode"] == "TOOL_BUDGET_EXCEEDED"
