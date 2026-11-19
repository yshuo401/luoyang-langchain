"""Knowledge retrieval tool that wraps the RAG service, never Milvus SDK."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools._common import begin_tool, success, text, tool_error


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="历史文化或旅游知识问题")
    top_k: int = Field(default=5, ge=1, le=8, description="最多检索分片数")


@tool("retrieve_tourism_knowledge", args_schema=KnowledgeSearchInput)
async def retrieve_tourism_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """从 Milvus 知识库检索景点讲解、历史文化和游玩建议。"""

    context, error = begin_tool("retrieve_tourism_knowledge")
    if error is not None:
        return error
    assert context is not None
    if context.rag_service is None:
        return {
            "ok": False,
            "errorCode": "MILVUS_UNAVAILABLE",
            "message": "知识库暂时不可用",
            "retryable": True,
        }
    try:
        documents = await context.rag_service.retrieve(query, top_k=top_k)
        if not documents:
            result = success([])
            result["meta"]["reason"] = "RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD"
            return result
        items: list[dict[str, Any]] = []
        for document in documents:
            marker = context.evidence.register_document(document)
            items.append(
                {
                    "marker": f"[{marker}]",
                    "title": text(document.metadata.get("title"), max_chars=120),
                    "content": text(document.page_content, max_chars=1200),
                    "page": document.metadata.get("page"),
                    "sectionPath": document.metadata.get("section_path"),
                    "score": document.metadata.get("_normalized_score"),
                }
            )
        return success(items)
    except Exception as exc:
        result = tool_error(exc)
        if "errorCode" in result:
            context.record_error(str(result["errorCode"]))
        return result
