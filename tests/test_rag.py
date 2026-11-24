from __future__ import annotations

from typing import Any

import pytest
from app.rag import (
    DocumentParseError,
    RAGService,
    create_rag_chain,
    extract_markdown,
    split_markdown,
)
from app.schemas import IndexJobSubmitRequest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        return self.documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        return self.documents


class FakeMilvus:
    def as_retriever(self, **_: Any) -> StaticRetriever:
        return StaticRetriever(
            documents=[
                Document(
                    page_content="建议游玩三到四小时。",
                    metadata={"title": "龙门石窟", "citation_marker": "[S1]"},
                )
            ]
        )

    def similarity_search_with_score(
        self, query: str, k: int, *, expr: str
    ) -> list[tuple[Document, float]]:
        del query, k, expr
        return [
            (
                Document(
                    page_content="建议游玩三到四小时。",
                    metadata={"document_id": "18", "vector_id": "vec-18-0", "title": "龙门石窟"},
                ),
                0.88,
            )
        ]


def _request() -> IndexJobSubmitRequest:
    return IndexJobSubmitRequest.model_validate(
        {
            "documentId": "18",
            "documentVersion": 1,
            "fileId": "88",
            "downloadUrl": "https://example.com/guide.md",
            "docType": 1,
            "title": "龙门石窟讲解",
            "category": "SCENIC_INTRO",
            "fileName": "guide.md",
            "idempotencyKey": "18:1:FULL_INDEX",
        }
    )


def test_text_and_markdown_are_loaded_without_mineru() -> None:
    markdown = extract_markdown("# 龙门石窟\n\n建议游玩三到四小时。".encode(), "guide.md")
    assert "龙门石窟" in markdown
    assert "三到四小时" in markdown

    clean = extract_markdown("洛阳龙门石窟\x00位于洛龙区。".encode(), "guide.txt")
    assert "\x00" not in clean


def test_split_markdown_keeps_heading_and_chunk_metadata() -> None:
    documents = split_markdown("# 龙门石窟\n\n## 游览建议\n\n建议游玩三到四小时。", _request())
    assert documents
    assert documents[0].metadata["document_id"] == "18"
    assert documents[0].metadata["chunk_index"] == 0
    assert documents[0].metadata["enabled"] is True
    assert "游览建议" in documents[0].metadata["section_path"]


def test_unsupported_document_is_rejected() -> None:
    with pytest.raises(DocumentParseError) as error:
        extract_markdown(b"abc", "guide.exe")
    assert error.value.code == "UNSUPPORTED_FILE_TYPE"


def test_rag_chain_uses_official_retrieval_chain_api() -> None:
    documents = [Document(page_content="建议游玩三到四小时。", metadata={"title": "龙门石窟"})]
    chain = create_rag_chain(
        StaticRetriever(documents=documents),
        FakeListChatModel(responses=["建议游玩三到四小时。[S1]"]),
    )
    result = chain.invoke({"input": "龙门石窟适合玩多久？"})
    assert result["answer"] == "建议游玩三到四小时。[S1]"
    assert result["context"][0].metadata["title"] == "龙门石窟"


@pytest.mark.asyncio
async def test_rag_service_adds_source_order_and_score() -> None:
    service = RAGService(
        vector_store=FakeMilvus(),  # type: ignore[arg-type]
        chat_model=FakeListChatModel(responses=["ok"]),
    )
    documents = await service.retrieve("龙门石窟好玩吗", top_k=3)
    assert documents
    assert documents[0].metadata["rank_no"] == 1
    assert documents[0].metadata["citation_marker"] == "[S1]"
    assert documents[0].metadata["_normalized_score"] == 0.88
