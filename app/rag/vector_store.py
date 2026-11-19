"""Milvus vector-store operations kept behind one small interface."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.core.config import Settings, get_settings
from app.core.errors import MilvusUnavailableError


class KnowledgeVectorStore:
    def __init__(self, store: Milvus, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings or get_settings()

    async def upsert_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        ids = [str(doc.metadata["vector_id"]) for doc in documents]
        try:
            await asyncio.to_thread(self._store.delete, ids=ids)
            await asyncio.to_thread(self._store.add_documents, documents, ids=ids)
        except Exception as exc:
            raise MilvusUnavailableError("知识库写入失败") from exc
        return ids

    async def similarity_search_with_score(
        self, query: str, *, k: int, expr: str | None = None
    ) -> list[tuple[Document, float]]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._store.similarity_search_with_score,
                    query,
                    k,
                    expr=expr,
                ),
                timeout=self._settings.milvus_search_timeout_seconds,
            )
        except TimeoutError as exc:
            raise MilvusUnavailableError("知识库检索超时") from exc
        except Exception as exc:
            raise MilvusUnavailableError("知识库检索失败") from exc

    async def delete_document(self, document_id: str) -> int:
        safe_id = document_id.replace('"', '\\"')
        expr = f'document_id == "{safe_id}"'
        try:
            existing = await asyncio.to_thread(self._store.get_pks, expr)
            await asyncio.to_thread(self._store.delete, expr=expr)
            return len(existing or [])
        except Exception as exc:
            raise MilvusUnavailableError("知识库删除失败") from exc

    def raw(self) -> Any:
        return self._store
