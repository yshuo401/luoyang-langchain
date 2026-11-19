"""Milvus retriever with deterministic score and visibility filtering."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.core.config import Settings, get_settings
from app.rag.vector_store import KnowledgeVectorStore


class ThresholdRetriever(BaseRetriever):
    """Return only enabled knowledge chunks above the configured threshold."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: KnowledgeVectorStore
    k: int = 5
    min_score: float = 0.0
    metric_type: str = "COSINE"
    search_expr: str | None = "enabled == true"

    def _normalize(self, score: float) -> float:
        if self.metric_type == "L2":
            return max(0.0, min(1.0, 1.0 / (1.0 + max(score, 0.0))))
        return max(0.0, min(1.0, float(score)))

    def _is_relevant(self, score: float) -> bool:
        if self.metric_type == "L2":
            return score <= self.min_score if self.min_score > 0 else True
        return score >= self.min_score

    def _filter(self, pairs: list[tuple[Document, float]]) -> list[Document]:
        documents: list[Document] = []
        for document, score in pairs:
            if not self._is_relevant(float(score)):
                continue
            rank_no = len(documents) + 1
            document.metadata = {
                **document.metadata,
                "_score": float(score),
                "_normalized_score": self._normalize(float(score)),
                "_source_type": "KNOWLEDGE",
                "rank_no": rank_no,
                "citation_marker": f"[S{rank_no}]",
            }
            documents.append(document)
        return documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        raw: Any = self.vector_store.raw()
        pairs = raw.similarity_search_with_score(query, self.k, expr=self.search_expr)
        return self._filter(pairs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        pairs = await self.vector_store.similarity_search_with_score(
            query,
            k=self.k,
            expr=self.search_expr,
        )
        return self._filter(pairs)


def create_retriever(
    vector_store: KnowledgeVectorStore,
    *,
    settings: Settings | None = None,
    k: int | None = None,
) -> ThresholdRetriever:
    config = settings or get_settings()
    return ThresholdRetriever(
        vector_store=vector_store,
        k=k or config.rag_top_k,
        min_score=config.rag_min_score,
        metric_type=config.milvus_metric_type,
    )





