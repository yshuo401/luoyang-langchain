from __future__ import annotations

import math
import os
import uuid

import pytest
from app.config import get_settings
from app.rag import create_vector_store
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INFRA_TESTS") != "1",
        reason="设置 RUN_INFRA_TESTS=1 后才连接 Milvus",
    ),
]


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        values = [0.0] * 16
        for index, char in enumerate(text):
            values[index % 16] += float(ord(char) % 31 + 1)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_milvus_vector_round_trip() -> None:
    settings = get_settings()
    collection = f"luoyang_ai_test_{uuid.uuid4().hex[:10]}"
    test_settings = settings.model_copy(update={"milvus_collection": collection})
    store = create_vector_store(DeterministicEmbeddings(), settings=test_settings)
    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
    )
    try:
        documents = [
            Document(
                page_content="龙门石窟建议游玩三到四小时。",
                metadata={
                    "vector_id": "vec-1",
                    "document_id": "doc-1",
                    "document_version": 1,
                    "chunk_index": 0,
                    "title": "龙门石窟",
                    "enabled": True,
                },
            ),
            Document(
                page_content="已停用的旧资料。",
                metadata={
                    "vector_id": "vec-2",
                    "document_id": "doc-1",
                    "document_version": 0,
                    "chunk_index": 1,
                    "title": "旧资料",
                    "enabled": False,
                },
            ),
        ]
        ids = [str(document.metadata["vector_id"]) for document in documents]
        store.delete(ids=ids)
        store.add_documents(documents, ids=ids)
        results = store.similarity_search_with_score(
            "龙门石窟游玩时间",
            5,
            expr='document_id == "doc-1" and enabled == true',
        )
        assert results
        assert all(document.metadata["document_version"] == 1 for document, _ in results)
    finally:
        client.drop_collection(collection)
        client.close()
