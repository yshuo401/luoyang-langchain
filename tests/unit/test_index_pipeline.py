from __future__ import annotations

import hashlib
from typing import Any

import pytest
from app.ingestion.pipeline import IndexPipeline
from app.integrations.file_downloader import DownloadedFile
from app.schemas.index import IndexJobSubmitRequest, KnowledgeChunkBatch


class FakeDownloader:
    async def download(self, url: str, *, file_name: str | None = None) -> DownloadedFile:
        del url
        return DownloadedFile(
            content="# 龙门石窟\n\n建议游玩三到四小时。".encode(),
            content_type="text/markdown",
            file_name=file_name or "guide.md",
        )

    async def aclose(self) -> None:
        return None


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[Any] = []

    async def upsert_documents(self, documents: list[Any]) -> list[str]:
        self.documents = documents
        return [str(document.metadata["vector_id"]) for document in documents]

    async def delete_document(self, document_id: str) -> int:
        del document_id
        return 0

    async def delete_document_versions_except(self, document_id: str, version: int) -> int:
        del document_id, version
        return 0


class FakeSpringBootClient:
    def __init__(self) -> None:
        self.batches: list[KnowledgeChunkBatch] = []

    async def upsert_knowledge_chunks(
        self, batch: KnowledgeChunkBatch, *, idempotency_key: str
    ) -> None:
        del idempotency_key
        self.batches.append(batch)


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


@pytest.mark.asyncio
async def test_pipeline_builds_stable_vectors_and_callback_batches() -> None:
    vector_store = FakeVectorStore()
    client = FakeSpringBootClient()
    pipeline = IndexPipeline(
        vector_store=vector_store,  # type: ignore[arg-type]
        springboot_client=client,  # type: ignore[arg-type]
        downloader=FakeDownloader(),  # type: ignore[arg-type]
    )
    progress_values: list[int] = []

    async def progress(value: int, chunk_count: int = 0) -> None:
        del chunk_count
        progress_values.append(value)

    count = await pipeline.run(_request(), progress)
    assert count == len(vector_store.documents) >= 1
    first = vector_store.documents[0]
    expected = hashlib.sha256(b"18:1:0").hexdigest()
    assert first.metadata["vector_id"] == expected
    assert first.metadata["content_hash"]
    assert client.batches[0].chunks[0].vector_id == expected
    assert progress_values[-1] == 95

