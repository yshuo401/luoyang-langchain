from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from app.errors import ConflictError
from app.indexer import DownloadedFile, IndexJobManager, IndexPipeline
from app.schemas import IndexJobSubmitRequest, KnowledgeChunkBatch


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, request: IndexJobSubmitRequest, progress: Any) -> int:
        del request
        self.calls += 1
        await progress(50, 2)
        await asyncio.sleep(0.01)
        return 2


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

    def delete(self, **_: Any) -> None:
        return None

    def add_documents(self, documents: list[Any], ids: list[str]) -> None:
        del ids
        self.documents = documents

    def get_pks(self, expr: str) -> list[str]:
        del expr
        return []


class FakeSpringBootClient:
    def __init__(self) -> None:
        self.batches: list[KnowledgeChunkBatch] = []

    async def upsert_knowledge_chunks(
        self,
        batch: KnowledgeChunkBatch,
        *,
        idempotency_key: str,
    ) -> None:
        del idempotency_key
        self.batches.append(batch)


def _request(
    *,
    version: int = 1,
    key: str = "18:1:FULL_INDEX",
    title: str = "龙门石窟讲解",
) -> IndexJobSubmitRequest:
    return IndexJobSubmitRequest.model_validate(
        {
            "documentId": "18",
            "documentVersion": version,
            "fileId": "88",
            "downloadUrl": "https://example.com/guide.txt",
            "docType": 1,
            "title": title,
            "category": "SCENIC_INTRO",
            "fileName": "guide.txt",
            "idempotencyKey": key,
        }
    )


@pytest.mark.asyncio
async def test_index_job_is_idempotent_and_reaches_success() -> None:
    pipeline = FakePipeline()
    manager = IndexJobManager(pipeline)  # type: ignore[arg-type]
    try:
        first = await manager.submit(_request())
        second = await manager.submit(_request())
        assert first.external_job_id == second.external_job_id

        current = None
        for _ in range(50):
            current = await manager.get(first.external_job_id)
            if current and current.status == "SUCCEEDED":
                break
            await asyncio.sleep(0.01)

        assert current is not None
        assert current.status == "SUCCEEDED"
        assert current.chunk_count == 2
        assert pipeline.calls == 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_same_key_with_different_payload_conflicts() -> None:
    manager = IndexJobManager(FakePipeline())  # type: ignore[arg-type]
    try:
        await manager.submit(_request())
        with pytest.raises(ConflictError):
            await manager.submit(_request(title="另一个标题"))
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_pipeline_builds_vector_ids_and_callback_batches() -> None:
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
