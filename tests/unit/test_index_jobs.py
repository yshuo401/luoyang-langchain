from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.core.errors import ConflictError
from app.ingestion.jobs import IndexJobManager, MemoryIndexJobStore
from app.schemas.index import IndexJobSubmitRequest


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, request: IndexJobSubmitRequest, progress: Any) -> int:
        self.calls += 1
        await progress(50, 2)
        await asyncio.sleep(0.01)
        return 2


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
    manager = IndexJobManager(MemoryIndexJobStore(), pipeline)
    try:
        first = await manager.submit(_request())
        second = await manager.submit(_request())
        assert first.external_job_id == second.external_job_id
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
    manager = IndexJobManager(MemoryIndexJobStore(), FakePipeline())
    try:
        await manager.submit(_request())
        with pytest.raises(ConflictError):
            await manager.submit(_request(title="另一个标题"))
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_old_document_version_cannot_overwrite_newer() -> None:
    manager = IndexJobManager(MemoryIndexJobStore(), FakePipeline())
    try:
        newest = await manager.submit(_request(version=2, key="18:2:FULL_INDEX"))
        for _ in range(50):
            current = await manager.get(newest.external_job_id)
            if current and current.status == "SUCCEEDED":
                break
            await asyncio.sleep(0.01)
        with pytest.raises(ConflictError):
            await manager.submit(_request(version=1, key="18:1:FULL_INDEX"))
    finally:
        await manager.close()
