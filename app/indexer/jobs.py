"""进程内索引任务表。

这张表只存在于当前 Python 进程里，服务重启后会清空。毕业设计采用这个
取舍，是为了把精力放在“索引流水线”本身，而不是引入 Redis 队列或任务表。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime

from app.config import new_ulid
from app.errors import ConflictError
from app.indexer.pipeline import IndexPipeline
from app.schemas import IndexJobSubmitRequest, IndexJobView


class IndexJobManager:
    """维护任务状态、幂等键，并串行调度索引流水线。"""

    def __init__(self, pipeline: IndexPipeline) -> None:
        self.pipeline = pipeline
        self._jobs: dict[str, IndexJobView] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._latest_versions: dict[str, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    async def submit(self, request: IndexJobSubmitRequest) -> IndexJobView:
        fingerprint = self._fingerprint(request)
        async with self._lock:
            existing = self._idempotency.get(request.idempotency_key)
            if existing is not None:
                old_fingerprint, old_job_id = existing
                if old_fingerprint != fingerprint:
                    raise ConflictError("相同幂等键对应不同索引参数")
                old_job = self._jobs.get(old_job_id)
                if old_job is not None:
                    return old_job

            latest = self._latest_versions.get(request.document_id)
            if latest is not None and request.document_version < latest:
                raise ConflictError("旧版本文档不能覆盖新版本")

            job_id = new_ulid()
            now = datetime.now(UTC)
            job = IndexJobView(
                job_id=job_id,
                external_job_id=job_id,
                document_id=request.document_id,
                document_version=request.document_version,
                status="QUEUED",
                progress=0,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._idempotency[request.idempotency_key] = (fingerprint, job_id)
            task = asyncio.create_task(self._run(job, request))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return job

    async def get(self, job_id: str) -> IndexJobView | None:
        return self._jobs.get(job_id)

    async def _run(self, job: IndexJobView, request: IndexJobSubmitRequest) -> None:
        # 单机演示串行执行任务，避免同时写 Milvus 造成难以解释的竞争问题。
        async with self._run_lock:
            await self._run_locked(job, request)

    async def _run_locked(self, job: IndexJobView, request: IndexJobSubmitRequest) -> None:
        job = job.model_copy(
            update={"status": "RUNNING", "progress": 5, "updated_at": datetime.now(UTC)}
        )
        self._jobs[job.external_job_id] = job
        try:

            async def progress(value: int, chunk_count: int = 0) -> None:
                nonlocal job
                job = job.model_copy(
                    update={
                        "progress": max(0, min(99, value)),
                        "chunk_count": chunk_count,
                        "updated_at": datetime.now(UTC),
                    }
                )
                self._jobs[job.external_job_id] = job

            chunk_count = await self.pipeline.run(request, progress)
            self._latest_versions[request.document_id] = max(
                request.document_version,
                self._latest_versions.get(request.document_id, 0),
            )
            job = job.model_copy(
                update={
                    "status": "SUCCEEDED",
                    "progress": 100,
                    "chunk_count": chunk_count,
                    "updated_at": datetime.now(UTC),
                }
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or "INDEX_INTERNAL_ERROR"
            message = getattr(exc, "message", None) or "索引任务失败"
            job = job.model_copy(
                update={
                    "status": "FAILED",
                    "error_code": str(code),
                    "error_message": str(message)[:300],
                    "updated_at": datetime.now(UTC),
                }
            )
        self._jobs[job.external_job_id] = job

    @staticmethod
    def _fingerprint(request: IndexJobSubmitRequest) -> str:
        values = {
            "documentId": request.document_id,
            "documentVersion": request.document_version,
            "fileId": request.file_id,
            "docType": request.doc_type,
            "title": request.title,
            "category": request.category,
        }
        return hashlib.sha256(
            json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
