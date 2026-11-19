"""Index-job state storage, with Redis for runtime and memory for tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.errors import ConflictError
from app.core.ids import new_ulid
from app.schemas.index import IndexJobSubmitRequest, IndexJobView


@dataclass(slots=True)
class IdempotencyRecord:
    external_job_id: str
    fingerprint: str


class IndexJobStore(Protocol):
    async def get(self, external_job_id: str) -> IndexJobView | None: ...

    async def save(self, job: IndexJobView) -> None: ...

    async def get_idempotency(self, key: str) -> IdempotencyRecord | None: ...

    async def set_idempotency(self, key: str, record: IdempotencyRecord) -> None: ...

    async def get_latest_version(self, document_id: str) -> int | None: ...

    async def set_latest_version(self, document_id: str, version: int) -> None: ...

    async def close(self) -> None: ...


def _key_fragment(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MemoryIndexJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IndexJobView] = {}
        self._idempotency: dict[str, IdempotencyRecord] = {}
        self._versions: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get(self, external_job_id: str) -> IndexJobView | None:
        async with self._lock:
            return self._jobs.get(external_job_id)

    async def save(self, job: IndexJobView) -> None:
        async with self._lock:
            self._jobs[job.external_job_id] = job

    async def get_idempotency(self, key: str) -> IdempotencyRecord | None:
        async with self._lock:
            return self._idempotency.get(key)

    async def set_idempotency(self, key: str, record: IdempotencyRecord) -> None:
        async with self._lock:
            self._idempotency[key] = record

    async def get_latest_version(self, document_id: str) -> int | None:
        async with self._lock:
            return self._versions.get(document_id)

    async def set_latest_version(self, document_id: str, version: int) -> None:
        async with self._lock:
            current = self._versions.get(document_id, 0)
            self._versions[document_id] = max(current, version)

    async def close(self) -> None:
        return None

class RedisIndexJobStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._redis = Redis.from_url(
            self._settings.redis_url,
            socket_connect_timeout=self._settings.redis_socket_timeout_seconds,
            socket_timeout=self._settings.redis_socket_timeout_seconds,
        )

    @staticmethod
    def _job_key(external_job_id: str) -> str:
        return f"ai:index:job:{external_job_id}"

    @staticmethod
    def _idem_key(key: str) -> str:
        return f"ai:index:idem:{_key_fragment(key)}"

    @staticmethod
    def _version_key(document_id: str) -> str:
        return f"ai:index:latest_version:{document_id}"

    async def get(self, external_job_id: str) -> IndexJobView | None:
        raw = await self._redis.get(self._job_key(external_job_id))
        if raw is None:
            return None
        return IndexJobView.model_validate_json(raw)

    async def save(self, job: IndexJobView) -> None:
        await self._redis.set(
            self._job_key(job.external_job_id),
            job.model_dump_json(by_alias=True),
            ex=self._settings.index_job_ttl_seconds,
        )

    async def get_idempotency(self, key: str) -> IdempotencyRecord | None:
        raw = await self._redis.get(self._idem_key(key))
        if raw is None:
            return None
        data = json.loads(raw)
        return IdempotencyRecord(
            external_job_id=str(data["externalJobId"]),
            fingerprint=str(data["fingerprint"]),
        )

    async def set_idempotency(self, key: str, record: IdempotencyRecord) -> None:
        await self._redis.set(
            self._idem_key(key),
            json.dumps(
                {"externalJobId": record.external_job_id, "fingerprint": record.fingerprint},
                separators=(",", ":"),
            ),
            ex=self._settings.index_job_ttl_seconds,
        )

    async def get_latest_version(self, document_id: str) -> int | None:
        raw = await self._redis.get(self._version_key(document_id))
        return int(raw) if raw is not None else None

    async def set_latest_version(self, document_id: str, version: int) -> None:
        key = self._version_key(document_id)
        current = await self.get_latest_version(document_id)
        if current is None or version > current:
            await self._redis.set(key, str(version), ex=self._settings.index_job_ttl_seconds)

    async def close(self) -> None:
        await self._redis.aclose()


class IndexJobManager:
    """Create, persist and execute bounded asyncio indexing jobs."""

    def __init__(
        self,
        store: IndexJobStore,
        pipeline: object,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._pipeline = pipeline
        self._settings = settings or get_settings()
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    async def submit(self, request: IndexJobSubmitRequest) -> IndexJobView:
        key = request.idempotency_key
        fingerprint = self._fingerprint(request)
        async with self._lock:
            existing = await self._store.get_idempotency(key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise ConflictError("相同幂等键对应不同索引参数")
                job = await self._store.get(existing.external_job_id)
                if job is not None:
                    return job
            latest = await self._store.get_latest_version(request.document_id)
            if latest is not None and request.document_version < latest:
                raise ConflictError("旧版本文档不能覆盖新版本")
            external_id = new_ulid()
            now = datetime.now(UTC)
            job = IndexJobView(
                job_id=external_id,
                external_job_id=external_id,
                document_id=request.document_id,
                document_version=request.document_version,
                status="QUEUED",
                progress=0,
                created_at=now,
                updated_at=now,
            )
            await self._store.save(job)
            await self._store.set_idempotency(
                key,
                IdempotencyRecord(external_job_id=external_id, fingerprint=fingerprint),
            )
            task = asyncio.create_task(self._run(job, request))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return job

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

    async def _run(self, job: IndexJobView, request: IndexJobSubmitRequest) -> None:
        async with self._run_lock:
            await self._run_locked(job, request)

    async def _run_locked(self, job: IndexJobView, request: IndexJobSubmitRequest) -> None:
        job = job.model_copy(
            update={"status": "RUNNING", "progress": 5, "updated_at": datetime.now(UTC)}
        )
        await self._store.save(job)
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
                await self._store.save(job)

            chunk_count = await self._pipeline.run(request, progress)  # type: ignore[attr-defined]
            await self._store.set_latest_version(request.document_id, request.document_version)
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
        await self._store.save(job)

    async def get(self, external_job_id: str) -> IndexJobView | None:
        return await self._store.get(external_job_id)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._store.close()



