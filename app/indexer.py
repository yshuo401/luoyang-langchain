"""知识库索引：下载 -> 解析 -> 切分 -> Milvus -> 回写 Java。

索引任务仍然使用进程内 `asyncio`，不做 Redis、消息队列或数据库任务表。
这对单机答辩足够，而且能清楚展示一条异步流水线如何运行。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.config import Settings, get_settings, new_ulid
from app.errors import ConflictError, MilvusUnavailableError
from app.rag import DocumentParseError, extract_markdown, split_markdown
from app.schemas import (
    IndexJobSubmitRequest,
    IndexJobSubmitResponse,
    IndexJobView,
    KnowledgeChunkBatch,
    KnowledgeChunkPayload,
)
from app.springboot import SpringBootClient

ProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(slots=True)
class DownloadedFile:
    content: bytes
    content_type: str | None
    file_name: str | None


class DownloadError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentDownloader:
    """下载 OSS 临时签名地址，限制最大文件大小。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=20.0, write=10.0, pool=3.0),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def download(self, url: str, *, file_name: str | None = None) -> DownloadedFile:
        chunks: list[bytes] = []
        total = 0
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise DownloadError("DOWNLOAD_HTTP_ERROR", "文档下载失败")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.settings.download_max_bytes:
                        raise DownloadError("DOCUMENT_TOO_LARGE", "文档超过允许大小")
                    chunks.append(chunk)
                return DownloadedFile(
                    content=b"".join(chunks),
                    content_type=response.headers.get("content-type"),
                    file_name=file_name
                    or self._name_from_disposition(response.headers.get("content-disposition")),
                )
        except httpx.HTTPError as exc:
            raise DownloadError("DOWNLOAD_ERROR", "文档下载失败") from exc

    @staticmethod
    def _name_from_disposition(value: str | None) -> str | None:
        if not value:
            return None
        for part in value.split(";"):
            key, _, raw = part.strip().partition("=")
            if key.lower() == "filename":
                return raw.strip('"') or None
        return None


class IndexPipeline:
    """真正执行一次索引的流水线。"""

    def __init__(
        self,
        *,
        vector_store: Milvus,
        springboot_client: SpringBootClient,
        downloader: DocumentDownloader,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.springboot_client = springboot_client
        self.downloader = downloader
        self.settings = settings or get_settings()

    async def run(self, request: IndexJobSubmitRequest, progress: ProgressCallback) -> int:
        file_name = request.file_name or self._file_name_from_url(str(request.download_url))
        await progress(10, 0)

        downloaded = await self.downloader.download(
            str(request.download_url),
            file_name=file_name,
        )
        await progress(25, 0)

        # MinerU 是同步 SDK，放到线程中，避免卡住 FastAPI 事件循环。
        markdown = await asyncio.to_thread(
            extract_markdown,
            downloaded.content,
            downloaded.file_name or file_name,
            self.settings,
        )
        documents = split_markdown(markdown, request, settings=self.settings)
        for document in documents:
            vector_id = self._vector_id(
                request.document_id,
                request.document_version,
                int(document.metadata["chunk_index"]),
            )
            content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
            document.metadata.update(
                {
                    "vector_id": vector_id,
                    "content_hash": content_hash,
                    "file_id": request.file_id,
                    "doc_type": request.doc_type,
                }
            )

        await progress(40, len(documents))
        await self._upsert(documents)
        await progress(75, len(documents))
        await self._notify_chunks(request, documents)
        await self._delete_old_versions(request.document_id, request.document_version)
        await progress(95, len(documents))
        return len(documents)

    async def _upsert(self, documents: list[Document]) -> None:
        ids = [str(document.metadata["vector_id"]) for document in documents]
        try:
            # 第一次索引时 collection 还不存在，delete 可能报“collection not found”。
            # 这种情况直接交给 add_documents 创建 collection；其他删除错误仍然抛出。
            try:
                await asyncio.to_thread(self.vector_store.delete, ids=ids)
            except Exception as exc:
                if "not found" not in str(exc).lower() and "not exist" not in str(exc).lower():
                    raise
            await asyncio.to_thread(self.vector_store.add_documents, documents, ids=ids)
        except Exception as exc:
            raise MilvusUnavailableError("知识库写入失败") from exc

    async def _delete_old_versions(self, document_id: str, version: int) -> int:
        safe_id = document_id.replace('"', '\\"')
        expr = f'document_id == "{safe_id}" and document_version != {int(version)}'
        try:
            existing = await asyncio.to_thread(self.vector_store.get_pks, expr)
            await asyncio.to_thread(self.vector_store.delete, expr=expr)
            return len(existing or [])
        except Exception as exc:
            raise MilvusUnavailableError("旧版本知识清理失败") from exc

    async def delete_document(self, document_id: str) -> int:
        safe_id = document_id.replace('"', '\\"')
        expr = f'document_id == "{safe_id}"'
        try:
            existing = await asyncio.to_thread(self.vector_store.get_pks, expr)
            await asyncio.to_thread(self.vector_store.delete, expr=expr)
            return len(existing or [])
        except Exception as exc:
            raise MilvusUnavailableError("知识库删除失败") from exc

    async def _notify_chunks(
        self,
        request: IndexJobSubmitRequest,
        documents: list[Document],
    ) -> None:
        batch_size = self.settings.index_callback_batch_size
        for start in range(0, len(documents), batch_size):
            items = documents[start : start + batch_size]
            batch = KnowledgeChunkBatch(
                document_id=request.document_id,
                document_version=request.document_version,
                chunks=[self._chunk_payload(document) for document in items],
            )
            await self.springboot_client.upsert_knowledge_chunks(
                batch,
                idempotency_key=f"{request.idempotency_key}:chunks:{start}",
            )

    @staticmethod
    def _chunk_payload(document: Document) -> KnowledgeChunkPayload:
        metadata = document.metadata
        return KnowledgeChunkPayload(
            chunk_index=int(metadata["chunk_index"]),
            vector_id=str(metadata["vector_id"]),
            content=document.page_content,
            content_hash=str(metadata["content_hash"]),
            char_count=len(document.page_content),
            metadata={
                "page": metadata.get("page"),
                "sectionPath": metadata.get("section_path"),
                "title": metadata.get("title"),
            },
        )

    @staticmethod
    def _vector_id(document_id: str, version: int, chunk_index: int) -> str:
        return hashlib.sha256(f"{document_id}:{version}:{chunk_index}".encode()).hexdigest()

    @staticmethod
    def _file_name_from_url(url: str) -> str:
        name = PurePosixPath(urlparse(url).path).name
        if not name or "." not in name:
            raise DocumentParseError("UNSUPPORTED_FILE_TYPE", "无法识别文档文件名")
        return name

    async def aclose(self) -> None:
        await self.downloader.aclose()


class IndexJobManager:
    """进程内索引任务表。

    只保存任务状态和幂等键，不把任务写入 MySQL。服务重启后任务丢失，
    重新上传/重新提交即可，这是单机毕设的明确取舍。
    """

    def __init__(self, pipeline: IndexPipeline, settings: Settings | None = None) -> None:
        self.pipeline = pipeline
        self.settings = settings or get_settings()
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


class IndexService:
    """API 层使用的索引服务门面。"""

    def __init__(self, manager: IndexJobManager, pipeline: IndexPipeline) -> None:
        self.manager = manager
        self.pipeline = pipeline

    async def submit(self, request: IndexJobSubmitRequest) -> IndexJobSubmitResponse:
        job = await self.manager.submit(request)
        return IndexJobSubmitResponse(
            job_id=job.job_id,
            external_job_id=job.external_job_id,
            status=job.status,
        )

    async def get_job(self, job_id: str) -> IndexJobView | None:
        return await self.manager.get(job_id)

    async def delete_document(self, document_id: str) -> int:
        return await self.pipeline.delete_document(document_id)

    async def close(self) -> None:
        await self.manager.close()
        await self.pipeline.aclose()
