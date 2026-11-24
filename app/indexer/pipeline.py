"""一次知识库索引任务的实际处理流水线。

流水线只负责一件事：把已经下载好的文档变成 Milvus 向量，并把可展示的
分片信息回写给 Spring Boot。任务排队、状态和幂等由 `jobs.py` 负责。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.config import Settings, get_settings
from app.errors import MilvusUnavailableError
from app.indexer.downloader import DocumentDownloader
from app.rag.documents import DocumentParseError, extract_markdown, split_markdown
from app.schemas import (
    IndexJobSubmitRequest,
    KnowledgeChunkBatch,
    KnowledgeChunkPayload,
)
from app.springboot import SpringBootClient

ProgressCallback = Callable[[int, int], Awaitable[None]]


class IndexPipeline:
    """下载 -> 解析 -> 切分 -> 写 Milvus -> 回写 Java 分片。"""

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
        """同一文档版本和分片序号始终得到同一个 Milvus 主键。"""

        return hashlib.sha256(f"{document_id}:{version}:{chunk_index}".encode()).hexdigest()

    @staticmethod
    def _file_name_from_url(url: str) -> str:
        name = PurePosixPath(urlparse(url).path).name
        if not name or "." not in name:
            raise DocumentParseError("UNSUPPORTED_FILE_TYPE", "无法识别文档文件名")
        return name

    async def aclose(self) -> None:
        await self.downloader.aclose()
