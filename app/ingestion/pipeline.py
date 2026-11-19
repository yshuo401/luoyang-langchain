"""Parse -> split -> embed -> Milvus -> Java chunk-metadata callback."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from urllib.parse import urlparse

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.integrations.file_downloader import DocumentDownloader
from app.integrations.springboot_client import SpringBootClient
from app.rag.loaders import DocumentParseError, parse_document
from app.rag.splitter import ChunkingOptions, split_document
from app.rag.vector_store import KnowledgeVectorStore
from app.schemas.index import (
    IndexJobSubmitRequest,
    KnowledgeChunkBatch,
    KnowledgeChunkPayload,
)

ProgressCallback = Callable[[int, int], Awaitable[None]]


class IndexPipeline:
    def __init__(
        self,
        *,
        vector_store: KnowledgeVectorStore,
        springboot_client: SpringBootClient,
        downloader: DocumentDownloader,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._client = springboot_client
        self._downloader = downloader
        self._settings = settings or get_settings()

    async def run(self, request: IndexJobSubmitRequest, progress: ProgressCallback) -> int:
        file_name = request.file_name or self._file_name_from_url(str(request.download_url))
        await progress(10, 0)
        downloaded = await self._downloader.download(str(request.download_url), file_name=file_name)
        await progress(25, 0)
        parsed = parse_document(downloaded.content, downloaded.file_name or file_name)
        documents = split_document(
            parsed,
            document_id=request.document_id,
            document_version=request.document_version,
            title=request.title,
            category=request.category,
            source_url="",
            options=ChunkingOptions(max_chunks=self._settings.index_max_chunks),
        )
        if not documents:
            raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可索引的分片")

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
        await self._vector_store.upsert_documents(documents)
        await progress(75, len(documents))
        await self._notify_chunks(request, documents)
        await progress(95, len(documents))
        return len(documents)

    async def delete_document(self, document_id: str) -> int:
        return await self._vector_store.delete_document(document_id)

    async def aclose(self) -> None:
        await self._downloader.aclose()
        await self._client.aclose()

    async def _notify_chunks(
        self, request: IndexJobSubmitRequest, documents: list[Document]
    ) -> None:
        batch_size = self._settings.index_callback_batch_size
        for start in range(0, len(documents), batch_size):
            items = documents[start : start + batch_size]
            batch = KnowledgeChunkBatch(
                document_id=request.document_id,
                document_version=request.document_version,
                chunks=[self._chunk_payload(document) for document in items],
            )
            await self._client.upsert_knowledge_chunks(
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
        return hashlib.sha256(
            f"{document_id}:{version}:{chunk_index}".encode()
        ).hexdigest()

    @staticmethod
    def _file_name_from_url(url: str) -> str:
        name = PurePosixPath(urlparse(url).path).name
        if not name or "." not in name:
            raise DocumentParseError("UNSUPPORTED_FILE_TYPE", "无法识别文档文件名")
        return name
