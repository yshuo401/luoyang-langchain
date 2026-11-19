"""Use-case orchestration for knowledge indexing and vector deletion."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.ingestion.jobs import IndexJobManager, RedisIndexJobStore
from app.ingestion.pipeline import IndexPipeline
from app.integrations.file_downloader import DocumentDownloader
from app.integrations.milvus import create_vector_store
from app.integrations.qwen import create_embedding_model
from app.integrations.springboot_client import SpringBootClient
from app.rag.vector_store import KnowledgeVectorStore
from app.schemas.index import IndexJobSubmitRequest, IndexJobSubmitResponse, IndexJobView


class IndexService:
    def __init__(self, manager: IndexJobManager, pipeline: IndexPipeline) -> None:
        self._manager = manager
        self._pipeline = pipeline

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> IndexService:
        config = settings or get_settings()
        embedding = create_embedding_model(config)
        vector_store = KnowledgeVectorStore(
            create_vector_store(embedding, settings=config), config
        )
        client = SpringBootClient(timeout_seconds=config.tool_timeout_seconds)
        pipeline = IndexPipeline(
            vector_store=vector_store,
            springboot_client=client,
            downloader=DocumentDownloader(config),
            settings=config,
        )
        manager = IndexJobManager(RedisIndexJobStore(config), pipeline, config)
        return cls(manager, pipeline)

    async def submit(self, request: IndexJobSubmitRequest) -> IndexJobSubmitResponse:
        job = await self._manager.submit(request)
        return IndexJobSubmitResponse(
            job_id=job.job_id,
            external_job_id=job.external_job_id,
            status=job.status,
        )

    async def get_job(self, job_id: str) -> IndexJobView | None:
        return await self._manager.get(job_id)

    async def delete_document(self, document_id: str) -> int:
        return await self._pipeline.delete_document(document_id)

    async def close(self) -> None:
        await self._manager.close()
        await self._pipeline.aclose()
