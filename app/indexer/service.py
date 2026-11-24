"""索引模块对 FastAPI 暴露的门面。"""

from __future__ import annotations

from app.indexer.jobs import IndexJobManager
from app.indexer.pipeline import IndexPipeline
from app.schemas import (
    IndexJobSubmitRequest,
    IndexJobSubmitResponse,
    IndexJobView,
)


class IndexService:
    """把“提交任务、查询任务、删除文档”收口成稳定接口。"""

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
