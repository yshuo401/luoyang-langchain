"""Java -> FastAPI endpoints for knowledge indexing."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import get_index_service
from app.core.constants import INTERNAL_API_PREFIX, REQUEST_ID_HEADER
from app.core.errors import InvalidRequestError, ResourceNotFoundError
from app.core.security import verify_internal_hmac
from app.schemas.common import ApiEnvelope
from app.schemas.index import (
    DeleteDocumentResult,
    IndexJobSubmitRequest,
    IndexJobSubmitResponse,
    IndexJobView,
)
from app.services.index_service import IndexService

router = APIRouter(
    prefix=INTERNAL_API_PREFIX,
    tags=["internal-index"],
    dependencies=[Depends(verify_internal_hmac)],
)


def _request_id(request: Request) -> str:
    return request.headers[REQUEST_ID_HEADER]


@router.post(
    "/index/jobs",
    response_model=ApiEnvelope[IndexJobSubmitResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交知识库索引任务",
)
async def submit_index_job(
    payload: IndexJobSubmitRequest,
    request: Request,
    service: Annotated[IndexService, Depends(get_index_service)],
) -> ApiEnvelope[IndexJobSubmitResponse]:
    result = await service.submit(payload)
    return ApiEnvelope.ok(result, request_id=_request_id(request))


@router.get(
    "/index/jobs/{job_id}",
    response_model=ApiEnvelope[IndexJobView],
    summary="查询索引任务状态",
)
async def get_index_job(
    job_id: str,
    request: Request,
    service: Annotated[IndexService, Depends(get_index_service)],
) -> ApiEnvelope[IndexJobView]:
    if len(job_id) < 8 or len(job_id) > 64:
        raise InvalidRequestError("jobId 不合法")
    job = await service.get_job(job_id)
    if job is None:
        raise ResourceNotFoundError("索引任务不存在")
    return ApiEnvelope.ok(job, request_id=_request_id(request))


@router.post(
    "/index/documents/{document_id}/delete",
    response_model=ApiEnvelope[DeleteDocumentResult],
    summary="删除文档的全部知识向量",
)
async def delete_document_vectors(
    document_id: str,
    request: Request,
    service: Annotated[IndexService, Depends(get_index_service)],
) -> ApiEnvelope[DeleteDocumentResult]:
    if len(document_id) < 1 or len(document_id) > 64:
        raise InvalidRequestError("documentId 不合法")
    count = await service.delete_document(document_id)
    result = DeleteDocumentResult(
        document_id=document_id,
        deleted=True,
        deleted_vector_count=count,
    )
    return ApiEnvelope.ok(result, request_id=_request_id(request))


