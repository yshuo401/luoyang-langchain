"""FastAPI 入口、路由和统一异常处理。

这个文件只负责“HTTP 适配”：

- 读取请求；
- 校验内部 token；
- 调用 ChatService 或 IndexService；
- 把异常变成统一 JSON。

业务逻辑不写在这里，保持接口层足够薄。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.chat import ChatService
from app.config import (
    INTERNAL_API_PREFIX,
    REQUEST_ID_HEADER,
    SERVICE_NAME,
    SERVICE_VERSION,
    ULID_PATTERN,
    configure_logging,
    get_logger,
    get_settings,
    verify_internal_token,
)
from app.errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    ResourceNotFoundError,
    ServiceError,
)
from app.indexer import IndexService
from app.rag import check_milvus
from app.runtime import ApplicationRuntime, build_runtime
from app.schemas import (
    ApiEnvelope,
    ChatRequest,
    ChatResult,
    CheckpointResetResult,
    DeleteDocumentResult,
    IndexJobSubmitRequest,
    IndexJobSubmitResponse,
    IndexJobView,
    LiveResponse,
    ReadyResponse,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.runtime = None
    app.state.index_service = None
    app.state.chat_service = None
    runtime: ApplicationRuntime | None = None
    try:
        runtime = await build_runtime(settings)
        app.state.runtime = runtime
        app.state.index_service = runtime.index_service
        app.state.chat_service = runtime.chat_service
        logger.info("service_started")
    except Exception:
        logger.exception("runtime_initialization_failed")

    try:
        yield
    finally:
        runtime = getattr(app.state, "runtime", None)
        if runtime is not None:
            await runtime.close()
        logger.info("service_stopped")


def _request_id(request: Request) -> str:
    return request.headers.get(REQUEST_ID_HEADER, "-")


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise DependencyUnavailableError("智能问答服务尚未就绪")
    return service


def get_index_service(request: Request) -> IndexService:
    service = getattr(request.app.state, "index_service", None)
    if service is None:
        raise DependencyUnavailableError("索引服务尚未就绪")
    return service


def create_app() -> FastAPI:
    app = FastAPI(
        title="洛阳景点智能推荐 AI 服务",
        description="仅供 Spring Boot 内网调用的 RAG / Agent 能力层。",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    # 即使 TestClient 没有执行 lifespan，健康检查和异常处理也能安全读取配置。
    app.state.settings = get_settings()
    app.state.runtime = None
    app.state.index_service = None
    app.state.chat_service = None

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        payload = ApiEnvelope.error(
            code=exc.code,
            message=exc.message,
            request_id=_request_id(request),
            data={"retryable": exc.retryable},
        )
        return JSONResponse(status_code=exc.http_status, content=payload.model_dump(by_alias=True))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        payload = ApiEnvelope.error(
            code=40000,
            message="请求参数不合法",
            request_id=_request_id(request),
            data={"errors": jsonable_encoder(exc.errors())},
        )
        return JSONResponse(status_code=400, content=payload.model_dump(by_alias=True))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")
        payload = ApiEnvelope.error(
            code=50000,
            message="AI 服务内部错误",
            request_id=_request_id(request),
            data={"retryable": False},
        )
        return JSONResponse(status_code=500, content=payload.model_dump(by_alias=True))

    # 所有内部业务接口都要求固定 token。健康检查故意不加 token，方便本地
    # 直接打开浏览器确认服务是否启动。
    internal = APIRouter(
        prefix=INTERNAL_API_PREFIX,
        tags=["internal"],
        dependencies=[Depends(verify_internal_token)],
    )
    health = APIRouter(tags=["health"])

    @internal.post("/chat", response_model=ApiEnvelope[ChatResult], summary="执行 RAG/Agent 问答")
    async def chat(
        payload: ChatRequest,
        request: Request,
        service: Annotated[ChatService, Depends(get_chat_service)],
    ) -> ApiEnvelope[ChatResult]:
        result = await service.chat(payload, _request_id(request))
        return ApiEnvelope.ok(result, request_id=payload.request_id)

    @internal.post(
        "/checkpoints/{thread_id}/reset",
        response_model=ApiEnvelope[CheckpointResetResult],
        summary="删除指定会话的 LangGraph checkpoint",
    )
    async def reset_checkpoint(
        thread_id: str,
        request: Request,
        service: Annotated[ChatService, Depends(get_chat_service)],
    ) -> ApiEnvelope[CheckpointResetResult]:
        if not re.fullmatch(ULID_PATTERN, thread_id):
            raise InvalidRequestError("threadId 必须是合法 ULID")
        await service.reset_checkpoint(thread_id)
        return ApiEnvelope.ok(
            CheckpointResetResult(thread_id=thread_id, reset=True),
            request_id=_request_id(request),
        )

    @internal.post(
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

    @internal.get(
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

    @internal.post(
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
        return ApiEnvelope.ok(
            DeleteDocumentResult(
                document_id=document_id,
                deleted=True,
                deleted_vector_count=count,
            ),
            request_id=_request_id(request),
        )

    @health.get("/health/live", response_model=LiveResponse, summary="进程存活检查")
    async def live() -> LiveResponse:
        return LiveResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)

    @health.get("/health/ready", response_model=ReadyResponse, summary="依赖就绪检查")
    async def ready(request: Request) -> ReadyResponse | JSONResponse:
        settings = request.app.state.settings
        checks: dict[str, bool] = {"milvus": False, "agent": False}
        try:
            checks["milvus"] = await asyncio.wait_for(
                asyncio.to_thread(check_milvus, settings),
                timeout=5,
            )
        except Exception:
            checks["milvus"] = False

        runtime = getattr(request.app.state, "runtime", None)
        checks["agent"] = bool(runtime is not None and runtime.agent is not None)
        all_ready = all(checks.values())
        payload = ReadyResponse(status="ready" if all_ready else "not_ready", checks=checks)
        if not all_ready:
            return JSONResponse(status_code=503, content=payload.model_dump())
        return payload

    app.include_router(internal)
    app.include_router(health)
    return app


app = create_app()
