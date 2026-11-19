"""FastAPI entry point and protocol-level exception mapping."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.internal_chat import router as internal_chat_router
from app.api.internal_index import router as internal_index_router
from app.core.config import get_settings
from app.core.constants import REQUEST_ID_HEADER, SERVICE_VERSION
from app.core.errors import ServiceError
from app.core.logging import configure_logging, get_logger
from app.runtime import ApplicationRuntime, build_runtime
from app.schemas.common import ApiEnvelope

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    configure_logging(settings.log_level)
    runtime: ApplicationRuntime | None = None
    app.state.index_service = None
    app.state.chat_service = None
    try:
        runtime = await build_runtime(settings)
        app.state.runtime = runtime
        app.state.index_service = runtime.index_service
        app.state.chat_service = runtime.chat_service
    except Exception:
        logger.exception("runtime_initialization_failed")
    logger.info("service_started", extra={"event": "service_started"})
    try:
        yield
    finally:
        if runtime is not None:
            await runtime.close()
        logger.info("service_stopped", extra={"event": "service_stopped"})


def _request_id(request: Request) -> str:
    return request.headers.get(REQUEST_ID_HEADER) or "-"


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
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        payload = ApiEnvelope.error(
            code=40000,
            message="请求参数不合法",
            request_id=_request_id(request),
            data={"errors": exc.errors()},
        )
        return JSONResponse(status_code=400, content=payload.model_dump(by_alias=True))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_error",
            extra={"event": "unhandled_error", "errorCode": 50000},
        )
        payload = ApiEnvelope.error(
            code=50000,
            message="AI 服务内部错误",
            request_id=_request_id(request),
            data={"retryable": False},
        )
        return JSONResponse(status_code=500, content=payload.model_dump(by_alias=True))

    app.include_router(health_router)
    app.include_router(internal_chat_router)
    app.include_router(internal_index_router)
    return app


app = create_app()
