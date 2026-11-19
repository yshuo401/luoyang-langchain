"""健康检查路由。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pymilvus import MilvusClient
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.constants import SERVICE_NAME, SERVICE_VERSION
from app.integrations.milvus import ensure_no_proxy_for_url
from app.schemas.health import LiveResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LiveResponse, summary="进程存活检查")
async def live() -> LiveResponse:
    """仅证明 FastAPI 事件循环可响应，不访问 Redis、Milvus 或 Qwen。"""

    return LiveResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


async def _check_redis(settings: Settings) -> bool:
    client = Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    try:
        return bool(await client.ping())
    finally:
        await client.aclose()


def _check_milvus(settings: Settings) -> bool:
    ensure_no_proxy_for_url(settings.milvus_uri)
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token.get_secret_value())
    try:
        client.list_collections()
        return True
    finally:
        client.close()


@router.get("/health/ready", response_model=ReadyResponse, summary="依赖就绪检查")
async def ready() -> ReadyResponse | JSONResponse:
    """检查 Redis 与 Milvus；任一不可用时返回 503，不检查模型外网。"""

    settings = get_settings()
    checks: dict[str, bool] = {"redis": False, "milvus": False}
    try:
        checks["redis"] = await _check_redis(settings)
    except Exception:
        checks["redis"] = False
    try:
        checks["milvus"] = await asyncio.wait_for(
            asyncio.to_thread(_check_milvus, settings),
            timeout=5,
        )
    except Exception:
        checks["milvus"] = False

    all_ready = all(checks.values())
    payload = ReadyResponse(status="ready" if all_ready else "not_ready", checks=checks)
    if not all_ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload