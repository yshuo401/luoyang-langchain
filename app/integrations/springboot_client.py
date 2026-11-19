"""HMAC-signed async client for Spring Boot internal read-only APIs."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from app.core.config import get_settings
from app.core.errors import SpringBootError
from app.core.ids import new_ulid
from app.core.security import build_hmac_headers
from app.schemas.index import KnowledgeChunkBatch

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SpringBootClient:
    """Reuse one connection pool and one canonical signing implementation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        secret: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        config = get_settings()
        self._base_url = (base_url or config.springboot_internal_base_url).rstrip("/")
        self._secret = secret or config.internal_shared_secret.get_secret_value()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=2.0, read=timeout_seconds, write=timeout_seconds, pool=2.0
            ),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            follow_redirects=False,
        )

    async def __aenter__(self) -> SpringBootClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _safe_id(value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise SpringBootError("资源标识不合法", error_code="INVALID_RESOURCE_ID")
        return value

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        retry_get: bool = False,
    ) -> Any:
        query = ""
        if params:
            cleaned = [(str(key), value) for key, value in params.items() if value is not None]
            query = urlencode(sorted(cleaned), doseq=True)
        target = f"{path}?{query}" if query else path
        raw_body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else b""
        )
        attempts = 2 if retry_get and method.upper() == "GET" else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            request_id = new_ulid()
            headers = build_hmac_headers(
                secret=self._secret,
                method=method,
                path=path,
                request_id=request_id,
                raw_body=raw_body,
                params=[
                    (str(key), value)
                    for key, value in (params or {}).items()
                    if value is not None
                ],
            )
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key
            try:
                response = await self._client.request(
                    method.upper(),
                    f"{self._base_url}{target}",
                    content=raw_body,
                    headers=headers,
                )
                if response.status_code >= 500 and retry_get and attempt + 1 < attempts:
                    await asyncio.sleep(0.1)
                    continue
                return self._decode(response)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.1)
                    continue
                break
        raise SpringBootError(
            "业务数据服务暂时不可用",
            error_code="SPRINGBOOT_TIMEOUT",
            retryable=True,
        ) from last_error

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        if response.status_code < 200 or response.status_code >= 300:
            retryable = response.status_code in {502, 503, 504}
            raise SpringBootError(
                "业务数据服务暂时不可用",
                error_code=f"SPRINGBOOT_{response.status_code}",
                retryable=retryable,
                http_status=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SpringBootError(
                "业务数据服务返回格式异常",
                error_code="SPRINGBOOT_INVALID_RESPONSE",
            ) from exc
        if isinstance(payload, dict) and "code" in payload:
            if payload.get("code") not in {0, "0", None}:
                raise SpringBootError(
                    "业务数据服务返回业务错误",
                    error_code=str(payload.get("code")),
                )
            return payload.get("data")
        return payload

    async def search_spots(
        self, *, keyword: str, category_id: str | None = None, limit: int = 5
    ) -> Any:
        return await self._request(
            "GET",
            "/internal/v1/ai/spots/search",
            params={"keyword": keyword, "categoryId": category_id, "limit": min(limit, 10)},
            retry_get=True,
        )

    async def get_spot(self, spot_id: str) -> Any:
        return await self._request(
            "GET",
            f"/internal/v1/ai/spots/{quote(self._safe_id(spot_id), safe='')}",
            retry_get=True,
        )

    async def search_guides(self, *, keyword: str, limit: int = 5) -> Any:
        return await self._request(
            "GET",
            "/internal/v1/ai/guides/search",
            params={"keyword": keyword, "limit": min(limit, 10)},
            retry_get=True,
        )

    async def get_guide(self, guide_id: str) -> Any:
        return await self._request(
            "GET",
            f"/internal/v1/ai/guides/{quote(self._safe_id(guide_id), safe='')}",
            retry_get=True,
        )

    async def search_foods(
        self, *, keyword: str, category_id: str | None = None, limit: int = 5
    ) -> Any:
        return await self._request(
            "GET",
            "/internal/v1/ai/foods/search",
            params={"keyword": keyword, "categoryId": category_id, "limit": min(limit, 10)},
            retry_get=True,
        )

    async def get_food(self, food_id: str) -> Any:
        return await self._request(
            "GET",
            f"/internal/v1/ai/foods/{quote(self._safe_id(food_id), safe='')}",
            retry_get=True,
        )

    async def recommend(
        self, *, user_id: str, scene: str, limit: int, context: str | None
    ) -> Any:
        return await self._request(
            "POST",
            "/internal/v1/ai/recommendations",
            payload={
                "userId": user_id,
                "scene": scene,
                "limit": min(limit, 10),
                "context": context,
            },
        )

    async def upsert_knowledge_chunks(
        self, batch: KnowledgeChunkBatch, *, idempotency_key: str
    ) -> Any:
        return await self._request(
            "POST",
            "/internal/v1/ai/knowledge/chunks/batch",
            payload=batch.model_dump(by_alias=True, mode="json"),
            idempotency_key=idempotency_key,
        )


