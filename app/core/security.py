"""HMAC-SHA256 authentication for the Spring Boot -> FastAPI boundary."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping, Sequence
from urllib.parse import parse_qsl, urlencode

from fastapi import Header, Request

from app.core.config import Settings, get_settings
from app.core.constants import HMAC_HEADER, REQUEST_ID_HEADER, TIMESTAMP_HEADER
from app.core.errors import InternalAuthError


def build_path_with_sorted_query(path: str, query: str) -> str:
    """Return path plus query parameters sorted by key and value."""

    items = sorted(parse_qsl(query, keep_blank_values=True))
    if not items:
        return path
    return f"{path}?{urlencode(items, doseq=True)}"


def build_canonical_request(
    *,
    timestamp: str,
    method: str,
    path: str,
    query: str,
    request_id: str,
    raw_body: bytes,
) -> bytes:
    """Build the canonical byte string shared by Java and Python."""

    target = build_path_with_sorted_query(path, query)
    return b"\n".join(
        [
            timestamp.encode("utf-8"),
            method.upper().encode("utf-8"),
            target.encode("utf-8"),
            request_id.encode("utf-8"),
            raw_body,
        ]
    )


def calculate_signature(secret: str, canonical: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def build_hmac_headers(
    *,
    secret: str,
    method: str,
    path: str,
    request_id: str,
    raw_body: bytes = b"",
    timestamp: str | None = None,
    params: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
) -> dict[str, str]:
    """Create headers for tests and for FastAPI -> Spring Boot calls."""

    query = ""
    if params:
        pairs = params.items() if isinstance(params, Mapping) else params
        query = urlencode(sorted((str(k), str(v)) for k, v in pairs if v is not None), doseq=True)
    ts = timestamp or str(int(time.time()))
    canonical = build_canonical_request(
        timestamp=ts,
        method=method,
        path=path,
        query=query,
        request_id=request_id,
        raw_body=raw_body,
    )
    return {
        TIMESTAMP_HEADER: ts,
        REQUEST_ID_HEADER: request_id,
        HMAC_HEADER: calculate_signature(secret, canonical),
        "Content-Type": "application/json; charset=UTF-8",
    }


async def verify_internal_hmac(
    request: Request,
    x_internal_timestamp: str = Header(alias=TIMESTAMP_HEADER),
    x_internal_request_id: str = Header(alias=REQUEST_ID_HEADER),
    x_internal_signature: str = Header(alias=HMAC_HEADER),
) -> None:
    """Validate timestamp, request ID, canonical path and raw body signature."""

    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    if not 8 <= len(x_internal_request_id) <= 64:
        raise InternalAuthError("内部请求标识不合法")
    try:
        timestamp_value = int(x_internal_timestamp)
    except ValueError as exc:
        raise InternalAuthError("内部鉴权时间戳不合法") from exc

    if abs(int(time.time()) - timestamp_value) > settings.internal_timestamp_tolerance_seconds:
        raise InternalAuthError("内部鉴权已过期")

    signature = x_internal_signature.strip().lower()
    if len(signature) != 64 or any(char not in "0123456789abcdef" for char in signature):
        raise InternalAuthError("内部签名格式不合法")

    raw_body = await request.body()
    canonical = build_canonical_request(
        timestamp=x_internal_timestamp,
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        request_id=x_internal_request_id,
        raw_body=raw_body,
    )
    expected = calculate_signature(
        settings.internal_shared_secret.get_secret_value(), canonical
    )
    if not hmac.compare_digest(expected, signature):
        raise InternalAuthError("内部签名校验失败")
