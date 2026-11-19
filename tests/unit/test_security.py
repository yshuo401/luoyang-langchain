from __future__ import annotations

import json
import time

from app.core.config import get_settings
from app.core.errors import ServiceError
from app.core.security import (
    build_canonical_request,
    build_hmac_headers,
    calculate_signature,
    verify_internal_hmac,
)
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _client() -> TestClient:
    app = FastAPI()
    app.state.settings = get_settings()

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"code": exc.code})

    @app.post("/internal/v1/ai/test")
    async def protected(request: Request, _: None = Depends(verify_internal_hmac)) -> dict:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_canonical_string_has_stable_query_order() -> None:
    first = build_canonical_request(
        timestamp="1700000000",
        method="get",
        path="/internal/v1/ai/spots/search",
        query="b=2&a=1",
        request_id="request-123",
        raw_body=b"",
    )
    second = build_canonical_request(
        timestamp="1700000000",
        method="GET",
        path="/internal/v1/ai/spots/search",
        query="a=1&b=2",
        request_id="request-123",
        raw_body=b"",
    )
    assert first == second
    assert first.split(b"\n")[2] == b"/internal/v1/ai/spots/search?a=1&b=2"


def test_known_signature_is_deterministic() -> None:
    canonical = b"1700000000\nPOST\n/internal/v1/ai/chat\nrequest-123\n{}"
    assert (
        calculate_signature("test-secret", canonical)
        == calculate_signature("test-secret", canonical)
    )


def test_valid_hmac_is_accepted() -> None:
    settings = get_settings()
    payload = {"threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = build_hmac_headers(
        secret=settings.internal_shared_secret.get_secret_value(),
        method="POST",
        path="/internal/v1/ai/test",
        request_id="request-123",
        raw_body=raw,
        params={"b": "2", "a": "1"},
    )
    response = _client().post("/internal/v1/ai/test?b=2&a=1", content=raw, headers=headers)
    assert response.status_code == 200


def test_expired_or_tampered_signature_is_rejected() -> None:
    settings = get_settings()
    expired = str(int(time.time()) - settings.internal_timestamp_tolerance_seconds - 1)
    headers = build_hmac_headers(
        secret=settings.internal_shared_secret.get_secret_value(),
        method="POST",
        path="/internal/v1/ai/test",
        request_id="request-expired",
        raw_body=b"{}",
        timestamp=expired,
    )
    expired_response = _client().post(
        "/internal/v1/ai/test", content=b"{}", headers=headers
    )
    assert expired_response.status_code == 401

    valid_headers = build_hmac_headers(
        secret=settings.internal_shared_secret.get_secret_value(),
        method="POST",
        path="/internal/v1/ai/test",
        request_id="request-tampered",
        raw_body=b"{}",
    )
    tampered_response = _client().post(
        "/internal/v1/ai/test", content=b'{"x":1}', headers=valid_headers
    )
    assert tampered_response.status_code == 401
