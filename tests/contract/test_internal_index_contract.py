from __future__ import annotations

import json

from app.api.dependencies import get_index_service
from app.core.config import get_settings
from app.core.security import build_hmac_headers
from app.main import create_app
from app.schemas.index import IndexJobSubmitResponse
from fastapi.testclient import TestClient


class FakeIndexService:
    async def submit(self, request: object) -> IndexJobSubmitResponse:
        del request
        return IndexJobSubmitResponse(
            job_id="job-12345678",
            external_job_id="job-12345678",
            status="QUEUED",
        )


def test_internal_index_openapi_contract() -> None:
    schema = create_app().openapi()
    assert "/internal/v1/ai/index/jobs" in schema["paths"]
    assert "/internal/v1/ai/index/jobs/{job_id}" in schema["paths"]
    assert "/internal/v1/ai/index/documents/{document_id}/delete" in schema["paths"]
    submit_schema = schema["components"]["schemas"]["IndexJobSubmitRequest"]
    assert "documentId" in submit_schema["properties"]
    assert "downloadUrl" in submit_schema["properties"]


def test_signed_index_submit_returns_camel_case_envelope() -> None:
    app = create_app()
    app.dependency_overrides[get_index_service] = lambda: FakeIndexService()
    settings = get_settings()
    payload = {
        "documentId": "18",
        "documentVersion": 1,
        "fileId": "88",
        "downloadUrl": "https://example.com/guide.txt",
        "docType": 1,
        "title": "龙门石窟讲解",
        "category": "SCENIC_INTRO",
        "fileName": "guide.txt",
        "idempotencyKey": "18:1:FULL_INDEX",
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = build_hmac_headers(
        secret=settings.internal_shared_secret.get_secret_value(),
        method="POST",
        path="/internal/v1/ai/index/jobs",
        request_id="request-index-001",
        raw_body=raw,
    )
    response = TestClient(app).post(
        "/internal/v1/ai/index/jobs", content=raw, headers=headers
    )
    assert response.status_code == 202
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["externalJobId"] == "job-12345678"
    assert body["traceId"] == "request-index-001"
