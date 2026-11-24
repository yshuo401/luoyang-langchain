from __future__ import annotations

import json

from app.config import get_settings
from app.main import create_app, get_chat_service, get_index_service
from app.schemas import ChatResult, IndexJobSubmitResponse
from fastapi.testclient import TestClient


class FakeChatService:
    async def chat(self, request: object, request_id: str) -> ChatResult:
        del request, request_id
        return ChatResult(
            answer="龙门石窟适合游玩半天。[S1]",
            grounded=True,
            insufficient=False,
            sources=[],
            latencyMs=12,
        )


class FakeIndexService:
    async def submit(self, request: object) -> IndexJobSubmitResponse:
        del request
        return IndexJobSubmitResponse(
            job_id="job-12345678",
            external_job_id="job-12345678",
            status="QUEUED",
        )


def _headers() -> dict[str, str]:
    return {
        "X-Internal-Token": get_settings().internal_token.get_secret_value(),
        "X-Internal-Request-Id": "request-contract-001",
        "Content-Type": "application/json",
    }


def test_openapi_contains_all_internal_routes() -> None:
    schema = create_app().openapi()
    assert "/internal/v1/ai/chat" in schema["paths"]
    assert "/internal/v1/ai/checkpoints/{thread_id}/reset" in schema["paths"]
    assert "/internal/v1/ai/index/jobs" in schema["paths"]
    assert "/internal/v1/ai/index/jobs/{job_id}" in schema["paths"]
    assert "/internal/v1/ai/index/documents/{document_id}/delete" in schema["paths"]
    chat_schema = schema["components"]["schemas"]["ChatRequest"]
    assert "threadId" in chat_schema["properties"]
    assert "recoveryHistory" in chat_schema["properties"]


def test_chat_route_uses_request_id_as_trace_id() -> None:
    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    payload = {
        "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "userId": "10001",
        "message": "龙门石窟适合玩多久？",
        "clientMessageId": "client-message-001",
        "requestId": "request-chat-001",
        "recoveryHistory": [],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    response = TestClient(app).post("/internal/v1/ai/chat", content=raw, headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["traceId"] == "request-chat-001"
    assert body["data"]["grounded"] is True


def test_index_route_returns_camel_case_envelope() -> None:
    app = create_app()
    app.dependency_overrides[get_index_service] = lambda: FakeIndexService()
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
    response = TestClient(app).post("/internal/v1/ai/index/jobs", content=raw, headers=_headers())
    assert response.status_code == 202
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["externalJobId"] == "job-12345678"
    assert body["traceId"] == "request-contract-001"
