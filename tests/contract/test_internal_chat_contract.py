from __future__ import annotations

import json

from app.api.dependencies import get_chat_service
from app.core.config import get_settings
from app.core.security import build_hmac_headers
from app.main import create_app
from app.schemas.chat import ChatResult
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


def test_chat_openapi_contract() -> None:
    schema = create_app().openapi()
    assert "/internal/v1/ai/chat" in schema["paths"]
    assert "/internal/v1/ai/checkpoints/{thread_id}/reset" in schema["paths"]
    chat_schema = schema["components"]["schemas"]["ChatRequest"]
    assert "threadId" in chat_schema["properties"]
    assert "recoveryHistory" in chat_schema["properties"]


def test_signed_chat_route_uses_request_id_as_trace_id() -> None:
    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    settings = get_settings()
    payload = {
        "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "userId": "10001",
        "message": "龙门石窟适合玩多久？",
        "clientMessageId": "client-message-001",
        "requestId": "request-chat-001",
        "recoveryHistory": [],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = build_hmac_headers(
        secret=settings.internal_shared_secret.get_secret_value(),
        method="POST",
        path="/internal/v1/ai/chat",
        request_id="request-chat-001",
        raw_body=raw,
    )
    response = TestClient(app).post("/internal/v1/ai/chat", content=raw, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["traceId"] == "request-chat-001"
    assert body["data"]["grounded"] is True
