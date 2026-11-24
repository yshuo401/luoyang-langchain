from __future__ import annotations

import pytest
from app.config import get_settings
from app.schemas import ChatRequest, ChatResult, IndexJobSubmitRequest, SourceItem
from pydantic import ValidationError


def test_settings_use_learning_friendly_defaults() -> None:
    settings = get_settings()
    assert settings.qwen_embedding_model == "text-embedding-v4"
    assert settings.milvus_collection.endswith("_v2")
    assert settings.mcp_12306_enabled is False


def test_chat_request_accepts_camel_case_and_rejects_unknown_fields() -> None:
    request = ChatRequest.model_validate(
        {
            "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "userId": "10001",
            "message": "龙门石窟适合玩多久？",
            "clientMessageId": "client-message-001",
            "requestId": "request-001",
            "recoveryHistory": [],
        }
    )
    assert request.thread_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "userId": "10001",
                "message": "龙门石窟",
                "clientMessageId": "client-message-001",
                "requestId": "request-001",
                "unexpected": True,
            }
        )


def test_source_snippet_is_cleaned_and_limited() -> None:
    source = SourceItem(
        sourceType="SPOT",
        title="龙门石窟",
        snippet="第一段\n\x00第二段",
        deepLink="https://evil.example",
    )
    assert source.snippet == "第一段 第二段"
    assert source.deep_link is None


def test_chat_result_serializes_camel_case() -> None:
    result = ChatResult(
        answer="资料不足",
        grounded=False,
        insufficient=True,
        insufficientReason="RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD",
        latencyMs=12,
    )
    payload = result.model_dump(by_alias=True)
    assert payload["insufficientReason"] == "RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD"
    assert payload["latencyMs"] == 12


def test_index_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        IndexJobSubmitRequest.model_validate(
            {
                "documentId": "18",
                "documentVersion": 1,
                "fileId": "88",
                "downloadUrl": "https://example.com/a.pdf",
                "docType": 1,
                "title": "龙门石窟",
                "idempotencyKey": "18:1:FULL_INDEX",
                "unexpected": True,
            }
        )
