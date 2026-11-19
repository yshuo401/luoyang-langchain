from __future__ import annotations

import pytest
from app.schemas.chat import ChatRequest, ChatResult
from app.schemas.index import IndexJobSubmitRequest
from app.schemas.source import SourceItem
from pydantic import ValidationError


def test_chat_request_uses_camel_case_aliases() -> None:
    request = ChatRequest.model_validate(
        {
            "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "userId": "10001",
            "message": "龙门石窟适合玩多久？",
            "clientMessageId": "client-message-001",
            "requestId": "request-001",
            "recoveryHistory": [{"role": "user", "content": "上一轮"}],
        }
    )
    dumped = request.model_dump(by_alias=True)
    assert dumped["threadId"] == request.thread_id
    assert dumped["recoveryHistory"][0]["role"] == "user"


def test_invalid_ulid_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAU",
                "userId": "10001",
                "message": "问题",
                "clientMessageId": "client-message-001",
                "requestId": "request-001",
            }
        )


def test_source_sanitizes_script_and_invalid_deep_link() -> None:
    source = SourceItem(
        sourceType="KNOWLEDGE",
        sourceId="7",
        title="龙门石窟",
        snippet="  第一段\x00 第二段  ",
        deepLink="javascript:alert(1)",
        rankNo=1,
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

