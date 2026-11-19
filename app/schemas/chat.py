"""Internal chat request/response contracts (camelCase over the wire)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.core.constants import ULID_PATTERN
from app.schemas.common import CamelModel
from app.schemas.source import SourceItem

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class RecoveryMessage(CamelModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def strip_control_chars(cls, value: str) -> str:
        return _CONTROL_CHARS.sub("", value).strip()


class ChatRequest(CamelModel):
    thread_id: str = Field(min_length=26, max_length=26, pattern=ULID_PATTERN)
    user_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=2000)
    client_message_id: str = Field(min_length=8, max_length=64)
    request_id: str = Field(min_length=8, max_length=64)
    recovery_history: list[RecoveryMessage] = Field(default_factory=list, max_length=10)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, value: str) -> str:
        normalized = _CONTROL_CHARS.sub("", value).strip()
        if not normalized:
            raise ValueError("消息不能为空")
        return normalized

    @model_validator(mode="after")
    def limit_recovery_history(self) -> ChatRequest:
        if sum(len(item.content) for item in self.recovery_history) > 30000:
            raise ValueError("recoveryHistory 总长度超过限制")
        return self


class ChatResult(CamelModel):
    answer: str
    grounded: bool
    insufficient: bool
    insufficient_reason: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    model_name: str | None = None
    latency_ms: int = Field(ge=0)
    token_usage: dict[str, Any] | None = None


class CheckpointResetResult(CamelModel):
    thread_id: str
    reset: bool = True
