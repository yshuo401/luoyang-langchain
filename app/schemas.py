"""FastAPI 对 Java 暴露的全部数据结构。

为什么集中在一个文件：这些模型共同描述同一条内部 HTTP 协议。把它们放在
一起，读接口时可以直接看到请求、响应、来源和索引任务长什么样，不必在
`schemas/` 目录里寻找 5 个文件。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from pydantic.alias_generators import to_camel

from app.config import ULID_PATTERN

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CamelModel(BaseModel):
    """Java 侧使用 camelCase，Python 代码内部使用 snake_case。

    `alias_generator=to_camel` 负责在序列化时把 `thread_id` 变成 `threadId`。
    `extra="forbid"` 让拼错的字段直接报 422，而不是悄悄忽略。
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ApiEnvelope[T](CamelModel):
    """Spring Boot 调用的统一响应外壳。"""

    code: int = 0
    message: str = "success"
    data: T | None = None
    trace_id: str
    timestamp: str

    @classmethod
    def ok(cls, data: T, *, request_id: str, message: str = "success") -> ApiEnvelope[T]:
        return cls(
            code=0,
            message=message,
            data=data,
            trace_id=request_id,
            timestamp=datetime.now(UTC).isoformat(),
        )

    @classmethod
    def error(
        cls,
        *,
        code: int,
        message: str,
        request_id: str,
        data: T | None = None,
    ) -> ApiEnvelope[T]:
        return cls(
            code=code,
            message=message,
            data=data,
            trace_id=request_id,
            timestamp=datetime.now(UTC).isoformat(),
        )


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


class SourceItem(CamelModel):
    """返回给小程序展示的来源卡片。

    现有 Java/小程序只支持五种 sourceType，因此 MCP 结果也暂时使用
    KNOWLEDGE，并在标题和摘要里标明“12306 实时查询”。
    """

    source_type: Literal["KNOWLEDGE", "SPOT", "GUIDE", "FOOD", "RECOMMENDATION"]
    source_id: str | None = Field(default=None, max_length=64)
    chunk_id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    snippet: str | None = Field(default=None, max_length=500)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rank_no: int = Field(default=1, ge=1)
    deep_link: str | None = Field(default=None, max_length=300)

    @field_validator("snippet")
    @classmethod
    def sanitize_snippet(cls, value: str | None) -> str | None:
        if value is None:
            return None
        compact = " ".join(_CONTROL_CHARS.sub(" ", value).split())
        return compact[:500] or None

    @field_validator("deep_link")
    @classmethod
    def validate_deep_link(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith("/pages/") or "://" in value or ".." in value:
            return None
        return value


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


IndexJobStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]


class IndexJobSubmitRequest(CamelModel):
    document_id: str = Field(min_length=1, max_length=64)
    document_version: int = Field(ge=1)
    file_id: str = Field(min_length=1, max_length=64)
    download_url: HttpUrl
    doc_type: int = Field(ge=1, le=99)
    title: str = Field(min_length=1, max_length=200)
    category: str | None = Field(default=None, max_length=64)
    file_name: str | None = Field(default=None, max_length=255)
    idempotency_key: str = Field(min_length=3, max_length=128)

    @field_validator("download_url")
    @classmethod
    def only_http(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme not in {"http", "https"}:
            raise ValueError("downloadUrl 必须是 http(s)")
        return value


class IndexJobSubmitResponse(CamelModel):
    job_id: str
    external_job_id: str
    status: IndexJobStatus


class IndexJobView(CamelModel):
    job_id: str
    external_job_id: str
    document_id: str
    document_version: int
    status: IndexJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    chunk_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class KnowledgeChunkPayload(CamelModel):
    chunk_index: int = Field(ge=0)
    vector_id: str = Field(min_length=16, max_length=128)
    content: str = Field(min_length=1)
    content_hash: str = Field(min_length=16, max_length=128)
    char_count: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunkBatch(CamelModel):
    document_id: str
    document_version: int = Field(ge=1)
    chunks: list[KnowledgeChunkPayload]


class DeleteDocumentResult(CamelModel):
    document_id: str
    deleted: bool
    deleted_vector_count: int = Field(default=0, ge=0)


class LiveResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
