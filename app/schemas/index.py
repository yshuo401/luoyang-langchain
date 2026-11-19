"""Internal knowledge-indexing contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, HttpUrl, field_validator

from app.schemas.common import CamelModel

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
