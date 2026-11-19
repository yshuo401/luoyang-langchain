"""Shared Pydantic models for the internal HTTP protocol."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Reject unknown fields and expose camelCase aliases to Java clients."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ApiEnvelope[T](CamelModel):
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

