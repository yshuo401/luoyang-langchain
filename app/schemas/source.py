"""Structured source references returned to the mini-program."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from app.schemas.common import CamelModel

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SourceItem(CamelModel):
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
