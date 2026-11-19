"""Structured JSON logging with request-scoped correlation IDs."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.constants import SERVICE_NAME

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thread_id", default=None
)


def set_log_context(*, request_id: str | None = None, thread_id: str | None = None) -> None:
    if request_id is not None:
        _request_id.set(request_id)
    if thread_id is not None:
        _thread_id.set(thread_id)


def clear_log_context() -> None:
    _request_id.set(None)
    _thread_id.set(None)


class JsonFormatter(logging.Formatter):
    """Small dependency-free JSON formatter; never logs request bodies."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "requestId": _request_id.get(),
            "threadId": _thread_id.get(),
        }
        for key in ("event", "errorCode", "latencyMs", "toolName", "resultCount", "modelName"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exceptionType"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Configure one stdout handler without duplicating handlers on reload."""

    normalized = level.upper()
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, normalized, logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logging.getLogger(SERVICE_NAME).setLevel(getattr(logging, normalized, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

