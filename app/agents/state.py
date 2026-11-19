"""Request-scoped context for Agent tools and evidence collection."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.chains.grounding import EvidenceRegistry


@dataclass(slots=True)
class RequestContext:
    user_id: str
    thread_id: str
    springboot_client: Any
    rag_service: Any
    max_tool_calls: int
    evidence: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)

    def reserve_tool_call(self, name: str) -> bool:
        if self.tool_call_count >= self.max_tool_calls:
            self.tool_errors.append("TOOL_BUDGET_EXCEEDED")
            return False
        self.tool_call_count += 1
        self.tool_names.append(name)
        return True

    def record_error(self, code: str) -> None:
        self.tool_errors.append(code)


_current_context: ContextVar[RequestContext | None] = ContextVar(
    "tourism_request_context", default=None
)


def current_request_context() -> RequestContext:
    context = _current_context.get()
    if context is None:
        raise RuntimeError("Agent tool called outside a request context")
    return context


def set_request_context(context: RequestContext) -> Token[RequestContext | None]:
    return _current_context.set(context)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _current_context.reset(token)
