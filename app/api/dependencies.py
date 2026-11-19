"""Dependency accessors shared by internal routes."""

from __future__ import annotations

from fastapi import Request

from app.core.errors import DependencyUnavailableError
from app.services.chat_service import ChatService
from app.services.index_service import IndexService


def get_index_service(request: Request) -> IndexService:
    service = getattr(request.app.state, "index_service", None)
    if service is None:
        raise DependencyUnavailableError("索引服务尚未就绪")
    return service


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise DependencyUnavailableError("智能问答服务尚未就绪")
    return service
