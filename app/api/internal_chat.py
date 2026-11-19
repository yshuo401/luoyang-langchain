"""Java -> FastAPI chat and checkpoint-reset routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_chat_service
from app.core.constants import INTERNAL_API_PREFIX, REQUEST_ID_HEADER, ULID_PATTERN
from app.core.errors import InvalidRequestError
from app.core.security import verify_internal_hmac
from app.schemas.chat import ChatRequest, ChatResult, CheckpointResetResult
from app.schemas.common import ApiEnvelope
from app.services.chat_service import ChatService

router = APIRouter(
    prefix=INTERNAL_API_PREFIX,
    tags=["internal-chat"],
    dependencies=[Depends(verify_internal_hmac)],
)


def _request_id(request: Request) -> str:
    return request.headers[REQUEST_ID_HEADER]


@router.post("/chat", response_model=ApiEnvelope[ChatResult], summary="执行 RAG/Agent 问答")
async def chat(
    payload: ChatRequest,
    request: Request,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ApiEnvelope[ChatResult]:
    result = await service.chat(payload, _request_id(request))
    return ApiEnvelope.ok(result, request_id=payload.request_id)


@router.post(
    "/checkpoints/{thread_id}/reset",
    response_model=ApiEnvelope[CheckpointResetResult],
    summary="删除指定会话的 LangGraph checkpoint",
)
async def reset_checkpoint(
    thread_id: str,
    request: Request,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ApiEnvelope[CheckpointResetResult]:
    import re

    if not re.fullmatch(ULID_PATTERN, thread_id):
        raise InvalidRequestError("threadId 必须是合法 ULID")
    await service.reset_checkpoint(thread_id)
    result = CheckpointResetResult(thread_id=thread_id, reset=True)
    return ApiEnvelope.ok(result, request_id=_request_id(request))
