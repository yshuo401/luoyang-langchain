"""Application runtime composition; dependencies are created once in lifespan."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agents.tourism_agent import create_tourism_agent
from app.chains.rag_chain import RAGService
from app.core.config import Settings
from app.core.logging import get_logger
from app.integrations.qwen import create_chat_model
from app.integrations.springboot_client import SpringBootClient
from app.memory.checkpointer import RedisCheckpointerManager, create_memory_checkpointer
from app.services.chat_service import ChatService
from app.services.index_service import IndexService

logger = get_logger(__name__)


@dataclass(slots=True)
class ApplicationRuntime:
    index_service: IndexService | None
    rag_service: RAGService | None
    springboot_client: SpringBootClient
    chat_service: ChatService
    checkpointer_manager: RedisCheckpointerManager | None
    agent: Any | None

    async def close(self) -> None:
        await self.chat_service.close()
        if self.checkpointer_manager is not None:
            await self.checkpointer_manager.close()
        if self.rag_service is not None:
            await self.rag_service.aclose()
        await self.springboot_client.aclose()
        if self.index_service is not None:
            await self.index_service.close()


async def build_runtime(settings: Settings) -> ApplicationRuntime:
    index_service: IndexService | None = None
    rag_service: RAGService | None = None
    checkpointer_manager: RedisCheckpointerManager | None = None
    agent: Any | None = None

    try:
        index_service = await asyncio.wait_for(
            asyncio.to_thread(IndexService.from_settings, settings),
            timeout=15,
        )
    except Exception:
        logger.exception("index_service_initialization_failed")

    try:
        rag_service = await asyncio.wait_for(
            asyncio.to_thread(RAGService.from_settings, settings),
            timeout=15,
        )
    except Exception:
        logger.exception("rag_service_initialization_failed")

    springboot_client = SpringBootClient(timeout_seconds=settings.tool_timeout_seconds)
    try:
        saver: Any
        if settings.checkpointer_backend == "memory":
            if settings.app_env == "production":
                raise RuntimeError("memory checkpointer is forbidden in production")
            saver = create_memory_checkpointer()
        else:
            checkpointer_manager = RedisCheckpointerManager(settings)
            saver = await checkpointer_manager.start()
        agent = create_tourism_agent(
            chat_model=create_chat_model(settings),
            checkpointer=saver,
        )
    except Exception:
        logger.exception("agent_checkpointer_initialization_failed")
        if checkpointer_manager is not None:
            await checkpointer_manager.close()
        checkpointer_manager = None
        agent = None

    chat_service = ChatService.from_settings(
        springboot_client=springboot_client,
        rag_service=rag_service,
        agent=agent,
        checkpointer=checkpointer_manager.saver if checkpointer_manager else None,
        settings=settings,
    )
    return ApplicationRuntime(
        index_service=index_service,
        rag_service=rag_service,
        springboot_client=springboot_client,
        chat_service=chat_service,
        checkpointer_manager=checkpointer_manager,
        agent=agent,
    )



