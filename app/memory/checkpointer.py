"""Lifecycle management for the official LangGraph Redis checkpointer."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.config import Settings, get_settings


class RedisCheckpointerManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._context: Any = None
        self.saver: AsyncRedisSaver | None = None

    async def start(self) -> AsyncRedisSaver:
        self._context = AsyncRedisSaver.from_conn_string(
            self._settings.redis_url,
            ttl={
                "default_ttl": self._settings.checkpoint_ttl_seconds / 60,
                "refresh_on_read": True,
            },
            checkpoint_prefix=self._settings.checkpoint_key_prefix,
        )
        saver = await self._context.__aenter__()
        try:
            await asyncio.wait_for(
                saver.asetup(),
                timeout=self._settings.checkpoint_setup_timeout_seconds,
            )
        except Exception:
            await self._context.__aexit__(None, None, None)
            self._context = None
            raise
        self.saver = saver
        return saver

    async def reset(self, thread_id: str) -> None:
        if self.saver is None:
            raise RuntimeError("checkpointer is not ready")
        await self.saver.adelete_thread(thread_id)

    async def close(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)
        self._context = None
        self.saver = None

    async def __aenter__(self) -> AsyncRedisSaver:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.close()


def create_memory_checkpointer() -> InMemorySaver:
    """Official in-memory checkpointer for tests and local fake-agent scenarios."""

    return InMemorySaver()
