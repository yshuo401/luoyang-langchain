"""Replay protection for signed internal requests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from redis.asyncio import Redis

from app.core.config import Settings, get_settings


class ReplayGuard:
    """Store one signature per request ID; identical retries remain idempotent."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._redis = Redis.from_url(
            self._settings.redis_url,
            socket_connect_timeout=self._settings.redis_socket_timeout_seconds,
            socket_timeout=self._settings.redis_socket_timeout_seconds,
        )
        self._memory: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, request_id: str, signature: str) -> None:
        key = f"internal:nonce:{request_id}"
        try:
            created = await self._redis.set(key, signature, nx=True, ex=600)
            if created:
                return
            existing = await self._redis.get(key)
            existing_text = existing.decode() if isinstance(existing, bytes) else existing
            if existing is None or existing_text == signature:
                return
            from app.core.errors import InternalAuthError

            raise InternalAuthError("内部请求标识被重复使用")
        except Exception as exc:
            from app.core.errors import InternalAuthError

            if isinstance(exc, InternalAuthError):
                raise
            await self._check_memory(key, signature)

    async def _check_memory(self, key: str, signature: str) -> None:
        now = time.monotonic()
        async with self._lock:
            expired = [name for name, (_, expires) in self._memory.items() if expires <= now]
            for name in expired:
                self._memory.pop(name, None)
            existing = self._memory.get(key)
            if existing is not None and existing[0] != signature:
                from app.core.errors import InternalAuthError

                raise InternalAuthError("内部请求标识被重复使用")
            self._memory[key] = (signature, now + 600)

    async def close(self) -> None:
        await self._redis.aclose()


def attach_replay_guard(app: Any, guard: ReplayGuard) -> None:
    app.state.replay_guard = guard

