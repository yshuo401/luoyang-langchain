"""Short-lived chat result cache and inflight locks."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from redis.asyncio import Redis

from app.core.config import Settings, get_settings


class ChatStateStore(Protocol):
    async def get_json(self, key: str) -> dict[str, Any] | None: ...

    async def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None: ...

    async def set_if_absent(self, key: str, value: str, *, ttl_seconds: int) -> bool: ...

    async def delete(self, key: str) -> None: ...

    async def close(self) -> None: ...


class MemoryChatStateStore:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def get_json(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self._values.get(key)
        return json.loads(value) if value is not None else None

    async def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        del ttl_seconds
        async with self._lock:
            self._values[key] = json.dumps(value, ensure_ascii=False)

    async def set_if_absent(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        del ttl_seconds
        async with self._lock:
            if key in self._values:
                return False
            self._values[key] = value
            return True

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._values.pop(key, None)

    async def close(self) -> None:
        return None


class RedisChatStateStore:
    def __init__(self, settings: Settings | None = None) -> None:
        config = settings or get_settings()
        self._redis = Redis.from_url(
            config.redis_url,
            socket_connect_timeout=config.redis_socket_timeout_seconds,
            socket_timeout=config.redis_socket_timeout_seconds,
        )

    async def get_json(self, key: str) -> dict[str, Any] | None:
        raw = await self._redis.get(key)
        return json.loads(raw) if raw is not None else None

    async def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        await self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    async def set_if_absent(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        result = await self._redis.set(key, value, nx=True, ex=ttl_seconds)
        return bool(result)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def close(self) -> None:
        await self._redis.aclose()
