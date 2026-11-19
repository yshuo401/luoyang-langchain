"""Bounded downloader for short-lived OSS signed URLs."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import ServiceError


@dataclass(slots=True)
class DownloadedFile:
    content: bytes
    content_type: str | None
    file_name: str | None


class DownloadError(ServiceError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(50301, message, 502, retryable=True)
        self.error_code = code


class DocumentDownloader:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=20.0, write=10.0, pool=3.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def download(self, url: str, *, file_name: str | None = None) -> DownloadedFile:
        chunks: list[bytes] = []
        total = 0
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise DownloadError("文档下载失败", code="DOWNLOAD_HTTP_ERROR")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._settings.download_max_bytes:
                        raise DownloadError("文档超过允许大小", code="DOCUMENT_TOO_LARGE")
                    chunks.append(chunk)
                return DownloadedFile(
                    content=b"".join(chunks),
                    content_type=response.headers.get("content-type"),
                    file_name=file_name or self._name_from_disposition(
                        response.headers.get("content-disposition")
                    ),
                )
        except httpx.HTTPError as exc:
            raise DownloadError("文档下载失败", code="DOWNLOAD_ERROR") from exc

    @staticmethod
    def _name_from_disposition(value: str | None) -> str | None:
        if not value:
            return None
        for part in value.split(";"):
            key, _, raw = part.strip().partition("=")
            if key.lower() == "filename":
                return raw.strip('"') or None
        return None
