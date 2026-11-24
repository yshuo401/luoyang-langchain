"""OSS 临时签名 URL 下载器。"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings, get_settings


@dataclass(slots=True)
class DownloadedFile:
    content: bytes
    content_type: str | None
    file_name: str | None


class DownloadError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentDownloader:
    """下载知识文档，限制最大文件大小。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=20.0, write=10.0, pool=3.0),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def download(self, url: str, *, file_name: str | None = None) -> DownloadedFile:
        chunks: list[bytes] = []
        total = 0
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise DownloadError("DOWNLOAD_HTTP_ERROR", "文档下载失败")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.settings.download_max_bytes:
                        raise DownloadError("DOCUMENT_TOO_LARGE", "文档超过允许大小")
                    chunks.append(chunk)
                return DownloadedFile(
                    content=b"".join(chunks),
                    content_type=response.headers.get("content-type"),
                    file_name=file_name
                    or self._name_from_disposition(response.headers.get("content-disposition")),
                )
        except httpx.HTTPError as exc:
            raise DownloadError("DOWNLOAD_ERROR", "文档下载失败") from exc

    @staticmethod
    def _name_from_disposition(value: str | None) -> str | None:
        if not value:
            return None
        for part in value.split(";"):
            key, _, raw = part.strip().partition("=")
            if key.lower() == "filename":
                return raw.strip('"') or None
        return None
