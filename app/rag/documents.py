"""文档加载与切分。

- TXT / Markdown 直接读取；
- PDF / DOC / DOCX / 图片交给 MinerU；
- MinerU 返回 Markdown 后，按标题和长度切分。
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from mineru import MinerU

from app.config import Settings, get_settings
from app.schemas import IndexJobSubmitRequest

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_MINERU_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


class DocumentParseError(Exception):
    """文档解析失败；`code` 会写入索引任务状态。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("PARSE_ERROR", "文本编码不受支持")


def sanitize_text(text: str) -> str:
    """去掉控制字符和空行，保留 Markdown 标题结构。"""

    cleaned = _CONTROL.sub("", text)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


@contextmanager
def _mineru_proxy_context(enabled: bool):
    """仅对 MinerU 同步调用临时移除代理环境变量。

    Windows 上常见的全局 HTTP_PROXY 可能让 MinerU 结果 ZIP 下载失败。
    调用结束后立即恢复原值。
    """

    if not enabled:
        yield
        return

    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def extract_markdown(
    content: bytes,
    file_name: str,
    settings: Settings | None = None,
) -> str:
    """把上传文件转成 Markdown；复杂文档使用 MinerU。"""

    if not content:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档内容为空")

    config = settings or get_settings()
    suffix = Path(file_name).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        text = sanitize_text(_decode_text(content))
        if not text:
            raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可解析正文")
        return text

    if suffix not in _MINERU_SUFFIXES:
        raise DocumentParseError(
            "UNSUPPORTED_FILE_TYPE",
            "仅支持 TXT、Markdown、PDF、DOC/DOCX 和常见图片格式",
        )

    token = config.mineru_api_key.get_secret_value()
    if not token:
        raise DocumentParseError(
            "MINERU_NOT_CONFIGURED",
            "解析 PDF/DOCX/图片需要配置 MINERU_API_KEY",
        )

    with tempfile.TemporaryDirectory(prefix="luoyang-ai-") as temp_dir:
        local_path = Path(temp_dir) / (Path(file_name).name or f"document{suffix}")
        local_path.write_bytes(content)

        kwargs: dict[str, object] = {"token": token}
        if config.mineru_base_url:
            kwargs["base_url"] = config.mineru_base_url
        client = MinerU(**kwargs)
        try:
            with _mineru_proxy_context(config.mineru_bypass_proxy):
                result = client.extract(
                    str(local_path),
                    model=config.mineru_model,
                    ocr=True,
                    timeout=config.mineru_timeout_seconds,
                )
        except Exception as exc:
            raise DocumentParseError("MINERU_FAILED", "MinerU 文档解析失败") from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    if result.state != "done" or not result.markdown:
        raise DocumentParseError("MINERU_FAILED", result.error or "MinerU 没有返回 Markdown")

    markdown = sanitize_text(result.markdown)
    if not markdown:
        raise DocumentParseError("EMPTY_DOCUMENT", "MinerU 返回的正文为空")
    return markdown


def split_markdown(
    markdown: str,
    request: IndexJobSubmitRequest,
    *,
    settings: Settings | None = None,
) -> list[Document]:
    """先按标题切分，再按长度切分并补齐索引元数据。"""

    config = settings or get_settings()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "，", " "],
        keep_separator=True,
    )

    sections = header_splitter.split_text(markdown)
    if not sections:
        sections = [Document(page_content=markdown, metadata={})]

    documents: list[Document] = []
    for section in sections:
        for chunk in text_splitter.split_documents([section]):
            content = chunk.page_content.strip()
            if not content:
                continue
            if len(documents) >= config.index_max_chunks:
                raise DocumentParseError("TOO_MANY_CHUNKS", "文档分片数超过上限")

            metadata = dict(chunk.metadata)
            heading_parts = [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
            metadata.update(
                {
                    "document_id": request.document_id,
                    "document_version": request.document_version,
                    "chunk_index": len(documents),
                    "title": request.title,
                    "category": request.category or "",
                    "source_url": "",
                    "section_path": " > ".join(heading_parts),
                    "enabled": True,
                }
            )
            documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可索引的分片")
    return documents
