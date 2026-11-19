"""Safe PDF / DOCX / Markdown / TXT parsing into section-level text."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

_SCRIPT_BLOCK = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_EVENT_ATTR = re.compile(r"\bon\w+\s*=", re.IGNORECASE)
_JS_URL = re.compile(r"javascript\s*:", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(slots=True)
class TextSection:
    text: str
    page: int | None = None
    section_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    file_name: str
    sections: list[TextSection]

    @property
    def char_count(self) -> int:
        return sum(len(section.text) for section in self.sections)


class DocumentParseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sanitize_text(text: str) -> str:
    cleaned = _SCRIPT_BLOCK.sub(" ", text)
    cleaned = _CONTROL.sub(" ", cleaned)
    cleaned = _EVENT_ATTR.sub(" ", cleaned)
    cleaned = _JS_URL.sub(" ", cleaned)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_document(content: bytes, file_name: str) -> ParsedDocument:
    if not content:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档内容为空")
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        sections = _parse_pdf(content)
    elif suffix == ".docx":
        sections = _parse_docx(content)
    elif suffix in {".md", ".markdown"}:
        sections = _parse_markdown(content)
    elif suffix == ".txt":
        sections = _parse_txt(content)
    else:
        raise DocumentParseError("UNSUPPORTED_FILE_TYPE", "仅支持 PDF、DOCX、Markdown、TXT")

    normalized = [
        TextSection(
            text=sanitize_text(section.text),
            page=section.page,
            section_path=section.section_path,
        )
        for section in sections
        if sanitize_text(section.text)
    ]
    if not normalized or sum(len(item.text) for item in normalized) < 2:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可解析正文")
    return ParsedDocument(file_name=file_name, sections=normalized)


def _parse_pdf(content: bytes) -> list[TextSection]:
    try:
        reader = PdfReader(io.BytesIO(content))
        return [
            TextSection(text=page.extract_text() or "", page=index + 1, section_path=["正文"])
            for index, page in enumerate(reader.pages)
        ]
    except Exception as exc:
        raise DocumentParseError("PARSE_ERROR", "PDF 解析失败") from exc


def _parse_docx(content: bytes) -> list[TextSection]:
    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception as exc:
        raise DocumentParseError("PARSE_ERROR", "DOCX 解析失败") from exc

    sections: list[TextSection] = []
    heading_path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(
                TextSection(text="\n".join(buffer), section_path=list(heading_path) or ["正文"])
            )
            buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = (paragraph.style.name if paragraph.style else "").lower()
        if style_name.startswith("heading"):
            flush()
            level_match = re.search(r"(\d+)", style_name)
            level = int(level_match.group(1)) if level_match else 1
            heading_path[:] = heading_path[: max(level - 1, 0)]
            heading_path.append(text)
        else:
            buffer.append(text)
    flush()
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        table_text = "\n".join(row for row in rows if row.replace("|", "").strip())
        if table_text:
            sections.append(TextSection(text=table_text, section_path=["表格"]))
    return sections


def _parse_markdown(content: bytes) -> list[TextSection]:
    text = _decode_text(content)
    sections: list[TextSection] = []
    heading_path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(
                TextSection(text="\n".join(buffer), section_path=list(heading_path) or ["正文"])
            )
            buffer.clear()

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading_path[:] = heading_path[: max(level - 1, 0)]
            heading_path.append(match.group(2))
        elif line.strip():
            buffer.append(line)
    flush()
    return sections


def _parse_txt(content: bytes) -> list[TextSection]:
    return [TextSection(text=_decode_text(content), section_path=["正文"])]


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = content.decode(encoding)
            return text
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("PARSE_ERROR", "文本编码不受支持或文件伪装为文本")

