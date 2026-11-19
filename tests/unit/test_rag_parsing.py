from __future__ import annotations

import pytest
from app.rag.loaders import DocumentParseError, parse_document, sanitize_text
from app.rag.splitter import ChunkingOptions, split_document


def test_txt_loader_sanitizes_scripts_and_control_chars() -> None:
    content = "洛阳龙门石窟<script>alert(1)</script>位于洛龙区\x00。".encode()
    parsed = parse_document(content, "guide.txt")
    assert parsed.char_count > 0
    assert "script" not in parsed.sections[0].text
    assert "龙门石窟" in parsed.sections[0].text


def test_markdown_loader_preserves_heading_path() -> None:
    content = "# 龙门石窟\n\n## 游览建议\n\n建议游玩三到四小时。".encode()
    parsed = parse_document(content, "guide.md")
    assert any("游览建议" in item.section_path for item in parsed.sections)
    assert "三到四小时" in parsed.sections[0].text


def test_unsupported_and_empty_documents_are_rejected() -> None:
    with pytest.raises(DocumentParseError) as unsupported:
        parse_document(b"abc", "guide.exe")
    assert unsupported.value.code == "UNSUPPORTED_FILE_TYPE"

    with pytest.raises(DocumentParseError) as empty:
        parse_document(b"   ", "guide.txt")
    assert empty.value.code == "EMPTY_DOCUMENT"


def test_splitter_assigns_stable_chunk_metadata() -> None:
    content = ("龙门石窟建议游玩三到四小时。" * 100).encode()
    parsed = parse_document(content, "guide.txt")
    documents = split_document(
        parsed,
        document_id="18",
        document_version=1,
        title="龙门石窟",
        category="SCENIC_INTRO",
        source_url="",
        options=ChunkingOptions(chunk_size=200, chunk_overlap=30, max_chunks=20),
    )
    assert len(documents) > 1
    assert [doc.metadata["chunk_index"] for doc in documents] == list(range(len(documents)))
    assert all(doc.metadata["enabled"] is True for doc in documents)
    assert documents[0].metadata["document_version"] == 1


def test_clean_text_removes_event_attributes() -> None:
    assert "onclick" not in sanitize_text('<a onclick="steal()">文字</a>')
