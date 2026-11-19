"""Chinese-friendly deterministic chunking."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loaders import ParsedDocument


@dataclass(slots=True)
class ChunkingOptions:
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunks: int = 2000


def create_splitter(options: ChunkingOptions | None = None) -> RecursiveCharacterTextSplitter:
    config = options or ChunkingOptions()
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " "],
        keep_separator=True,
    )


def split_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    document_version: int,
    title: str,
    category: str | None,
    source_url: str,
    options: ChunkingOptions | None = None,
) -> list[Document]:
    config = options or ChunkingOptions()
    splitter = create_splitter(config)
    documents: list[Document] = []
    chunk_index = 0
    for section in parsed.sections:
        for chunk in splitter.split_text(section.text):
            content = chunk.strip()
            if not content:
                continue
            if chunk_index >= config.max_chunks:
                raise ValueError("TOO_MANY_CHUNKS")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": document_id,
                        "document_version": document_version,
                        "chunk_index": chunk_index,
                        "title": title,
                        "category": category or "",
                        "source_url": source_url,
                        "page": section.page,
                        "section_path": " > ".join(section.section_path),
                        "enabled": True,
                    },
                )
            )
            chunk_index += 1
    return documents
