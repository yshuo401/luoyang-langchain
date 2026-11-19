"""Deterministic evidence registration, citation checks and fallback results."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document

from app.core.constants import DEFAULT_SUGGESTED_QUESTIONS, InsufficientReason, SourceType
from app.schemas.chat import ChatResult
from app.schemas.source import SourceItem

_CITATION = re.compile(r"\[S(\d+)\]")
_HTML = re.compile(r"<[^>]+>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def elapsed_ms(started_at: float) -> int:
    return max(0, int((time.monotonic() - started_at) * 1000))


def clean_text(value: str, *, max_chars: int = 500) -> str:
    text = _CONTROL.sub(" ", _HTML.sub(" ", value))
    return " ".join(text.split())[:max_chars]


@dataclass(slots=True)
class EvidenceRegistry:
    _sources: list[SourceItem] = field(default_factory=list)
    _keys: dict[tuple[str, str, str], int] = field(default_factory=dict)

    @property
    def sources(self) -> list[SourceItem]:
        return list(self._sources)

    def register(self, source: SourceItem) -> str:
        key = (
            source.source_type,
            source.source_id or "",
            source.chunk_id or "",
        )
        if key in self._keys:
            rank_no = self._keys[key]
            return f"S{rank_no}"
        rank_no = len(self._sources) + 1
        stored = source.model_copy(update={"rank_no": rank_no})
        self._sources.append(stored)
        self._keys[key] = rank_no
        return f"S{rank_no}"

    def register_document(self, document: Document) -> str:
        return self.register(source_from_document(document))

    def register_business_item(
        self,
        source_type: Literal["SPOT", "GUIDE", "FOOD", "RECOMMENDATION"],
        item: dict[str, Any],
    ) -> str:
        return self.register(source_from_business_item(source_type, item))


def source_from_document(document: Document) -> SourceItem:
    metadata = document.metadata
    vector_id = str(metadata.get("vector_id") or metadata.get("pk") or "")
    normalized_score = metadata.get("_normalized_score")
    score = (
        max(0.0, min(1.0, float(normalized_score)))
        if isinstance(normalized_score, int | float)
        else None
    )
    return SourceItem(
        source_type=SourceType.KNOWLEDGE,
        source_id=str(metadata.get("document_id") or "") or None,
        chunk_id=vector_id or None,
        title=str(metadata.get("title") or "知识库资料"),
        snippet=clean_text(document.page_content, max_chars=500),
        score=score,
        deep_link=None,
    )


def source_from_business_item(
    source_type: Literal["SPOT", "GUIDE", "FOOD", "RECOMMENDATION"],
    item: dict[str, Any],
) -> SourceItem:
    item_id = item.get("id") or item.get("spotId") or item.get("guideId") or item.get("foodId")
    title = item.get("name") or item.get("title") or "旅游业务数据"
    summary = item.get("summary") or item.get("descriptionText") or item.get("reason")
    return SourceItem(
        source_type=source_type,
        source_id=str(item_id) if item_id is not None else None,
        chunk_id=None,
        title=str(title),
        snippet=clean_text(str(summary)) if summary else None,
        score=None,
        deep_link=item.get("deepLink") if isinstance(item.get("deepLink"), str) else None,
    )


def parse_citation_numbers(answer: str) -> list[int]:
    return [int(number) for number in _CITATION.findall(answer)]


def citation_failure_reason(answer: str, source_count: int) -> str | None:
    citations = parse_citation_numbers(answer)
    if not citations:
        return InsufficientReason.CITATION_MISSING
    if any(number < 1 or number > source_count for number in citations):
        return InsufficientReason.CITATION_INVALID
    return None


def suggested_questions(sources: list[SourceItem]) -> list[str]:
    types = {source.source_type for source in sources}
    if SourceType.KNOWLEDGE in types:
        return ["这个景点附近还有什么值得去？", "帮我安排半天的游览路线。"]
    if SourceType.GUIDE in types:
        return ["这条路线适合老人吗？", "附近有什么洛阳特色美食？"]
    if SourceType.FOOD in types:
        return ["这些美食在哪些景点附近？", "洛阳还有哪些特色小吃？"]
    if SourceType.SPOT in types or SourceType.RECOMMENDATION in types:
        return ["帮我安排半天的游览路线。", "这些景点附近有什么美食？"]
    return list(DEFAULT_SUGGESTED_QUESTIONS)


def insufficient_result(
    reason: str,
    *,
    started_at: float,
    model_name: str | None = None,
    answer: str = "暂未找到可靠资料，暂时无法给出有依据的回答。",
    token_usage: dict[str, Any] | None = None,
) -> ChatResult:
    return ChatResult(
        answer=answer,
        grounded=False,
        insufficient=True,
        insufficient_reason=reason,
        sources=[],
        suggested_questions=list(DEFAULT_SUGGESTED_QUESTIONS),
        model_name=model_name,
        latency_ms=elapsed_ms(started_at),
        token_usage=token_usage,
    )


def finalize_grounded_result(
    *,
    answer: str,
    evidence: EvidenceRegistry,
    started_at: float,
    model_name: str,
    token_usage: dict[str, Any] | None = None,
) -> ChatResult:
    sources = evidence.sources
    if not sources:
        return insufficient_result(
            InsufficientReason.RETRIEVAL_EMPTY,
            started_at=started_at,
            model_name=model_name,
            token_usage=token_usage,
        )
    failure = citation_failure_reason(answer, len(sources))
    if failure:
        return insufficient_result(
            failure,
            started_at=started_at,
            model_name=model_name,
            answer="当前回答未能通过来源校验，因此不展示未经验证的内容。",
            token_usage=token_usage,
        )
    return ChatResult(
        answer=clean_text(answer, max_chars=4000),
        grounded=True,
        insufficient=False,
        sources=sources,
        suggested_questions=suggested_questions(sources),
        model_name=model_name,
        latency_ms=elapsed_ms(started_at),
        token_usage=token_usage,
    )
