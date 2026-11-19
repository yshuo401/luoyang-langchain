from __future__ import annotations

import time

from app.chains.grounding import (
    EvidenceRegistry,
    citation_failure_reason,
    finalize_grounded_result,
    parse_citation_numbers,
    source_from_business_item,
)
from app.schemas.source import SourceItem
from langchain_core.documents import Document


def test_citation_parser_supports_multiple_markers() -> None:
    assert parse_citation_numbers("结论一[S1]，结论二[S2][S3]。") == [1, 2, 3]
    assert citation_failure_reason("没有引用", 2) == "CITATION_MISSING"
    assert citation_failure_reason("错误引用[S9]", 2) == "CITATION_INVALID"
    assert citation_failure_reason("有效引用[S2]", 2) is None


def test_evidence_registry_deduplicates_by_source_identity() -> None:
    registry = EvidenceRegistry()
    first = registry.register(
        SourceItem(sourceType="SPOT", sourceId="1001", title="龙门石窟", rankNo=1)
    )
    second = registry.register(
        SourceItem(sourceType="SPOT", sourceId="1001", title="龙门石窟（重复）", rankNo=1)
    )
    assert first == second == "S1"
    assert len(registry.sources) == 1


def test_document_and_business_sources_are_built_server_side() -> None:
    registry = EvidenceRegistry()
    marker = registry.register_document(
        Document(
            page_content="建议游玩三到四小时。",
            metadata={
                "document_id": "18",
                "vector_id": "abc123",
                "title": "龙门石窟讲解",
                "_normalized_score": 0.82,
            },
        )
    )
    assert marker == "S1"
    assert registry.sources[0].source_type == "KNOWLEDGE"
    assert registry.sources[0].score == 0.82

    spot = source_from_business_item(
        "SPOT",
        {
            "id": "1001",
            "name": "龙门石窟",
            "summary": "世界文化遗产",
            "deepLink": "/pages/spot/detail?id=1001",
        },
    )
    assert spot.source_id == "1001"
    assert spot.deep_link == "/pages/spot/detail?id=1001"


def test_finalize_returns_insufficient_for_bad_citation() -> None:
    registry = EvidenceRegistry()
    registry.register(SourceItem(sourceType="SPOT", sourceId="1", title="景点", rankNo=1))
    result = finalize_grounded_result(
        answer="不存在的来源[S99]",
        evidence=registry,
        started_at=time.monotonic(),
        model_name="qwen3.8-flash",
    )
    assert result.insufficient is True
    assert result.insufficient_reason == "CITATION_INVALID"
    assert result.sources == []
