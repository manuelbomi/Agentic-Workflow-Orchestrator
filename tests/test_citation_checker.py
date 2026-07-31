"""Tests for the citation checker tool's heuristic validation logic."""

from __future__ import annotations

from orchestrator.tools.citation_checker_tool import check_citation

SOURCE_CHUNKS = [
    {
        "source_id": "company_notes",
        "chunk_id": "company_notes#0",
        "text": "Annual recurring revenue reached 42 million dollars, up 28 percent year over year.",
        "paragraph_index": 0,
    },
    {
        "source_id": "company_notes",
        "chunk_id": "company_notes#1",
        "text": "Headcount stands at 340 employees across engineering and product roles.",
        "paragraph_index": 1,
    },
]


def test_exact_source_sentence_is_valid() -> None:
    result = check_citation(
        claim_text="Annual recurring revenue reached 42 million dollars, up 28 percent year over year.",
        chunk_id="company_notes#0",
        source_chunks=SOURCE_CHUNKS,
    )

    assert result["chunk_found"] is True
    assert result["is_valid"] is True
    assert result["overlap_ratio"] > 0.9


def test_unknown_chunk_id_is_invalid() -> None:
    result = check_citation(
        claim_text="Annual recurring revenue reached 42 million dollars.",
        chunk_id="does_not_exist#0",
        source_chunks=SOURCE_CHUNKS,
    )

    assert result["chunk_found"] is False
    assert result["is_valid"] is False
    assert result["overlap_ratio"] == 0.0


def test_unrelated_claim_against_real_chunk_is_invalid() -> None:
    result = check_citation(
        claim_text="The weather in a distant unrelated city was unusually mild this spring.",
        chunk_id="company_notes#1",
        source_chunks=SOURCE_CHUNKS,
    )

    assert result["chunk_found"] is True
    assert result["is_valid"] is False
    assert result["overlap_ratio"] < 0.3


def test_paraphrase_with_partial_overlap_can_pass_threshold() -> None:
    result = check_citation(
        claim_text="Headcount is 340 employees.",
        chunk_id="company_notes#1",
        source_chunks=SOURCE_CHUNKS,
        overlap_threshold=0.3,
    )

    assert result["is_valid"] is True


def test_overlap_threshold_is_configurable() -> None:
    lenient = check_citation(
        claim_text="Roughly 340 people work there today, give or take a few.",
        chunk_id="company_notes#1",
        source_chunks=SOURCE_CHUNKS,
        overlap_threshold=0.1,
    )
    strict = check_citation(
        claim_text="Roughly 340 people work there today, give or take a few.",
        chunk_id="company_notes#1",
        source_chunks=SOURCE_CHUNKS,
        overlap_threshold=0.9,
    )

    assert lenient["is_valid"] is True
    assert strict["is_valid"] is False
