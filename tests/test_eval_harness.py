"""Tests for the rubric-based evaluation harness."""

from __future__ import annotations

from pathlib import Path

from orchestrator.eval.eval_harness import default_test_set, evaluate, evaluate_case


def test_default_test_set_points_at_real_documents() -> None:
    test_cases = default_test_set()

    assert len(test_cases) >= 2
    for case in test_cases:
        assert case.expected_facts
        for path in case.document_paths:
            assert Path(path).is_file()


def test_evaluate_case_scores_full_corpus_highly() -> None:
    test_cases = default_test_set()
    full_corpus_case = next(c for c in test_cases if c.name == "full_corpus")

    result = evaluate_case(full_corpus_case)

    assert result.missing_facts == ()
    assert result.missing_sections == ()
    assert result.factual_coverage == 1.0
    assert result.citation_validity == 1.0
    assert result.completeness == 1.0
    assert result.aggregate == 1.0


def test_evaluate_produces_report_across_all_cases() -> None:
    report = evaluate(default_test_set())

    assert len(report.case_results) == len(default_test_set())
    assert set(report.dimension_averages) == {
        "factual_coverage",
        "citation_validity",
        "completeness",
    }
    for score in report.dimension_averages.values():
        assert 0.0 <= score <= 1.0
    assert 0.0 <= report.aggregate_score <= 1.0
    # The bundled sample documents and expected-fact lists are designed so
    # the offline pipeline should score well against this rubric.
    assert report.aggregate_score >= 0.9


def test_evaluate_case_reports_missing_facts_when_absent() -> None:
    test_cases = default_test_set()
    company_only = next(c for c in test_cases if c.name == "company_only")
    tampered = company_only.__class__(
        name=company_only.name,
        document_paths=company_only.document_paths,
        expected_facts=company_only.expected_facts + ("a fact that will never appear",),
    )

    result = evaluate_case(tampered)

    assert "a fact that will never appear" in result.missing_facts
    assert result.factual_coverage < 1.0
