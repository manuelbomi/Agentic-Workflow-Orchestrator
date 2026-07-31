"""Rubric-based evaluation harness for the research-to-memo pipeline.

Runs the full pipeline against a small labeled test set and scores the
resulting memo on three dimensions, each in the ``0.0..1.0`` range:

* **factual_coverage** -- fraction of a test case's expected key facts that
  appear verbatim (case-insensitive) somewhere in the generated memo.
* **citation_validity** -- fraction of citations in the generated memo that
  the ``citation_checker`` tool confirmed were valid, as reported by the
  writer agent.
* **completeness** -- fraction of the memo's required sections
  (``writer_agent.REQUIRED_MEMO_SECTIONS``) that are actually present in
  the output.

The aggregate score for a test case is the unweighted mean of the three
dimensions. This rubric is intentionally simple and legible -- a real
deployment would likely calibrate per-dimension weights and add a
semantic-similarity or LLM-judge dimension -- but even this lightweight
version directly catches the two failure modes that matter most for
advisory work: missing facts and unsupported claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from orchestrator.agents.writer_agent import REQUIRED_MEMO_SECTIONS
from orchestrator.pipeline import PipelineResult, run_pipeline

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "sample_documents"


@dataclass(frozen=True)
class LabeledTestCase:
    """One labeled test case: a document set plus its expected key facts.

    Attributes:
        name: Short identifier for the test case.
        document_paths: Source documents to run the pipeline against.
        expected_facts: Strings expected to appear verbatim
            (case-insensitive) somewhere in the generated memo if the
            pipeline correctly surfaced the corresponding source facts.
    """

    name: str
    document_paths: tuple[str, ...]
    expected_facts: tuple[str, ...]


@dataclass
class EvalCaseResult:
    """Rubric scores for a single test case."""

    case_name: str
    factual_coverage: float
    citation_validity: float
    completeness: float
    aggregate: float
    missing_facts: tuple[str, ...] = field(default_factory=tuple)
    missing_sections: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class EvalReport:
    """Aggregate evaluation results across an entire labeled test set."""

    case_results: list[EvalCaseResult]
    dimension_averages: dict[str, float]
    aggregate_score: float


def _score_factual_coverage(
    memo_markdown: str, expected_facts: Sequence[str]
) -> tuple[float, tuple[str, ...]]:
    if not expected_facts:
        return 1.0, ()
    lowered_memo = memo_markdown.lower()
    missing = tuple(fact for fact in expected_facts if fact.lower() not in lowered_memo)
    covered = len(expected_facts) - len(missing)
    return covered / len(expected_facts), missing


def _score_completeness(memo_markdown: str) -> tuple[float, tuple[str, ...]]:
    missing = tuple(
        section for section in REQUIRED_MEMO_SECTIONS if f"## {section}" not in memo_markdown
    )
    covered = len(REQUIRED_MEMO_SECTIONS) - len(missing)
    return covered / len(REQUIRED_MEMO_SECTIONS), missing


def _score_citation_validity(citation_results: dict) -> float:
    if not citation_results:
        return 0.0
    valid = sum(1 for result in citation_results.values() if result.get("is_valid"))
    return valid / len(citation_results)


def evaluate_case(test_case: LabeledTestCase) -> EvalCaseResult:
    """Runs the pipeline once for a single labeled test case and scores its output.

    Args:
        test_case: The document set and expected facts to evaluate.

    Returns:
        An ``EvalCaseResult`` with per-dimension and aggregate scores.
    """
    result: PipelineResult = run_pipeline(list(test_case.document_paths))
    memo_markdown = result.memo_markdown or ""

    factual_coverage, missing_facts = _score_factual_coverage(
        memo_markdown, test_case.expected_facts
    )
    completeness, missing_sections = _score_completeness(memo_markdown)
    citation_validity = _score_citation_validity(dict(result.citation_results))

    aggregate = (factual_coverage + citation_validity + completeness) / 3

    return EvalCaseResult(
        case_name=test_case.name,
        factual_coverage=round(factual_coverage, 4),
        citation_validity=round(citation_validity, 4),
        completeness=round(completeness, 4),
        aggregate=round(aggregate, 4),
        missing_facts=missing_facts,
        missing_sections=missing_sections,
    )


def evaluate(test_cases: Sequence[LabeledTestCase]) -> EvalReport:
    """Runs and scores the pipeline across an entire labeled test set.

    Args:
        test_cases: The labeled test set to evaluate.

    Returns:
        An ``EvalReport`` with per-case results plus per-dimension and
        overall aggregate averages.
    """
    case_results = [evaluate_case(case) for case in test_cases]

    dimension_names = ("factual_coverage", "citation_validity", "completeness")
    if case_results:
        dimension_averages = {
            dim: round(sum(getattr(r, dim) for r in case_results) / len(case_results), 4)
            for dim in dimension_names
        }
        aggregate_score = round(sum(r.aggregate for r in case_results) / len(case_results), 4)
    else:
        dimension_averages = {dim: 0.0 for dim in dimension_names}
        aggregate_score = 0.0

    return EvalReport(
        case_results=case_results,
        dimension_averages=dimension_averages,
        aggregate_score=aggregate_score,
    )


def default_test_set(data_dir: str | Path | None = None) -> list[LabeledTestCase]:
    """Builds the default labeled test set used by CI, the CLI, and the charts script.

    Three test cases are drawn from the three bundled sample documents in
    different combinations, each with a small set of facts that must appear
    in the resulting memo for the case to score well on factual coverage.

    Args:
        data_dir: Directory containing the sample documents. Defaults to
            the repository's ``data/sample_documents`` directory.

    Returns:
        A list of ``LabeledTestCase`` covering the full corpus and two
        partial-document-set scenarios.
    """
    base = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    company = str(base / "company_notes.txt")
    market = str(base / "market_notes.txt")
    risk = str(base / "risk_notes.txt")

    return [
        LabeledTestCase(
            name="full_corpus",
            document_paths=(company, market, risk),
            expected_facts=(
                "42 million dollars",
                "610 active customers",
                "8.2 billion dollars",
                "34 percent",
                "Dana Whitfield",
            ),
        ),
        LabeledTestCase(
            name="company_only",
            document_paths=(company,),
            expected_facts=(
                "42 million dollars",
                "610 active customers",
                "340 employees",
                "108 percent",
            ),
        ),
        LabeledTestCase(
            name="market_and_risk",
            document_paths=(market, risk),
            expected_facts=(
                "8.2 billion dollars",
                "14 percent",
                "34 percent",
                "Data Handling Act of 2025",
            ),
        ),
    ]


def _print_report(report: EvalReport) -> None:
    print(f"Aggregate score: {report.aggregate_score:.2f}")
    for dimension, score in report.dimension_averages.items():
        print(f"  {dimension}: {score:.2f}")
    print()
    for case in report.case_results:
        print(
            f"[{case.case_name}] aggregate={case.aggregate:.2f} "
            f"factual_coverage={case.factual_coverage:.2f} "
            f"citation_validity={case.citation_validity:.2f} "
            f"completeness={case.completeness:.2f}"
        )
        if case.missing_facts:
            print(f"    missing facts: {list(case.missing_facts)}")
        if case.missing_sections:
            print(f"    missing sections: {list(case.missing_sections)}")


if __name__ == "__main__":
    _print_report(evaluate(default_test_set()))
