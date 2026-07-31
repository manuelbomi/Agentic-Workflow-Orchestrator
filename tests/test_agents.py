"""Tests for the four concrete pipeline agents, using the deterministic offline stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.agents.analyst_agent import AnalystAgent
from orchestrator.agents.researcher_agent import ResearcherAgent
from orchestrator.agents.reviewer_agent import ReviewerAgent, build_checklist
from orchestrator.agents.writer_agent import REQUIRED_MEMO_SECTIONS, WriterAgent
from orchestrator.tool_registry import ToolPermissionError, ToolRegistry
from orchestrator.tools import calculator_tool, citation_checker_tool, document_parser_tool

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
COMPANY_NOTES = str(DATA_DIR / "company_notes.txt")


def _full_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(document_parser_tool.build_spec())
    registry.register(calculator_tool.build_spec())
    registry.register(citation_checker_tool.build_spec())
    registry.grant("researcher_agent", ["document_parser"])
    registry.grant("analyst_agent", ["calculator"])
    registry.grant("writer_agent", ["citation_checker"])
    return registry


# --- researcher_agent -------------------------------------------------------


def test_researcher_agent_extracts_source_attributed_findings() -> None:
    registry = _full_registry()
    researcher = ResearcherAgent(registry)

    output = researcher.act({"document_paths": [COMPANY_NOTES]})

    findings = output.content["findings"]
    assert len(findings) > 0
    for finding in findings:
        assert finding["source_id"] == "company_notes"
        assert finding["chunk_id"].startswith("company_notes#")
        assert finding["text"]


def test_researcher_agent_requires_tool_permission() -> None:
    """Without a grant for document_parser, the researcher must not be able to run."""
    registry = ToolRegistry()
    registry.register(document_parser_tool.build_spec())
    # Deliberately no grant for researcher_agent.
    researcher = ResearcherAgent(registry)

    with pytest.raises(ToolPermissionError):
        researcher.act({"document_paths": [COMPANY_NOTES]})


# --- analyst_agent -----------------------------------------------------------


def test_analyst_agent_classifies_signals_and_scores_recommendation() -> None:
    registry = ToolRegistry()
    registry.register(calculator_tool.build_spec())
    registry.grant("analyst_agent", ["calculator"])
    analyst = AnalystAgent(registry)

    findings = [
        {
            "finding_id": "f0",
            "source_id": "doc",
            "chunk_id": "doc#0",
            "text": "Revenue increased due to strong growth.",
            "is_numeric": False,
        },
        {
            "finding_id": "f1",
            "source_id": "doc",
            "chunk_id": "doc#1",
            "text": "Customer churn risk increased due to concentration.",
            "is_numeric": False,
        },
    ]

    output = analyst.act({"findings": findings})
    analysis = output.content

    assert "Doc" in analysis["themes"]
    assert len(analysis["risks"]) == 1
    assert len(analysis["opportunities"]) == 1
    # (1 opportunity - 1 risk) / (1 + 1) * 100 == 0.0
    assert analysis["recommendation"]["score"] == 0.0
    assert analysis["recommendation"]["label"] == "Mixed / Monitor"


def test_analyst_agent_favors_opportunity_heavy_findings() -> None:
    registry = ToolRegistry()
    registry.register(calculator_tool.build_spec())
    registry.grant("analyst_agent", ["calculator"])
    analyst = AnalystAgent(registry)

    findings = [
        {
            "finding_id": f"f{i}",
            "source_id": "doc",
            "chunk_id": f"doc#{i}",
            "text": "Strong growth and expansion opportunity.",
            "is_numeric": False,
        }
        for i in range(3)
    ]

    output = analyst.act({"findings": findings})

    assert output.content["recommendation"]["score"] == 100.0
    assert output.content["recommendation"]["label"] == "Favorable"


# --- reviewer_agent ----------------------------------------------------------


def test_reviewer_agent_auto_approves_complete_analysis() -> None:
    registry = ToolRegistry()
    reviewer = ReviewerAgent(registry)

    analysis = {
        "themes": {"Doc": [{"text": "x"}]},
        "risks": [{"text": "risk"}],
        "opportunities": [],
        "recommendation": {"score": 10.0, "label": "Mixed / Monitor", "rationale": "because"},
    }

    output = reviewer.act({"analysis": analysis})

    assert output.content["approved"] is True
    assert all(output.content["checklist"].values())


def test_reviewer_agent_rejects_incomplete_analysis() -> None:
    registry = ToolRegistry()
    reviewer = ReviewerAgent(registry)

    empty_analysis = {"themes": {}, "risks": [], "opportunities": [], "recommendation": {}}

    output = reviewer.act({"analysis": empty_analysis})

    assert output.content["approved"] is False
    assert output.content["checklist"] == build_checklist(empty_analysis)
    assert not all(output.content["checklist"].values())


# --- writer_agent ------------------------------------------------------------


def _researcher_and_analyst_output(registry: ToolRegistry):
    researcher = ResearcherAgent(registry)
    analyst = AnalystAgent(registry)
    researcher_output = researcher.act({"document_paths": [COMPANY_NOTES]})
    analyst_output = analyst.act({"findings": researcher_output.content["findings"]})
    return researcher_output, analyst_output


def test_writer_agent_produces_all_required_sections_with_valid_citations() -> None:
    registry = _full_registry()
    researcher_output, analyst_output = _researcher_and_analyst_output(registry)
    writer = WriterAgent(registry)

    output = writer.act(
        {
            "approved": True,
            "analysis": analyst_output.content,
            "findings": researcher_output.content["findings"],
            "chunks": researcher_output.content["chunks"],
        }
    )

    memo = output.content["memo_markdown"]
    assert output.content["status"] == "complete"
    assert memo is not None
    for section in REQUIRED_MEMO_SECTIONS:
        assert f"## {section}" in memo

    citation_results = output.content["citation_results"]
    assert len(citation_results) > 0
    assert all(result["is_valid"] for result in citation_results.values())


def test_writer_agent_blocks_when_not_approved() -> None:
    registry = _full_registry()
    researcher_output, analyst_output = _researcher_and_analyst_output(registry)
    writer = WriterAgent(registry)

    output = writer.act(
        {
            "approved": False,
            "analysis": analyst_output.content,
            "findings": researcher_output.content["findings"],
            "chunks": researcher_output.content["chunks"],
        }
    )

    assert output.content["memo_markdown"] is None
    assert output.content["status"] == "blocked_pending_revision"
