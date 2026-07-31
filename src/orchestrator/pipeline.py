"""Shared pipeline wiring used by both the example script and the eval harness.

This module builds the tool registry (with each agent's permission grants),
constructs the four agents, wires them into an ``OrchestrationGraph``, and
runs the graph end to end -- including the human-in-the-loop pause at the
reviewer stage. Keeping this wiring in one place means ``examples/run_pipeline.py``
and ``orchestrator.eval.eval_harness`` exercise exactly the same pipeline
rather than two subtly different copies of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from orchestrator.agents.analyst_agent import AnalystAgent
from orchestrator.agents.reviewer_agent import ApprovalFn, ReviewerAgent, default_auto_approve
from orchestrator.agents.researcher_agent import ResearcherAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.orchestration_graph import (
    GraphNode,
    NodeType,
    OrchestrationGraph,
    WorkflowState,
    WorkflowStatus,
)
from orchestrator.tool_registry import ToolRegistry
from orchestrator.tools import calculator_tool, citation_checker_tool, document_parser_tool


def build_tool_registry() -> ToolRegistry:
    """Builds a ``ToolRegistry`` with every pipeline tool registered and granted.

    Permission scopes granted here are the concrete instance of the
    MCP-style tool-permission pattern described in the project README:
    each agent can only call the tool(s) its role actually needs.
    ``reviewer_agent`` is intentionally granted no tools at all, since its
    job is a policy decision over already-computed analysis, not a new
    computation.
    """
    registry = ToolRegistry()
    registry.register(document_parser_tool.build_spec())
    registry.register(calculator_tool.build_spec())
    registry.register(citation_checker_tool.build_spec())

    registry.grant("researcher_agent", ["document_parser"])
    registry.grant("analyst_agent", ["calculator"])
    registry.grant("writer_agent", ["citation_checker"])

    return registry


@dataclass
class PipelineResult:
    """The full output of one end-to-end pipeline run.

    Attributes:
        memo_markdown: The final memo, or ``None`` if the reviewer rejected
            the analysis.
        status: ``"complete"`` or ``"blocked_pending_revision"``.
        approved: Whether the human-in-the-loop reviewer approved the
            analysis.
        findings: The researcher agent's extracted findings.
        analysis: The analyst agent's structured analysis.
        citation_results: Per-finding citation-check results from the
            writer agent.
        stage_durations_seconds: Wall-clock ``time.perf_counter()`` duration
            of each pipeline stage, keyed by agent name. Used to render the
            latency-by-stage chart from real measurements.
        workflow_state: The final ``WorkflowState`` from the orchestration
            graph, for callers that want the raw execution trace.
    """

    memo_markdown: str | None
    status: str
    approved: bool
    findings: list[Mapping[str, Any]]
    analysis: Mapping[str, Any]
    citation_results: Mapping[str, Any]
    stage_durations_seconds: dict[str, float] = field(default_factory=dict)
    workflow_state: WorkflowState | None = None


def run_pipeline(
    document_paths: list[str],
    tool_registry: ToolRegistry | None = None,
    approval_fn: ApprovalFn = default_auto_approve,
) -> PipelineResult:
    """Runs the full researcher -> analyst -> reviewer -> writer pipeline once.

    Wires the DAG so that the reviewer stage is a ``PAUSE_FOR_HUMAN`` node:
    execution genuinely halts there, and the (simulated, in this offline
    demo) human decision is obtained and fed back in via
    ``OrchestrationGraph.resume()`` -- the same mechanic a production
    deployment would use to wait on a real reviewer.

    Args:
        document_paths: Paths to the plain-text source documents to run the
            pipeline against.
        tool_registry: Optional pre-built registry (mainly for tests).
            Defaults to a fresh registry from ``build_tool_registry()``.
        approval_fn: Decision function passed through to ``ReviewerAgent``.
            Defaults to auto-approving when the quality checklist passes,
            which is what keeps the offline demo runnable end to end.

    Returns:
        A ``PipelineResult`` describing the run's output and timings.
    """
    registry = tool_registry if tool_registry is not None else build_tool_registry()

    researcher = ResearcherAgent(registry)
    analyst = AnalystAgent(registry)
    reviewer = ReviewerAgent(registry, approval_fn=approval_fn)
    writer = WriterAgent(registry)

    stage_durations: dict[str, float] = {}

    def _timed(stage_name: str, fn):  # type: ignore[no-untyped-def]
        def _wrapped(context: Mapping[str, Any]) -> Any:
            start = time.perf_counter()
            result = fn(context)
            stage_durations[stage_name] = time.perf_counter() - start
            return result

        return _wrapped

    def _run_researcher(context: Mapping[str, Any]) -> Mapping[str, Any]:
        return researcher.act({"document_paths": context["document_paths"]}).content

    def _run_analyst(context: Mapping[str, Any]) -> Mapping[str, Any]:
        return analyst.act({"findings": context["researcher_agent"]["findings"]}).content

    def _run_writer(context: Mapping[str, Any]) -> Mapping[str, Any]:
        reviewer_decision = context["reviewer_agent"]
        researcher_output = context["researcher_agent"]
        return writer.act(
            {
                "approved": reviewer_decision["approved"],
                "analysis": reviewer_decision["analysis"],
                "findings": researcher_output["findings"],
                "chunks": researcher_output["chunks"],
            }
        ).content

    graph = OrchestrationGraph()
    graph.add_node(
        GraphNode(
            node_id="researcher_agent",
            node_type=NodeType.AGENT,
            dependencies=(),
            fn=_timed("researcher_agent", _run_researcher),
        )
    )
    graph.add_node(
        GraphNode(
            node_id="analyst_agent",
            node_type=NodeType.AGENT,
            dependencies=("researcher_agent",),
            fn=_timed("analyst_agent", _run_analyst),
        )
    )
    graph.add_node(
        GraphNode(
            node_id="reviewer_agent",
            node_type=NodeType.PAUSE_FOR_HUMAN,
            dependencies=("analyst_agent",),
        )
    )
    graph.add_node(
        GraphNode(
            node_id="writer_agent",
            node_type=NodeType.AGENT,
            dependencies=("reviewer_agent",),
            fn=_timed("writer_agent", _run_writer),
        )
    )

    state = graph.run(initial_context={"document_paths": document_paths})

    if state.status is not WorkflowStatus.PAUSED or state.pending_node_id != "reviewer_agent":
        raise RuntimeError(
            "Expected the pipeline to pause at 'reviewer_agent' for human "
            f"review, but got status={state.status!r} pending_node_id="
            f"{state.pending_node_id!r}"
        )

    review_start = time.perf_counter()
    reviewer_decision = reviewer.act({"analysis": state.context["analyst_agent"]}).content
    stage_durations["reviewer_agent"] = time.perf_counter() - review_start

    state = graph.resume(state, reviewer_decision)

    writer_output = state.context["writer_agent"]

    return PipelineResult(
        memo_markdown=writer_output["memo_markdown"],
        status=writer_output["status"],
        approved=reviewer_decision["approved"],
        findings=state.context["researcher_agent"]["findings"],
        analysis=state.context["analyst_agent"],
        citation_results=writer_output["citation_results"],
        stage_durations_seconds=stage_durations,
        workflow_state=state,
    )
