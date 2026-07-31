"""Reviewer agent: the mandatory human-in-the-loop review gate.

Unlike the researcher and analyst agents, this agent is deliberately *not*
an LLM-backed reasoning step. Its entire purpose is to stand between the
analyst's draft and the writer's final memo and force an explicit
approve/reject decision before anything can reach a client-facing
deliverable.

In the automated demo path, the decision is produced by a pluggable
``approval_fn``. The default implementation runs a quality checklist against
the analyst's output and auto-approves when every checklist item passes --
this keeps ``examples/run_pipeline.py`` runnable end-to-end with no
interactive input. In a production deployment, ``approval_fn`` would instead
be swapped for something that actually pauses and waits on a qualified human
reviewer (see ``orchestration_graph.PAUSE_FOR_HUMAN`` and its ``resume()``
mechanic, which is what this agent is wired through in the DAG).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from orchestrator.agent_base import Agent, AgentOutput

REVIEWER_SYSTEM_PROMPT = (
    "You are the human-in-the-loop review gate for this pipeline. Your job "
    "is not to generate new analysis, but to check the analyst's output "
    "against a quality checklist before it is allowed to reach the writer "
    "agent and, ultimately, a client."
)

# The type of a pluggable approval decision function: takes the analyst's
# analysis content plus the computed checklist, returns (approved, notes).
ApprovalFn = Callable[[Mapping[str, Any], Mapping[str, bool]], tuple[bool, str]]


def build_checklist(analysis: Mapping[str, Any]) -> dict[str, bool]:
    """Runs a fixed quality checklist against the analyst's draft output.

    Args:
        analysis: The ``content`` dict produced by ``AnalystAgent.act``.

    Returns:
        A mapping of checklist item name -> pass/fail boolean.
    """
    themes = analysis.get("themes", {})
    risks = analysis.get("risks", [])
    opportunities = analysis.get("opportunities", [])
    recommendation = analysis.get("recommendation", {})

    return {
        "has_at_least_one_theme": len(themes) >= 1,
        "has_risk_or_opportunity_signal": (len(risks) + len(opportunities)) >= 1,
        "has_scored_recommendation": "score" in recommendation and "label" in recommendation,
        "recommendation_has_rationale": bool(recommendation.get("rationale")),
    }


def default_auto_approve(
    analysis: Mapping[str, Any], checklist: Mapping[str, bool]
) -> tuple[bool, str]:
    """Default approval function used by the offline demo pipeline.

    Approves when every checklist item passes. This simulates the decision
    a human reviewer would make when a draft is clean, so the end-to-end
    example can run unattended. Swap this for a function that blocks on
    real human input in a production deployment.
    """
    if all(checklist.values()):
        return True, "All checklist items passed; auto-approved for demo purposes."
    failed = [name for name, passed in checklist.items() if not passed]
    return False, f"Checklist item(s) failed: {failed}"


class ReviewerAgent(Agent):
    """Implements the human-in-the-loop approval gate between analysis and writing."""

    def __init__(
        self,
        tool_registry,  # type: ignore[no-untyped-def]
        approval_fn: ApprovalFn = default_auto_approve,
    ) -> None:
        super().__init__(
            name="reviewer_agent",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=[],
            tool_registry=tool_registry,
            llm_client=None,
        )
        self.approval_fn = approval_fn

    def act(self, input_context: Mapping[str, Any]) -> AgentOutput:
        """Runs the checklist and records an approve/reject decision.

        Args:
            input_context: Must contain ``analysis``, the content dict
                produced by ``AnalystAgent.act``.

        Returns:
            An ``AgentOutput`` whose ``content`` has keys ``approved``
            (bool), ``checklist`` (dict[str, bool]), ``notes`` (str), and
            ``analysis`` (the original analysis, passed through so the
            writer agent receives it unchanged).
        """
        analysis = input_context["analysis"]
        checklist = build_checklist(analysis)
        approved, notes = self.approval_fn(analysis, checklist)

        decision = {
            "approved": approved,
            "checklist": checklist,
            "notes": notes,
            "analysis": analysis,
        }
        self.remember(decision)

        return AgentOutput(
            agent_name=self.name,
            content=decision,
            metadata={
                "approved": approved,
                "checklist_pass_count": sum(checklist.values()),
                "checklist_total": len(checklist),
            },
        )
