"""Analyst agent: synthesizes researcher findings into structured analysis.

Production seam: a real deployment would send the findings list to an LLM
and ask for thematic synthesis, risk/opportunity classification, and a
scored recommendation, using ``llm_client``. The offline demo instead uses
``_synthesize_stub``, a deterministic keyword/frequency-based heuristic
standing in for that LLM call: findings are grouped into themes by source
document, classified as risk- or opportunity-flavored by simple keyword
lists, and rolled up into a single recommendation score computed through
the sandboxed ``calculator`` tool (never a raw arithmetic ``eval``).
"""

from __future__ import annotations

from typing import Any, Mapping

from orchestrator.agent_base import Agent, AgentOutput

ANALYST_SYSTEM_PROMPT = (
    "You are a senior analyst at an advisory firm. Given a set of "
    "source-attributed findings, identify the major themes, separate risk "
    "signals from opportunity signals, and produce a single, defensible, "
    "numerically scored recommendation. Ground every claim in the supplied "
    "findings; do not introduce facts that are not present in them."
)

_THEME_LABELS: dict[str, str] = {
    "company_notes": "Company Performance",
    "market_notes": "Market Context",
    "risk_notes": "Risk Factors",
}

_RISK_KEYWORDS = frozenset(
    {
        "risk", "risks", "churn", "concentration", "dependency", "decline",
        "declined", "pressure", "cost", "costs", "compliance", "loss",
        "competitive", "competition",
    }
)

_OPPORTUNITY_KEYWORDS = frozenset(
    {
        "growth", "grow", "growing", "expansion", "expand", "retention",
        "increase", "increased", "opportunity", "opportunities", "raised",
        "target", "up",
    }
)


def _theme_label(source_id: str) -> str:
    return _THEME_LABELS.get(source_id, source_id.replace("_", " ").title())


def _classify_finding(text: str) -> str:
    """Classifies a finding's text as 'risk', 'opportunity', or 'neutral'."""
    lowered = text.lower()
    is_risk = any(keyword in lowered for keyword in _RISK_KEYWORDS)
    is_opportunity = any(keyword in lowered for keyword in _OPPORTUNITY_KEYWORDS)
    if is_risk and not is_opportunity:
        return "risk"
    if is_opportunity and not is_risk:
        return "opportunity"
    if is_risk and is_opportunity:
        # Mixed-signal findings (e.g. "growth alongside rising competitive
        # pressure") are conservatively counted as risk for scoring purposes.
        return "risk"
    return "neutral"


def _recommendation_label(score: float) -> str:
    if score >= 20:
        return "Favorable"
    if score <= -20:
        return "Unfavorable"
    return "Mixed / Monitor"


def _synthesize_stub(
    findings: list[Mapping[str, Any]], call_tool
) -> dict[str, Any]:
    """Deterministic offline stand-in for an LLM synthesis call.

    Args:
        findings: Findings produced by the researcher agent.
        call_tool: Bound ``Agent.call_tool`` used to compute the
            recommendation score through the permissioned ``calculator``
            tool rather than inline Python arithmetic, so the score's
            derivation goes through the same auditable tool-call path a
            production agent's arithmetic would.

    Returns:
        A dict with keys ``themes``, ``risks``, ``opportunities``, and
        ``recommendation``.
    """
    themes: dict[str, list[Mapping[str, Any]]] = {}
    risks: list[Mapping[str, Any]] = []
    opportunities: list[Mapping[str, Any]] = []

    for finding in findings:
        theme_key = _theme_label(str(finding["source_id"]))
        themes.setdefault(theme_key, []).append(finding)

        classification = _classify_finding(str(finding["text"]))
        if classification == "risk":
            risks.append(finding)
        elif classification == "opportunity":
            opportunities.append(finding)

    risk_count = len(risks)
    opportunity_count = len(opportunities)
    total_signals = risk_count + opportunity_count

    if total_signals == 0:
        score = 0.0
    else:
        expression = (
            f"({opportunity_count} - {risk_count}) / "
            f"({opportunity_count} + {risk_count}) * 100"
        )
        score = round(float(call_tool("calculator", expression=expression)), 2)

    recommendation = {
        "score": score,
        "label": _recommendation_label(score),
        "risk_count": risk_count,
        "opportunity_count": opportunity_count,
        "rationale": (
            f"Identified {opportunity_count} opportunity signal(s) and "
            f"{risk_count} risk signal(s) across {len(themes)} theme(s); "
            f"net signal score is {score:+.2f} on a -100..+100 scale."
        ),
    }

    return {
        "themes": {label: items for label, items in themes.items()},
        "risks": risks,
        "opportunities": opportunities,
        "recommendation": recommendation,
    }


class AnalystAgent(Agent):
    """Synthesizes findings into themes, risk/opportunity signals, and a score."""

    def __init__(self, tool_registry, llm_client=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(
            name="analyst_agent",
            system_prompt=ANALYST_SYSTEM_PROMPT,
            allowed_tools=["calculator"],
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

    def act(self, input_context: Mapping[str, Any]) -> AgentOutput:
        """Synthesizes structured analysis from upstream findings.

        Args:
            input_context: Must contain ``findings``, the list produced by
                ``ResearcherAgent.act``.

        Returns:
            An ``AgentOutput`` whose ``content`` has keys ``themes``,
            ``risks``, ``opportunities``, and ``recommendation``.
        """
        findings = list(input_context["findings"])

        if self.uses_real_llm():
            prompt = "\n".join(f"- {f['text']} (source: {f['chunk_id']})" for f in findings)
            _ = self.llm_client(self.system_prompt, prompt)  # pragma: no cover
            raise NotImplementedError(
                "Real LLM synthesis is a documented extension point, not part "
                "of the offline demo path. Construct AnalystAgent without an "
                "llm_client to use the deterministic stub."
            )

        analysis = _synthesize_stub(findings, self.call_tool)
        self.remember(analysis)

        return AgentOutput(
            agent_name=self.name,
            content=analysis,
            metadata={
                "num_themes": len(analysis["themes"]),
                "num_risks": len(analysis["risks"]),
                "num_opportunities": len(analysis["opportunities"]),
                "recommendation_score": analysis["recommendation"]["score"],
                "used_llm": self.uses_real_llm(),
            },
        )
