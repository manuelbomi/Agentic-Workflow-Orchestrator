"""Agentic Workflow Orchestrator.

A small, self-contained multi-agent orchestration framework for advisory and
professional-services style knowledge work: turning a pile of source
documents into a structured, cited research/analysis memo.

The package demonstrates three reusable design patterns that generalize well
beyond this specific demo:

1. An MCP-style tool registry with per-agent permission scopes
   (``orchestrator.tool_registry``).
2. A DAG-based orchestration engine with a first-class human-in-the-loop
   pause/resume mechanic (``orchestrator.orchestration_graph``).
3. A rubric-based evaluation harness for scoring generative pipeline output
   (``orchestrator.eval.eval_harness``).

Everything in this package runs fully offline. Agents accept an optional
``llm_client`` callable as the seam where a production deployment would plug
in a real large language model (see ``orchestrator.agent_base.Agent``).
"""

__version__ = "1.0.0"
