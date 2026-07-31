"""Tests for the DAG orchestration engine, including pause/resume."""

from __future__ import annotations

import pytest

from orchestrator.orchestration_graph import (
    GraphCycleError,
    GraphDefinitionError,
    GraphNode,
    NodeType,
    OrchestrationGraph,
    WorkflowStatus,
)


def test_topological_order_respects_dependencies() -> None:
    """A diamond dependency graph (a -> b, a -> c, b+c -> d) must place a
    before b and c, and both b and c before d."""
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("a", NodeType.AGENT, (), fn=lambda ctx: "a"))
    graph.add_node(GraphNode("b", NodeType.AGENT, ("a",), fn=lambda ctx: "b"))
    graph.add_node(GraphNode("c", NodeType.AGENT, ("a",), fn=lambda ctx: "c"))
    graph.add_node(GraphNode("d", NodeType.AGENT, ("b", "c"), fn=lambda ctx: "d"))

    order = graph.topological_order()

    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")
    assert set(order) == {"a", "b", "c", "d"}


def test_topological_order_is_deterministic() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("z", NodeType.AGENT, (), fn=lambda ctx: None))
    graph.add_node(GraphNode("y", NodeType.AGENT, (), fn=lambda ctx: None))
    graph.add_node(GraphNode("x", NodeType.AGENT, (), fn=lambda ctx: None))

    # All three are independent roots; ties must break alphabetically.
    assert graph.topological_order() == ["x", "y", "z"]


def test_cycle_is_detected() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("a", NodeType.AGENT, ("b",), fn=lambda ctx: None))
    graph.add_node(GraphNode("b", NodeType.AGENT, ("a",), fn=lambda ctx: None))

    with pytest.raises(GraphCycleError):
        graph.topological_order()


def test_unregistered_dependency_raises_definition_error() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("a", NodeType.AGENT, ("missing",), fn=lambda ctx: None))

    with pytest.raises(GraphDefinitionError):
        graph.topological_order()


def test_agent_node_requires_fn() -> None:
    with pytest.raises(GraphDefinitionError):
        GraphNode("a", NodeType.AGENT, ())


def test_run_executes_agent_nodes_in_order() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("double", NodeType.AGENT, (), fn=lambda ctx: ctx["n"] * 2))
    graph.add_node(
        GraphNode(
            "increment", NodeType.AGENT, ("double",), fn=lambda ctx: ctx["double"] + 1
        )
    )

    state = graph.run(initial_context={"n": 10})

    assert state.status is WorkflowStatus.COMPLETED
    assert state.context["double"] == 20
    assert state.context["increment"] == 21


def test_pause_for_human_halts_execution() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("step_one", NodeType.AGENT, (), fn=lambda ctx: "done"))
    graph.add_node(GraphNode("review", NodeType.PAUSE_FOR_HUMAN, ("step_one",)))
    graph.add_node(
        GraphNode(
            "step_two",
            NodeType.AGENT,
            ("review",),
            fn=lambda ctx: f"following up on {ctx['review']}",
        )
    )

    state = graph.run()

    assert state.status is WorkflowStatus.PAUSED
    assert state.pending_node_id == "review"
    assert "step_two" not in state.context
    assert state.context["step_one"] == "done"


def test_resume_continues_graph_with_injected_decision() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("step_one", NodeType.AGENT, (), fn=lambda ctx: "done"))
    graph.add_node(GraphNode("review", NodeType.PAUSE_FOR_HUMAN, ("step_one",)))
    graph.add_node(
        GraphNode(
            "step_two",
            NodeType.AGENT,
            ("review",),
            fn=lambda ctx: f"following up on {ctx['review']}",
        )
    )

    paused_state = graph.run()
    resumed_state = graph.resume(paused_state, decision="approved")

    assert resumed_state.status is WorkflowStatus.COMPLETED
    assert resumed_state.context["review"] == "approved"
    assert resumed_state.context["step_two"] == "following up on approved"


def test_resume_without_pause_raises() -> None:
    graph = OrchestrationGraph()
    graph.add_node(GraphNode("a", NodeType.AGENT, (), fn=lambda ctx: "a"))
    state = graph.run()

    with pytest.raises(RuntimeError):
        graph.resume(state, decision="anything")
