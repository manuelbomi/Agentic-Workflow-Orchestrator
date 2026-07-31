"""A small DAG-based workflow engine with a first-class human-in-the-loop pause.

Nodes are agent steps declared with explicit dependencies and executed in
topological order. One node type, ``PAUSE_FOR_HUMAN``, is special: reaching
it halts execution immediately and returns a resumable ``WorkflowState``
instead of a result. The caller is expected to obtain a real (or, in this
offline demo, simulated) human decision out-of-band and then call
``OrchestrationGraph.resume(state, decision)`` to inject that decision back
into the graph's context and continue execution from exactly where it left
off.

This pause/resume mechanic -- not just the topological-sort scheduler -- is
the design pattern this module exists to demonstrate: it lets an
otherwise-automated pipeline stop and wait for a qualified human reviewer
before a downstream step (in this project, publishing a client memo) can
run, without losing any of the work already completed upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Mapping, Sequence


class NodeType(Enum):
    """The kind of work a ``GraphNode`` represents."""

    AGENT = auto()
    """A normal computation step: ``fn(context) -> result``."""

    PAUSE_FOR_HUMAN = auto()
    """Halts the graph; a result must be supplied later via ``resume()``."""


class GraphCycleError(RuntimeError):
    """Raised when the graph's declared dependencies contain a cycle."""


class GraphDefinitionError(ValueError):
    """Raised when a node references a dependency that was never registered."""


class WorkflowStatus(Enum):
    """Overall status of a ``WorkflowState``."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()


NodeFn = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class GraphNode:
    """A single step in the orchestration DAG.

    Attributes:
        node_id: Unique identifier for this node, also the key its result
            is stored under in the workflow context once it completes.
        node_type: ``NodeType.AGENT`` or ``NodeType.PAUSE_FOR_HUMAN``.
        dependencies: Node ids that must complete before this node runs.
            Purely used for scheduling order; ``fn`` receives the *entire*
            accumulated context (not just direct dependencies' results), so
            an agent function can read any earlier node's output by its
            node_id key.
        fn: The function to execute for ``AGENT`` nodes, called as
            ``fn(context) -> result``. Ignored (may be ``None``) for
            ``PAUSE_FOR_HUMAN`` nodes, whose result instead comes from a
            later call to ``OrchestrationGraph.resume()``.
    """

    node_id: str
    node_type: NodeType
    dependencies: tuple[str, ...] = ()
    fn: NodeFn | None = None

    def __post_init__(self) -> None:
        if self.node_type is NodeType.AGENT and self.fn is None:
            raise GraphDefinitionError(
                f"AGENT node '{self.node_id}' must be given a callable `fn`"
            )


@dataclass
class WorkflowState:
    """Resumable execution state for one run of an ``OrchestrationGraph``.

    Attributes:
        context: Accumulated key/value store. Seeded with the initial input
            context passed to ``run()``; gains one new entry per completed
            node, keyed by that node's ``node_id``.
        completed: Set of node ids that have finished executing.
        status: Current ``WorkflowStatus``.
        pending_node_id: When ``status is PAUSED``, the id of the
            ``PAUSE_FOR_HUMAN`` node awaiting a decision via ``resume()``.
    """

    context: dict[str, Any] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    status: WorkflowStatus = WorkflowStatus.PENDING
    pending_node_id: str | None = None


class OrchestrationGraph:
    """A minimal DAG scheduler supporting pause/resume for human review steps."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}

    def add_node(self, node: GraphNode) -> None:
        """Registers ``node`` with the graph."""
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> GraphNode:
        """Returns the registered node with id ``node_id``."""
        return self._nodes[node_id]

    def topological_order(self) -> list[str]:
        """Computes a deterministic topological ordering of all registered nodes.

        Uses Kahn's algorithm; ties among simultaneously-ready nodes are
        broken alphabetically by ``node_id`` so the resulting order is
        deterministic and stable across runs.

        Raises:
            GraphDefinitionError: If a node declares a dependency that was
                never registered.
            GraphCycleError: If the dependency graph contains a cycle.
        """
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise GraphDefinitionError(
                        f"Node '{node.node_id}' depends on unregistered node '{dep}'"
                    )

        in_degree: dict[str, int] = {node_id: 0 for node_id in self._nodes}
        dependents: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        for node in self._nodes.values():
            in_degree[node.node_id] = len(node.dependencies)
            for dep in node.dependencies:
                dependents[dep].append(node.node_id)

        ready = sorted(node_id for node_id, degree in in_degree.items() if degree == 0)
        order: list[str] = []

        while ready:
            ready.sort()
            current = ready.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        if len(order) != len(self._nodes):
            remaining = set(self._nodes) - set(order)
            raise GraphCycleError(
                f"Cycle detected among node(s): {sorted(remaining)}"
            )
        return order

    def run(
        self,
        initial_context: Mapping[str, Any] | None = None,
        state: WorkflowState | None = None,
    ) -> WorkflowState:
        """Executes registered nodes in topological order until done or paused.

        Args:
            initial_context: Seed values available to every node's ``fn``
                (e.g. ``{"document_paths": [...]}``). Ignored if ``state``
                is provided (an in-progress state already carries its own
                context).
            state: An existing ``WorkflowState`` to continue running (used
                internally by ``resume()``). Pass ``None`` to start a fresh
                run.

        Returns:
            The resulting ``WorkflowState``. If execution reaches a
            ``PAUSE_FOR_HUMAN`` node, ``status`` will be ``PAUSED`` and
            ``pending_node_id`` will identify that node; otherwise
            ``status`` will be ``COMPLETED``.
        """
        order = self.topological_order()
        if state is None:
            state = WorkflowState(context=dict(initial_context or {}))
        state.status = WorkflowStatus.RUNNING

        for node_id in order:
            if node_id in state.completed:
                continue
            node = self._nodes[node_id]

            if node.node_type is NodeType.PAUSE_FOR_HUMAN:
                state.status = WorkflowStatus.PAUSED
                state.pending_node_id = node_id
                return state

            assert node.fn is not None  # guaranteed by GraphNode.__post_init__
            result = node.fn(state.context)
            state.context[node_id] = result
            state.completed.add(node_id)

        state.status = WorkflowStatus.COMPLETED
        state.pending_node_id = None
        return state

    def resume(self, state: WorkflowState, decision: Any) -> WorkflowState:
        """Injects a human decision into a paused workflow and continues it.

        Args:
            state: A ``WorkflowState`` previously returned by ``run()`` with
                ``status == WorkflowStatus.PAUSED``.
            decision: The value to store as the paused node's result (e.g.
                the reviewer agent's decision dict). Downstream nodes read
                it from ``context[pending_node_id]`` exactly like any other
                completed node's output.

        Returns:
            The updated ``WorkflowState`` after resuming execution, which
            may be ``COMPLETED`` or, if another ``PAUSE_FOR_HUMAN`` node is
            reached, ``PAUSED`` again.

        Raises:
            RuntimeError: If ``state`` is not currently paused.
        """
        if state.status is not WorkflowStatus.PAUSED or state.pending_node_id is None:
            raise RuntimeError("Cannot resume a workflow that is not currently paused.")

        node_id = state.pending_node_id
        state.context[node_id] = decision
        state.completed.add(node_id)
        state.pending_node_id = None
        return self.run(state=state)

    def node_ids(self) -> Sequence[str]:
        """Returns all registered node ids in insertion order."""
        return tuple(self._nodes.keys())
