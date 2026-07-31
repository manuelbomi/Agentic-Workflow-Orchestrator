"""Abstract base class shared by every agent in the pipeline.

Production seam
----------------
Each concrete agent's ``act()`` method is where, in a production deployment,
you would call a real large language model. To keep that seam explicit and
swappable, every ``Agent`` accepts an optional ``llm_client`` constructor
argument:

    llm_client: Callable[[str, str], str] | None

``llm_client`` is called as ``llm_client(system_prompt, user_prompt) ->
completion_text``. That is a deliberately provider-agnostic shape: an
Anthropic Messages API call, an OpenAI Chat Completions call, or any other
LLM HTTP client can be wrapped in a one-line lambda/function that matches
this signature (see ``README.md`` -> "Plugging in a real LLM" for a worked
example against the Anthropic SDK).

If no ``llm_client`` is supplied -- which is the path used everywhere in
this repository's tests, examples, and CI -- each agent falls back to a
deterministic, fully local stub implementation of its reasoning step. The
stub functions are simple keyword/frequency/template based heuristics; they
are clearly labeled as stand-ins throughout the code and are what makes it
possible to run this entire project with no network access and no API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from orchestrator.tool_registry import ToolRegistry

# Type alias for the LLM plug-in seam described above.
LLMClient = Callable[[str, str], str]


class AgentOutput(BaseModel):
    """The structured result of a single agent's ``act()`` call.

    Modeled with pydantic rather than a plain dataclass because this is the
    boundary object every agent hands to the next stage of the pipeline --
    exactly the kind of structured, validated payload a real LLM-backed
    agent would need to parse and validate out of a model completion in
    production. Using pydantic here even though the demo path is fully
    deterministic keeps that validation boundary real rather than notional.

    Attributes:
        agent_name: Name of the agent that produced this output.
        content: The primary payload. Concrete agents document the expected
            shape (e.g. researcher_agent returns a list of finding dicts
            under ``content["findings"]``).
        metadata: Free-form auxiliary information (timings, tool calls made,
            counts, etc.) useful for debugging, evaluation, and charts.
    """

    agent_name: str
    content: Mapping[str, Any]
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    """Abstract base class for every agent in the orchestration pipeline.

    Attributes:
        name: Unique agent identifier, used for tool-permission lookups and
            logging (e.g. ``"researcher_agent"``).
        system_prompt: The role / system-prompt string describing this
            agent's persona and responsibilities. In the offline demo this
            is stored for documentation and future LLM use, but the
            deterministic stub logic does not need to "read" it to run; a
            real LLM-backed implementation would send it as the system
            message on every call.
        allowed_tools: The explicit list of tool names this agent is
            permitted to call through the shared ``ToolRegistry``. This is
            the agent-side half of the permission-scope pattern implemented
            in ``orchestrator.tool_registry``.
        tool_registry: The shared registry instance used to make tool
            calls. Must already have granted ``allowed_tools`` to ``name``
            (the orchestration graph / example wiring does this).
        memory: An append-only list of prior messages/artifacts this agent
            has produced or received, useful for multi-turn or reflective
            agent designs even though the current pipeline is single-pass.
        llm_client: Optional production seam described in the module
            docstring. Defaults to ``None``, which routes ``act()`` through
            this agent's deterministic offline stub.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        allowed_tools: Sequence[str],
        tool_registry: ToolRegistry,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.allowed_tools: tuple[str, ...] = tuple(allowed_tools)
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.memory: list[Any] = []

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Convenience wrapper around ``tool_registry.call`` bound to this agent.

        Delegates permission enforcement entirely to the registry: if
        ``tool_name`` is not in this agent's granted allow-list, the
        registry raises ``ToolPermissionError``.
        """
        return self.tool_registry.call(self.name, tool_name, **kwargs)

    def remember(self, item: Any) -> None:
        """Appends ``item`` to this agent's memory log."""
        self.memory.append(item)

    def uses_real_llm(self) -> bool:
        """Returns ``True`` if this agent was configured with a real LLM client."""
        return self.llm_client is not None

    @abstractmethod
    def act(self, input_context: Mapping[str, Any]) -> AgentOutput:
        """Executes this agent's reasoning step and returns a structured output.

        Args:
            input_context: Upstream data this agent needs (e.g. source
                documents for the researcher, findings for the analyst).

        Returns:
            An ``AgentOutput`` capturing this agent's contribution to the
            pipeline.
        """
        raise NotImplementedError
