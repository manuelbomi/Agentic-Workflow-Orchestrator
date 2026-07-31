"""Researcher agent: ingests source documents and extracts cited findings.

Production seam: if this agent is constructed with a real ``llm_client``,
``act()`` sends each document chunk to the LLM and asks it to extract key
facts and quotes. In the shipped offline demo (``llm_client is None``), the
agent instead runs ``_extract_findings_stub``, a deterministic heuristic:
sentences within a chunk that contain a digit (a number, percentage, or
dollar figure) are treated as "findings" worth surfacing to the analyst,
since numeric claims are exactly the kind of thing an advisory memo needs
to state precisely and cite correctly.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from orchestrator.agent_base import Agent, AgentOutput

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HAS_DIGIT_RE = re.compile(r"\d")
# Heuristic proper-noun detector: two adjacent capitalized words (e.g. "Dana
# Whitfield", "Northbridge Analytics"). Used, alongside the digit check
# below, to decide which sentences are worth surfacing as findings without
# any real named-entity-recognition model -- consistent with this project's
# fully offline, dependency-light approach.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")

RESEARCHER_SYSTEM_PROMPT = (
    "You are a meticulous research analyst at an advisory firm. Read the "
    "provided source documents and extract discrete, verifiable facts and "
    "notable quotes. Every fact you record must be traceable back to the "
    "exact source chunk it came from. Do not invent facts that are not "
    "present in the source text."
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _extract_findings_stub(chunks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic offline stand-in for an LLM fact-extraction call.

    Surfaces every sentence containing at least one digit (a numeric fact)
    or a proper-noun-like bigram (a named entity, e.g. a person or company
    name) as a finding, and additionally surfaces the first sentence of
    every chunk as a contextual finding (so purely qualitative chunks still
    contribute at least one finding). Each finding records the exact
    ``chunk_id`` and ``source_id`` it was drawn from, which is what allows
    the writer agent to cite it correctly downstream.
    """
    findings: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for chunk in chunks:
        sentences = _split_sentences(str(chunk["text"]))
        if not sentences:
            continue

        candidate_sentences: list[str] = [sentences[0]]
        candidate_sentences.extend(
            s for s in sentences
            if _HAS_DIGIT_RE.search(s) or _PROPER_NOUN_RE.search(s)
        )

        for sentence in candidate_sentences:
            if sentence in seen_texts:
                continue
            seen_texts.add(sentence)
            findings.append(
                {
                    "finding_id": f"finding_{len(findings)}",
                    "source_id": chunk["source_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": sentence,
                    "is_numeric": bool(_HAS_DIGIT_RE.search(sentence)),
                }
            )
    return findings


class ResearcherAgent(Agent):
    """Extracts structured, source-attributed findings from source documents."""

    def __init__(self, tool_registry, llm_client=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(
            name="researcher_agent",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            allowed_tools=["document_parser"],
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

    def act(self, input_context: Mapping[str, Any]) -> AgentOutput:
        """Parses source documents and extracts a structured findings list.

        Args:
            input_context: Must contain ``document_paths``, a list of
                filesystem paths to plain-text source documents.

        Returns:
            An ``AgentOutput`` whose ``content`` has keys ``chunks`` (the
            full parsed corpus, needed downstream for citation checking)
            and ``findings`` (the extracted, source-attributed facts).
        """
        document_paths = list(input_context["document_paths"])
        chunks = self.call_tool("document_parser", paths=document_paths)

        if self.uses_real_llm():
            # Production path: send each chunk to the real LLM and parse its
            # response into findings. Left unimplemented in this offline
            # demo; see README "Plugging in a real LLM" for the pattern.
            prompt = "\n\n".join(str(c["text"]) for c in chunks)
            _ = self.llm_client(self.system_prompt, prompt)  # pragma: no cover
            raise NotImplementedError(
                "Real LLM parsing is a documented extension point, not part of "
                "the offline demo path. Construct ResearcherAgent without an "
                "llm_client to use the deterministic stub."
            )

        findings = _extract_findings_stub(chunks)
        self.remember(findings)

        return AgentOutput(
            agent_name=self.name,
            content={"chunks": chunks, "findings": findings},
            metadata={
                "num_documents": len(document_paths),
                "num_chunks": len(chunks),
                "num_findings": len(findings),
                "used_llm": self.uses_real_llm(),
            },
        )
