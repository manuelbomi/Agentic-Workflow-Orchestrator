"""Citation checker tool: validates inline citations against source chunks.

The writer agent is required to route every claim it writes through this
tool before the claim is allowed into the final memo. A citation is
considered valid when:

1. The cited ``chunk_id`` actually exists in the parsed source corpus, and
2. The claim text has "reasonable" lexical overlap with that chunk's text,
   measured as a simple token-overlap ratio.

This is a heuristic, not a semantic entailment check -- it is intentionally
simple and dependency-free so the whole project can run offline -- but it
is enough to catch the two most common and damaging citation failures in
practice: citing a source that does not exist, and citing a real source
that does not actually support the claim being made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from orchestrator.tool_registry import ToolSpec

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is",
        "are", "was", "were", "be", "been", "with", "as", "by", "at", "from",
        "that", "this", "it", "its", "has", "have", "had", "will", "would",
        "than", "into", "over", "under", "per", "about",
    }
)


def _tokenize(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {token for token in tokens if token not in _STOPWORDS}


@dataclass(frozen=True)
class CitationCheckResult:
    """Outcome of checking a single claim/citation pair.

    Attributes:
        chunk_id: The cited chunk id.
        claim_text: The claim text being checked.
        chunk_found: Whether ``chunk_id`` exists in the provided source chunks.
        overlap_ratio: Fraction of the claim's meaningful tokens that also
            appear in the cited chunk's text (0.0 if the chunk was not found).
        is_valid: ``True`` when the chunk exists and ``overlap_ratio`` meets
            or exceeds the overlap threshold used for this check.
    """

    chunk_id: str
    claim_text: str
    chunk_found: bool
    overlap_ratio: float
    is_valid: bool


def check_citation(
    claim_text: str,
    chunk_id: str,
    source_chunks: Sequence[Mapping[str, object]],
    overlap_threshold: float = 0.3,
) -> Mapping[str, object]:
    """Checks whether ``claim_text`` is adequately supported by ``chunk_id``.

    Args:
        claim_text: The sentence/claim from the generated memo.
        chunk_id: The chunk id the memo cites for this claim, e.g.
            ``"company_notes#2"``.
        source_chunks: The full parsed source corpus (list of chunk dicts
            with at least ``chunk_id`` and ``text`` keys), typically the
            output of ``document_parser_tool.parse_documents``.
        overlap_threshold: Minimum fraction of the claim's non-stopword
            tokens that must also appear in the cited chunk for the
            citation to be considered valid. Defaults to ``0.3``.

    Returns:
        A dict form of ``CitationCheckResult`` (JSON-serializable, matching
        the other tool boundaries in this project).
    """
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in source_chunks}
    chunk = chunk_lookup.get(chunk_id)

    if chunk is None:
        result = CitationCheckResult(
            chunk_id=chunk_id,
            claim_text=claim_text,
            chunk_found=False,
            overlap_ratio=0.0,
            is_valid=False,
        )
        return result.__dict__

    claim_tokens = _tokenize(claim_text)
    chunk_tokens = _tokenize(str(chunk["text"]))

    if not claim_tokens:
        overlap_ratio = 0.0
    else:
        overlap_ratio = len(claim_tokens & chunk_tokens) / len(claim_tokens)

    result = CitationCheckResult(
        chunk_id=chunk_id,
        claim_text=claim_text,
        chunk_found=True,
        overlap_ratio=round(overlap_ratio, 4),
        is_valid=overlap_ratio >= overlap_threshold,
    )
    return result.__dict__


def build_spec() -> ToolSpec:
    """Builds the ``ToolSpec`` for registering this tool with a ``ToolRegistry``."""
    return ToolSpec(
        name="citation_checker",
        description=(
            "Validates that a citation marker (chunk_id) exists and that the "
            "claim text has adequate lexical overlap with the cited source chunk."
        ),
        input_schema={
            "required": ["claim_text", "chunk_id", "source_chunks"],
            "properties": {
                "claim_text": {"type": "string"},
                "chunk_id": {"type": "string"},
                "source_chunks": {"type": "array"},
            },
        },
        handler=check_citation,
    )
