"""Document parser tool: splits source documents into cited paragraph chunks.

This tool is the researcher agent's window onto the outside world. Rather
than letting an agent read arbitrary files directly, it goes through this
narrow, registry-mediated tool -- exactly the kind of narrow, auditable
capability boundary a real deployment would want between an LLM agent and
a firm's document store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from orchestrator.tool_registry import ToolSpec


@dataclass(frozen=True)
class DocumentChunk:
    """A single paragraph-level chunk of a source document.

    Attributes:
        source_id: Stable identifier of the document this chunk came from
            (the file stem, e.g. ``"company_notes"``).
        chunk_id: Stable identifier for this specific chunk, e.g.
            ``"company_notes#0"``. This is the identifier that citation
            markers in the generated memo point back to.
        text: The paragraph text itself, whitespace-normalized.
        paragraph_index: Zero-based position of this paragraph within its
            source document.
    """

    source_id: str
    chunk_id: str
    text: str
    paragraph_index: int


def parse_document(path: str | Path) -> list[DocumentChunk]:
    """Parses a single plain-text document into paragraph-level chunks.

    Paragraphs are separated by one or more blank lines. Each paragraph's
    internal whitespace (including embedded newlines) is collapsed to
    single spaces so downstream lexical-overlap checks behave predictably.

    Args:
        path: Path to a ``.txt`` source document.

    Returns:
        A list of ``DocumentChunk``, one per non-empty paragraph, in
        document order.
    """
    file_path = Path(path)
    source_id = file_path.stem
    raw_text = file_path.read_text(encoding="utf-8")

    raw_paragraphs = re.split(r"\n\s*\n", raw_text.strip())
    chunks: list[DocumentChunk] = []
    for index, paragraph in enumerate(raw_paragraphs):
        normalized = " ".join(paragraph.split())
        if not normalized:
            continue
        chunks.append(
            DocumentChunk(
                source_id=source_id,
                chunk_id=f"{source_id}#{index}",
                text=normalized,
                paragraph_index=index,
            )
        )
    return chunks


def parse_documents(paths: list[str]) -> list[Mapping[str, object]]:
    """Parses multiple documents and returns plain-dict chunks.

    A list-of-dicts return type (rather than ``DocumentChunk`` instances) is
    used at the tool boundary so results stay JSON-serializable, matching
    how a real MCP-style tool call would return data across a process/network
    boundary.

    Args:
        paths: Paths to ``.txt`` source documents.

    Returns:
        A flat list of chunk dicts with keys ``source_id``, ``chunk_id``,
        ``text``, ``paragraph_index``, in document then paragraph order.
    """
    all_chunks: list[Mapping[str, object]] = []
    for path in paths:
        for chunk in parse_document(path):
            all_chunks.append(
                {
                    "source_id": chunk.source_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "paragraph_index": chunk.paragraph_index,
                }
            )
    return all_chunks


def build_spec() -> ToolSpec:
    """Builds the ``ToolSpec`` for registering this tool with a ``ToolRegistry``."""
    return ToolSpec(
        name="document_parser",
        description=(
            "Parses one or more plain-text source documents into paragraph-level "
            "chunks, each with a stable source-attributed chunk_id."
        ),
        input_schema={
            "required": ["paths"],
            "properties": {"paths": {"type": "array"}},
        },
        handler=parse_documents,
    )
