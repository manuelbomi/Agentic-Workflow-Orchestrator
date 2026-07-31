#!/usr/bin/env python
"""End-to-end example: documents -> researcher -> analyst -> reviewer -> writer.

Runs the full offline demo pipeline against the bundled fictional sample
documents in ``data/sample_documents/``, using the auto-approve human review
path so the script completes unattended. Prints the resulting memo to
stdout and writes it to ``examples/sample_output.md``.

Run from the repository root:

    python examples/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src-layout `orchestrator` package importable without requiring an
# editable install -- keeps the project's setup to `pip install -r
# requirements.txt` as documented in the README.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchestrator.pipeline import PipelineResult, run_pipeline  # noqa: E402


def main() -> None:
    """Runs the pipeline once and writes/prints the resulting memo."""
    data_dir = _REPO_ROOT / "data" / "sample_documents"
    document_paths = [
        str(data_dir / "company_notes.txt"),
        str(data_dir / "market_notes.txt"),
        str(data_dir / "risk_notes.txt"),
    ]

    result: PipelineResult = run_pipeline(document_paths)

    print(f"Pipeline status: {result.status}")
    print(f"Human review approved: {result.approved}")
    print(f"Stage durations (seconds): {result.stage_durations_seconds}")
    print("-" * 72)

    if result.memo_markdown is None:
        print("No memo was generated -- the reviewer did not approve the analysis.")
        return

    print(result.memo_markdown)

    output_path = Path(__file__).resolve().parent / "sample_output.md"
    output_path.write_text(result.memo_markdown, encoding="utf-8")
    print("-" * 72)
    print(f"Memo written to {output_path}")


if __name__ == "__main__":
    main()
