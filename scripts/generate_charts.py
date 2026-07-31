#!/usr/bin/env python
"""Renders real PNG charts from an actual evaluation run and pipeline run.

Produces two charts into ``docs/images/``:

1. ``eval_scores_by_case.png`` -- grouped bar chart of the three rubric
   dimension scores (factual coverage, citation validity, completeness) for
   each labeled test case, from a real ``eval_harness.evaluate()`` run.
2. ``pipeline_latency_by_stage.png`` -- bar chart of wall-clock time spent
   in each pipeline stage for one real pipeline run, measured with
   ``time.perf_counter()`` inside ``orchestrator.pipeline.run_pipeline``
   (see ``PipelineResult.stage_durations_seconds``). No numbers in either
   chart are invented.

Run from the repository root:

    python scripts/generate_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orchestrator.eval.eval_harness import EvalReport, default_test_set, evaluate
from orchestrator.pipeline import PipelineResult, run_pipeline

# --- Palette (validated categorical + chrome roles; light surface) --------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SLOT_BLUE = "#2a78d6"
SLOT_GREEN = "#008300"
SLOT_MAGENTA = "#e87ba4"

OUTPUT_DIR = _REPO_ROOT / "docs" / "images"


def _apply_chart_chrome(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def render_eval_scores_chart(report: EvalReport, output_path: Path) -> None:
    """Renders a grouped bar chart of rubric dimension scores per test case."""
    case_names = [case.case_name for case in report.case_results]
    factual = [case.factual_coverage for case in report.case_results]
    citation = [case.citation_validity for case in report.case_results]
    completeness = [case.completeness for case in report.case_results]

    x = np.arange(len(case_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _apply_chart_chrome(ax)

    bars_factual = ax.bar(
        x - bar_width, factual, bar_width, label="Factual coverage", color=SLOT_BLUE, zorder=3
    )
    bars_citation = ax.bar(
        x, citation, bar_width, label="Citation validity", color=SLOT_GREEN, zorder=3
    )
    bars_completeness = ax.bar(
        x + bar_width, completeness, bar_width, label="Completeness", color=SLOT_MAGENTA, zorder=3
    )

    for bar_group in (bars_factual, bars_citation, bars_completeness):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_PRIMARY,
            )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0-1)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Evaluation rubric scores by test case (real eval_harness run)",
        color=INK_PRIMARY,
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, color=INK_SECONDARY)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)


def render_latency_chart(stage_durations_seconds: dict[str, float], output_path: Path) -> None:
    """Renders a bar chart of measured wall-clock time per pipeline stage."""
    stage_order = ["researcher_agent", "analyst_agent", "reviewer_agent", "writer_agent"]
    stages = [s for s in stage_order if s in stage_durations_seconds]
    durations_ms = [stage_durations_seconds[s] * 1000 for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _apply_chart_chrome(ax)

    bars = ax.bar(stages, durations_ms, width=0.5, color=SLOT_BLUE, zorder=3)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_PRIMARY,
        )

    ax.set_ylabel("Wall-clock time (milliseconds)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Pipeline stage latency, one real run (time.perf_counter)",
        color=INK_PRIMARY,
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, color=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running eval_harness against the default labeled test set...")
    report = evaluate(default_test_set())
    for case in report.case_results:
        print(
            f"  [{case.case_name}] factual_coverage={case.factual_coverage:.2f} "
            f"citation_validity={case.citation_validity:.2f} "
            f"completeness={case.completeness:.2f} aggregate={case.aggregate:.2f}"
        )
    print(f"  aggregate_score={report.aggregate_score:.2f}")

    eval_chart_path = OUTPUT_DIR / "eval_scores_by_case.png"
    render_eval_scores_chart(report, eval_chart_path)
    print(f"Wrote {eval_chart_path}")

    print("\nRunning one real pipeline execution to measure stage latency...")
    data_dir = _REPO_ROOT / "data" / "sample_documents"
    document_paths = [
        str(data_dir / "company_notes.txt"),
        str(data_dir / "market_notes.txt"),
        str(data_dir / "risk_notes.txt"),
    ]
    result: PipelineResult = run_pipeline(document_paths)
    for stage, duration in result.stage_durations_seconds.items():
        print(f"  {stage}: {duration * 1000:.3f} ms")

    latency_chart_path = OUTPUT_DIR / "pipeline_latency_by_stage.png"
    render_latency_chart(result.stage_durations_seconds, latency_chart_path)
    print(f"Wrote {latency_chart_path}")


if __name__ == "__main__":
    main()
