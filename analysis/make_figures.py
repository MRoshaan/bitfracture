"""Generate paper figures from Phase 2 results.

Reads results/phase2/results_phase2/phase2_results.json and writes PDF
figures into paper/figures/. Run from repo root on the machine that has
the pulled kernel output:

    python analysis/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase2" / "results_phase2" / "phase2_results.json"
OUT = ROOT / "paper" / "figures"

SHORT_MODEL = {"Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-4B": "Qwen3-4B"}
CLASSES = [
    "correct",
    "missed_required",
    "wrong_args",
    "truncation",
    "wrong_tool",
    "unnecessary",
    "malformed",
    "failed_exec",
]
COLORS = {"fp16": "#4878d0", "nf4": "#d65f5f"}


def load() -> list[dict]:
    return json.loads(RESULTS.read_text(encoding="utf-8"))["formats"]


def by_cell(formats: list[dict]) -> dict[tuple[str, str], dict]:
    return {(s["model_id"], s["format"]): s for s in formats}


def fig_distribution(cell: dict[tuple[str, str], dict]) -> None:
    """Grouped bars: error-type distribution per format, one panel per model."""
    models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    width = 0.38
    for ax, model in zip(axes, models):
        n = cell[(model, "fp16")]["n_entries"]
        xs = range(len(CLASSES))
        for i, fmt in enumerate(("fp16", "nf4")):
            s = cell[(model, fmt)]
            pct = [100 * s["breakdown"].get(c, 0) / n for c in CLASSES]
            ax.bar(
                [x + (i - 0.5) * width for x in xs],
                pct,
                width,
                label=fmt.upper(),
                color=COLORS[fmt],
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(
            ["corr", "miss", "warg", "trunc", "wtl", "unn", "malf", "exec"], fontsize=8
        )
        ax.set_title(SHORT_MODEL[model], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of answers")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_distribution.pdf")
    plt.close(fig)


def fig_interaction(cell: dict[tuple[str, str], dict]) -> None:
    """Size x format interaction plot for missed_required with bootstrap CIs."""
    models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i, fmt in enumerate(("fp16", "nf4")):
        xs, ys, lo_s, hi_s = [], [], [], []
        for j, model in enumerate(models):
            s = cell[(model, fmt)]
            ci = s["bootstrap_ci"]["missed_required"]
            xs.append(j + (i - 0.5) * 0.06)
            ys.append(100 * ci["mean"])
            lo_s.append(100 * (ci["mean"] - ci["ci_lo"]))
            hi_s.append(100 * (ci["ci_hi"] - ci["mean"]))
        ax.errorbar(
            xs, ys, yerr=[lo_s, hi_s], marker="o", capsize=4, label=fmt.upper(), color=COLORS[fmt]
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([SHORT_MODEL[m] for m in models])
    ax.set_ylabel("missed_required (% of answers)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "interaction_missed_required.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell = by_cell(load())
    fig_distribution(cell)
    fig_interaction(cell)
    print(f"WROTE figures in {OUT}")


if __name__ == "__main__":
    main()
