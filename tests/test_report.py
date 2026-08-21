"""Tests for run/report.py — human-readable summary rendering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))

from classify import bootstrap_diff_ci  # noqa: E402
from report import render_report  # noqa: E402

CLASSES = ["correct", "missed_required", "truncation"]


def _summary(fmt: str, labels: list[str]) -> dict:
    n = len(labels)
    return {
        "model_id": "Qwen/Qwen3-1.7B",
        "format": fmt,
        "n_entries": n,
        "breakdown": {c: labels.count(c) for c in CLASSES},
        "bootstrap_ci": {
            c: {"mean": labels.count(c) / n, "ci_lo": 0.0, "ci_hi": 1.0} for c in CLASSES
        },
        "per_category_labels": {"cat1": labels},
        "versions": {"torch": "test"},
    }


def test_bootstrap_diff_ci_direction_and_bounds():
    fp = ["correct"] * 9 + ["missed_required"]
    nf = ["correct"] * 6 + ["missed_required"] * 4
    diff = bootstrap_diff_ci(fp, nf, CLASSES, n_boot=500, seed=1)
    # NF4 has more missed_required -> delta (fp - nf) is negative.
    assert diff["missed_required"]["mean"] < 0
    assert diff["correct"]["mean"] > 0
    assert diff["truncation"]["ci_lo"] <= diff["truncation"]["ci_hi"]


def test_render_report_contains_comparison_and_verdicts():
    results = {
        "taxonomy": CLASSES,
        "formats": [
            _summary("fp16", ["correct"] * 8 + ["missed_required"] * 2),
            _summary("nf4", ["correct"] * 5 + ["missed_required"] * 4 + ["truncation"]),
        ],
    }
    text = render_report(results)
    assert "COMPARISON  fp16 vs nf4" in text
    assert "SOLID" in text or "(may be luck)" in text
    assert "WHAT EACH MISTAKE TYPE MEANS" in text
