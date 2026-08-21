"""BitFracture — human-readable result reports.

Turns the machine-oriented results JSON into a plain-text summary anyone can
open and read (no tooling needed). Every Phase 2 run writes BOTH:
  phase2_results.json   (full data, for the pipeline)
  phase2_summary.txt    (this report, for humans)

CLI usage:  python report.py <results.json> [out.txt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from classify import bootstrap_diff_ci

LINE = "=" * 64
THIN = "-" * 64

CLASS_EXPLANATIONS = {
    "correct": "model called the right tool with the right arguments",
    "missed_required": "model FORGOT to call a tool it should have called",
    "unnecessary": "model called a tool when none was needed",
    "wrong_tool": "model picked the wrong tool",
    "wrong_args": "right tool, but bad/missing arguments",
    "malformed": "output could not be understood as a tool call",
    "failed_exec": "tool call looked fine but would fail when run",
    "truncation": "answer got cut off before finishing",
}


def _model_id(summary: dict) -> str:
    """Model id from either Phase 2 shape (top-level) or pilot shape (versions)."""
    if "model_id" in summary:
        return summary["model_id"]
    return summary.get("versions", {}).get("model_id", "unknown-model")


def render_format_section(summary: dict) -> list[str]:
    """Render one model+format block: counts, percentages, CIs."""
    n = summary["n_entries"]
    breakdown = summary["breakdown"]
    ci = summary["bootstrap_ci"]
    lines = [
        f"Model: {_model_id(summary)}   Format: {summary['format'].upper()}",
        f"Questions answered: {n}",
        THIN,
        f"{'Mistake type':<18}{'count':>7}{'percent':>10}{'95% range':>16}",
        THIN,
    ]
    for cls, count in breakdown.items():
        pct = 100.0 * count / n
        entry = ci.get(cls, {})
        lo, hi = entry.get("ci_lo", 0.0), entry.get("ci_hi", 0.0)
        lines.append(f"{cls:<18}{count:>7}{pct:>9.1f}%{f'[{lo * 100:.0f}%-{hi * 100:.0f}%]':>16}")
    return lines


def render_diff_section(
    labels_fp: list[str], labels_nf: list[str], classes: list[str]
) -> list[str]:
    """Render the fp16-vs-nf4 comparison table with two-sample bootstrap CIs."""
    diff = bootstrap_diff_ci(labels_fp, labels_nf, classes)
    n_fp, n_nf = len(labels_fp), len(labels_nf)
    c_fp: dict[str, int] = {}
    c_nf: dict[str, int] = {}
    for lbl in labels_fp:
        c_fp[lbl] = c_fp.get(lbl, 0) + 1
    for lbl in labels_nf:
        c_nf[lbl] = c_nf.get(lbl, 0) + 1
    lines = [
        THIN,
        f"COMPARISON  fp16 vs nf4   (n={n_fp} vs n={n_nf})",
        "'diff' = fp16 percent minus nf4 percent.",
        "'range' is the 95% confidence interval of that difference:",
        "  if the range crosses 0, the gap might be luck.",
        "  if it does NOT cross 0, the difference is statistically solid.",
        THIN,
        f"{'Mistake type':<18}{'fp16':>8}{'nf4':>8}{'diff':>9}{'range':>18}  verdict",
        THIN,
    ]
    for cls in classes:
        d = diff[cls]
        p_fp = 100.0 * c_fp.get(cls, 0) / n_fp
        p_nf = 100.0 * c_nf.get(cls, 0) / n_nf
        lo, hi = d["ci_lo"] * 100, d["ci_hi"] * 100
        rng = f"[{lo:+.1f} to {hi:+.1f}]"
        if lo > 0 or hi < 0:
            margin = min(abs(lo), abs(hi))
            verdict = "SOLID" if margin >= 2.0 else "SOLID but borderline"
        else:
            verdict = "(may be luck)"
        lines.append(
            f"{cls:<18}{p_fp:>7.1f}%{p_nf:>7.1f}%{d['mean'] * 100:>+8.1f}%{rng:>18}  {verdict}"
        )
    return lines


def render_report(results: dict) -> str:
    """Render the full plain-text report from a combined results dict."""
    classes = results["taxonomy"]
    formats = results["formats"]
    lines: list[str] = [
        LINE,
        "BITFRACTURE RESULTS SUMMARY (plain text)",
        LINE,
        "",
        "What this file is: a simple readable version of the experiment.",
        "We asked small AI models to answer questions by calling tools,",
        "in a normal version (FP16) and a compressed/squeezed version (NF4).",
        "Then we counted what KIND of mistakes each version made.",
        "",
    ]
    # Group summaries by model so each model gets its own comparison.
    by_model: dict[str, dict[str, dict]] = {}
    for s in formats:
        by_model.setdefault(_model_id(s), {})[s["format"]] = s

    for model_id, fmts in by_model.items():
        lines.append(LINE)
        lines.append(f"MODEL: {model_id}")
        lines.append(LINE)
        for fmt in ("fp16", "nf4"):
            if fmt in fmts:
                lines.extend(render_format_section(fmts[fmt]))
                lines.append("")
        if "fp16" in fmts and "nf4" in fmts:
            per_cat_fp = fmts["fp16"].get("per_category_labels", {})
            per_cat_nf = fmts["nf4"].get("per_category_labels", {})
            labels_fp = [lab for cats in per_cat_fp.values() for lab in cats]
            labels_nf = [lab for cats in per_cat_nf.values() for lab in cats]
            lines.extend(render_diff_section(labels_fp, labels_nf, classes))
            lines.append("")

    lines.append(THIN)
    lines.append("WHAT EACH MISTAKE TYPE MEANS")
    lines.append(THIN)
    for cls in classes:
        expl = CLASS_EXPLANATIONS.get(cls, "")
        marker = "*" if cls == "missed_required" else " "
        lines.append(f"{marker}{cls:<18} {expl}")
    lines.append("")
    versions = formats[0].get("versions", {}) if formats else {}
    if versions:
        lines.append(THIN)
        lines.append("SOFTWARE USED")
        lines.append(THIN)
        for k, v in versions.items():
            lines.append(f"  {k}: {v}")
    lines.append(LINE)
    return "\n".join(lines) + "\n"


def write_summary_txt(results_path: Path, out_path: Path | None = None) -> Path:
    """Load a combined results JSON and write its .txt report next to it."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    out_path = out_path or results_path.with_name(
        results_path.stem.replace("_results", "_summary") + ".txt"
    )
    out_path.write_text(render_report(results), encoding="utf-8")
    print(f"WROTE {out_path}")
    return out_path


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    write_summary_txt(src, dst)
