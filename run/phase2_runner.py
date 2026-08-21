"""BitFracture Phase 2 — powered run.

Repeats the pilot with the pre-registered powered design
(analysis/PREREGISTRATION.md):
  - BOTH models: Qwen3-1.7B and Qwen3-4B (replication leg)
  - both formats: fp16 baseline vs nf4
  - 30 entries per category x 5 categories = n=150 per model-format

Outputs (under /kaggle/working/results_phase2/):
  phase2_results.json   full machine-readable results
  phase2_summary.txt    human-readable report (open this one)
  <model>/<format>/     per-category BFCL result files + labels
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from classify import bootstrap_ci, classify
from gen_responses import generate_one, write_result_file
from pilot_runner import (
    MAX_NEW_TOKENS,
    SEED,
    TAXONOMY_CLASSES,
    _pkg_version,
    assemble_entries,
    load_model,
)
from report import write_summary_txt

# Pre-registered design — do not change after seeing results.
MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
FORMATS = ["fp16", "nf4"]
CATEGORIES = ["simple_python", "parallel", "multiple", "live_simple", "live_multiple"]
CATEGORY_CAP = 30  # 5 cats x 30 = 150 per model-format


def run_cell(
    model_id: str,
    format_name: str,
    bfcl_root: Path,
    out_dir: Path,
    version_info: dict,
) -> dict:
    """Run one model+format combination. Returns its summary dict."""
    print(f"[{model_id} | {format_name}] loading…")
    t_load = time.time()
    model, tokenizer = load_model(format_name, model_id=model_id)
    print(f"[{model_id} | {format_name}] loaded in {time.time() - t_load:.0f}s")

    all_labels: list[str] = []
    per_cat_labels: dict[str, list[str]] = {}
    total_latency = 0.0
    total_out_tokens = 0
    peak_mems: list[float] = []

    for category in CATEGORIES:
        entries = assemble_entries(bfcl_root, category)[:CATEGORY_CAP]
        cat_results = []
        cat_labels = []
        for entry in entries:
            meta = generate_one(model, tokenizer, entry, seed=SEED, max_new_tokens=MAX_NEW_TOKENS)
            label = classify(
                entry,
                meta["result"],
                category,
                max_new_tokens=MAX_NEW_TOKENS,
                output_token_count=meta["output_token_count"],
            )
            meta["label"] = label
            meta["category"] = category
            cat_results.append(meta)
            cat_labels.append(label)
            total_latency += meta["latency"]
            total_out_tokens += meta["output_token_count"]
            peak_mems.append(meta["peak_mem_gb"])
        write_result_file(out_dir / model_id.replace("/", "_") / format_name, category, cat_results)
        per_cat_labels[category] = cat_labels
        all_labels.extend(cat_labels)
        print(f"[{model_id} | {format_name}] {category}: {len(cat_labels)} done")

    del model
    tokenizer = None
    torch.cuda.empty_cache()

    n = len(all_labels)
    summary = {
        "model_id": model_id,
        "format": format_name,
        "n_entries": n,
        "breakdown": {cls: all_labels.count(cls) for cls in TAXONOMY_CLASSES},
        "bootstrap_ci": bootstrap_ci(all_labels, TAXONOMY_CLASSES, seed=SEED),
        "per_category_labels": per_cat_labels,
        "throughput_tok_per_s": round(total_out_tokens / max(total_latency, 1e-9), 2),
        "mean_latency_s": round(total_latency / max(n, 1), 3),
        "peak_mem_gb": round(max(peak_mems), 3) if peak_mems else None,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "versions": {**version_info, "model_id": model_id},
    }
    return summary


def main(bfcl_root: str | Path) -> None:
    work = Path("/kaggle/working")
    bfcl_root = Path(bfcl_root)
    out_dir = work / "results_phase2"

    base_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _pkg_version("transformers"),
        "bitsandbytes": _pkg_version("bitsandbytes"),
        "design": f"{len(MODELS)} models x {len(FORMATS)} formats x {CATEGORY_CAP}/cat",
        "preregistration": "analysis/PREREGISTRATION.md",
    }

    summaries = []
    for model_id in MODELS:
        for fmt in FORMATS:
            print(f"\n===== {model_id} | {fmt} =====")
            summaries.append(run_cell(model_id, fmt, bfcl_root, out_dir, base_versions))
            # Checkpoint after every cell so a crash keeps finished legs.
            with open(out_dir / "phase2_results.json", "w", encoding="utf-8") as f:
                json.dump({"formats": summaries, "taxonomy": TAXONOMY_CLASSES}, f, indent=2)

    combined_path = out_dir / "phase2_results.json"
    print("WROTE", combined_path)
    write_summary_txt(combined_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/bfcl/gorilla-main")
