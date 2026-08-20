"""BitFracture Phase 1 — pilot orchestrator.

Runs the Go/No-Go gate on Qwen3-1.7B in two formats (fp16 baseline vs nf4),
over a small sample of BFCL single-turn core categories, and writes:
  - per-format BFCL-format result files
  - per-format taxonomy breakdown + bootstrap CIs
  - a combined pilot_results.json with versions + gate inputs

Run on Kaggle T4 under /kaggle/working/. Re-verifies NF4 load before trusting
any NF4 numbers (transformers 5.0.0 may differ from Phase 0 flags).

The BFCL prompt/ground-truth root is passed explicitly so the Kaggle wrapper can
point at whichever layout the pinned repo actually has on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from classify import bootstrap_ci, classify, format_breakdown
from gen_responses import MODEL_ID, generate_one, write_result_file

TAXONOMY_CLASSES = [
    "correct",
    "missed_required",
    "unnecessary",
    "wrong_tool",
    "wrong_args",
    "malformed",
    "failed_exec",
    "truncation",
]

SEED = 0
MAX_NEW_TOKENS = 192

# Pilot: BFCL single-turn core categories. Sample is capped by CATEGORY_CAP.
CATEGORIES = ["simple_python", "parallel", "multiple", "live_simple", "live_multiple"]
CATEGORY_CAP = 10  # -> up to ~30-50 pilot entries across the categories above


def resolve_files(bfcl_root: Path, category: str) -> tuple[Path | None, Path | None]:
    """Best-effort resolve the prompt and ground-truth files for a category.

    Returns (prompt_file, ground_truth_file). Both may be None if the layout
    of this BFCL checkout doesn't match any known pattern.
    """
    prompt_paths = [
        Path("data") / category / f"{category}.json",
        Path("data") / category / "prompt.json",
        Path("bfcl") / "data" / category / f"{category}.json",
        Path("bfcl_eval") / "data" / "eval_data" / "single_turn" / category / f"{category}.json",
    ]
    gt_paths = [
        Path("possible_answer") / f"{category}.json",
        Path("data") / category / "answer.json",
        Path("bfcl") / "data" / "possible_answer" / f"{category}.json",
        Path("bfcl_eval") / "data" / "possible_answer" / f"{category}.json",
    ]
    prompt_file = next((bfcl_root / c for c in prompt_paths if (bfcl_root / c).exists()), None)
    gt_file = next((bfcl_root / c for c in gt_paths if (bfcl_root / c).exists()), None)
    return prompt_file, gt_file


def load_ground_truth(gt_file: Path | None) -> dict[str, dict]:
    """Index ground-truth entries by id (best-effort)."""
    if gt_file is None or not gt_file.exists():
        return {}
    gt_list = json.loads(gt_file.read_text(encoding="utf-8"))
    if isinstance(gt_list, list):
        return {e.get("id", str(i)): e for i, e in enumerate(gt_list)}
    if isinstance(gt_list, dict):
        return gt_list
    return {}


def assemble_entries(bfcl_root: Path, category: str) -> list[dict]:
    """Merge prompt entries with their ground truth into one working list."""
    prompt_file, gt_file = resolve_files(bfcl_root, category)
    if prompt_file is None:
        print(f"[data] no prompt file found for {category} (bfcl_root={bfcl_root})")
        return []
    prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
    gt_index = load_ground_truth(gt_file)

    entries = []
    for i, p in enumerate(prompts):
        entry = dict(p)
        entry["ground_truth"] = gt_index.get(p.get("id", str(i)), {}).get("ground_truth")
        entries.append(entry)
    return entries


def load_model(format_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if format_name == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
    elif format_name == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown format: {format_name}")
    return model, tokenizer


def run_format(
    format_name: str,
    bfcl_root: Path,
    out_dir: Path,
    version_info: dict,
) -> dict:
    """Generate + classify + summarize one format. Returns a summary dict."""
    print(f"[{format_name}] loading model…")
    model, tokenizer = load_model(format_name)
    print(f"[{format_name}] loaded.")

    all_labels: list[str] = []
    per_cat_labels: dict[str, list[str]] = {}

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
            cat_results.append(meta)
            cat_labels.append(label)
        write_result_file(out_dir / format_name, category, cat_results)
        per_cat_labels[category] = cat_labels
        all_labels.extend(cat_labels)
        print(f"[{format_name}] {category}: {len(cat_labels)} entries done")

    del model
    torch.cuda.empty_cache()

    breakdown = format_breakdown(all_labels, TAXONOMY_CLASSES)
    ci = bootstrap_ci(all_labels, TAXONOMY_CLASSES, seed=SEED)

    summary = {
        "format": format_name,
        "n_entries": len(all_labels),
        "breakdown": dict(breakdown),
        "bootstrap_ci": ci,
        "per_category_labels": per_cat_labels,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "versions": version_info,
    }
    return summary


def main(bfcl_root: str | Path) -> None:
    work = Path("/kaggle/working")
    bfcl_root = Path(bfcl_root)
    out_dir = work / "results_pilot"

    version_info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_id": MODEL_ID,
        "transformers": _pkg_version("transformers"),
        "bitsandbytes": _pkg_version("bitsandbytes"),
    }

    summaries = []
    for fmt in ["fp16", "nf4"]:
        print(f"\n===== {fmt} =====")
        summary = run_format(fmt, bfcl_root, out_dir, version_info)
        summaries.append(summary)

    combined = {"formats": summaries, "taxonomy": TAXONOMY_CLASSES}
    with open(out_dir / "pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print("WROTE", out_dir / "pilot_results.json")


def _pkg_version(pkg: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except Exception:
        return None


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/bfcl/gorilla-main")
