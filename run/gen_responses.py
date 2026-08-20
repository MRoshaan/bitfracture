"""BitFracture Phase 1 — generate BFCL-format responses with a custom HF script.

We cannot use BFCL's built-in OSS generation path on Kaggle T4 (it expects
vLLM/sglang; T4 is SM75 and sglang needs SM80+, and we need identical decoding
across quantization formats). So we generate responses ourselves with
transformers, then write them in BFCL's single-turn result-file format so that
`bfcl evaluate` (and our own taxonomy classifier) can consume them.

Controlled decoding shared by every format: greedy (temp=0), fixed seed,
enable_thinking=False. Only the quantization of the loaded weights differs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-1.7B"


def build_messages(entry: dict) -> list[dict]:
    """Reconstruct the chat messages for a BFCL single-turn entry.

    BFCL `question` is a list of turns; each turn is a list of role/content
    messages. Single-turn entries use the first (and usually only) turn.
    """
    question = entry["question"]
    if question and isinstance(question[0], list):
        messages = list(question[0])
    else:
        messages = [m for m in question if isinstance(m, dict)]
    return messages


def build_tools(entry: dict) -> list[dict] | None:
    """Convert BFCL `function` docs into transformers `tools=` schema."""
    functions = entry.get("function")
    if not functions:
        return None
    tools = []
    for fn in functions:
        schema = {"type": "function", "function": {"name": fn["name"]}}
        if fn.get("description"):
            schema["function"]["description"] = fn["description"]
        params = fn.get("parameters")
        if params:
            # Normalize BFCL's non-standard schema type for transformers.
            params = dict(params)
            if params.get("type") in ("dict", "object"):
                params["type"] = "object"
            schema["function"]["parameters"] = params
        tools.append(schema)
    return tools


def apply_template(tokenizer, messages: list[dict], tools: list[dict] | None) -> str:
    """Apply the chat template with tools; fall back to tools-less on older builds."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )


def generate_one(model, tokenizer, entry: dict, seed: int, max_new_tokens: int) -> dict:
    """Generate one response; return a BFCL result-file entry dict (or None on failure)."""
    torch.manual_seed(seed)
    text = apply_template(tokenizer, build_messages(entry), build_tools(entry))
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    prompt_len = inputs["input_ids"].shape[-1]
    raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    mem = torch.cuda.max_memory_allocated() / 1e9
    n_out = int(out.shape[1] - prompt_len)
    return {
        "id": entry["id"],
        "result": raw,
        "input_token_count": int(prompt_len),
        "output_token_count": n_out,
        "latency": round(dt, 3),
        "peak_mem_gb": round(mem, 3),
    }


def write_result_file(out_dir: Path, category: str, entries: list[dict]) -> Path:
    """Write a BFCL-format NDJSON result file for one model+category."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"BFCL_v3_{category}_result.json"
    entries = sorted(entries, key=lambda e: e["id"])
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path
