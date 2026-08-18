"""BitFracture Phase 0 env-check kernel.

Goals:
  1. Record the installed versions of the experiment stack on Kaggle T4.
  2. Load Qwen3-1.7B in FP16 and BNB-NF4; generate one tool-call prompt each.
  3. Probe W4A16 via AutoGPTQ (small on-the-fly quantize) for feasibility.
  4. Write versions.json + results.json to /kaggle/working/.

Findings are recorded verbatim; failures are reported, not hidden.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import subprocess
import sys
import time

import torch

MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = "Get the current weather for Tokyo."

PIP_DEPS = [
    "transformers==4.51.3",
    "accelerate==1.5.0",
    "bitsandbytes==0.45.0",
    "auto-gptq==2.7.1",
    "bfcl-eval",
]


def pip_install(pkg: str) -> None:
    print(f"[pip] {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def collect_versions() -> dict[str, object]:
    out: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    for pkg in [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "auto_gptq",
        "bfcl_eval",
        "numpy",
    ]:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def make_prompt(tokenizer, use_tools: bool = True) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT},
    ]
    tools = None
    if use_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def run(model, tokenizer, name: str) -> dict[str, object]:
    text = make_prompt(tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{name}] time={dt:.2f}s peak_mem={mem:.2f}GB")
    print(f"[{name}] output={decoded[:200]!r}")
    return {
        "name": name,
        "seconds": round(dt, 2),
        "peak_mem_gb": round(mem, 2),
        "output_preview": decoded[:200],
    }


def probe_autogptq(tokenizer) -> dict[str, object]:
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    except Exception as exc:
        return {"w4a16": "import_failed", "detail": str(exc)[:300]}
    quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
    samples = [{"prompt": make_prompt(tokenizer)} for _ in range(4)]
    t0 = time.time()
    try:
        model = AutoGPTQForCausalLM.from_pretrained(
            MODEL_ID, quantize_config=quantize_config, torch_dtype=torch.float16, device="cuda:0"
        )
        model.quantize(samples, cache_examples_on_gpu=False)
        return {
            "w4a16": "quantized_ok",
            "seconds": round(time.time() - t0, 2),
            "version": md.version("auto_gptq"),
        }
    except Exception as exc:
        return {"w4a16": "failed", "detail": str(exc)[:300]}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    for pkg in PIP_DEPS:
        try:
            pip_install(pkg)
        except Exception as exc:
            print(f"WARN: failed to install {pkg}: {exc}")

    vers = collect_versions()
    print("[versions]", json.dumps(vers, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    results: list[dict[str, object]] = []

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    results.append(run(fp16, tokenizer, "fp16"))
    del fp16
    torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    nf4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    results.append(run(nf4, tokenizer, "nf4"))
    del nf4
    torch.cuda.empty_cache()

    gptq = probe_autogptq(tokenizer)
    print("[w4a16]", json.dumps(gptq, indent=2))

    with open("/kaggle/working/versions.json", "w") as f:
        json.dump(vers, f, indent=2)
    with open("/kaggle/working/results.json", "w") as f:
        json.dump({"runs": results, "w4a16_probe": gptq}, f, indent=2)
    print("WROTE /kaggle/working/versions.json and results.json")


if __name__ == "__main__":
    main()
