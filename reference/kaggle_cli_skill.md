# Kaggle CLI GPU Execution (reference)

> Source: brother's gist https://gist.github.com/aaliyan1230/dd72b04d1c64d0318f5d2a1eb381bb92
> Curated essentials for the BitFracture project. Verify against the live gist before relying on it.

## Core workflow
```
pip install kaggle
# place API token at ~/.kaggle/kaggle.json (or use ACCESS_TOKEN auth)
kaggle config view
```

A script kernel dir needs two files:
- `kernel-metadata.json`
- `script.py` (entrypoint; write all artifacts under `/kaggle/working/`)

Push / monitor / pull:
```
kaggle kernels push -p ./kernel-dir/ --accelerator NvidiaTeslaT4
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p ./local-out/
```

## GPU selection
- `enable_gpu: true` only *enables* GPU; it does NOT pick the type.
- Kaggle defaults to P100 unless an accelerator is passed.
- Request T4 with `--accelerator NvidiaTeslaT4`.
- Valid: `NvidiaTeslaP100`, `NvidiaTeslaT4`, `TpuVm`.
- Do NOT put `accelerator` in `kernel-metadata.json` — the API ignores it; use the CLI flag.

## Secrets / private data
- CLI-pushed scripts do NOT receive notebook UI secrets or local env vars.
- Use a PRIVATE Kaggle dataset for secrets; attach via `dataset_sources`.
- Read from `/kaggle/input/<dataset-slug>/`.

## Install / deps
- Avoid `pip install -e .` in kernels (build backend/conflicts).
- Prefer `git clone --depth 1 <repo> /tmp/repo` + `sys.path.insert(0, "/tmp/repo/src")`.
- Pin versions when compatibility matters.

## Runtime details
- T4 = Turing (SM75): use fp16, NOT bf16 (no native bf16); avoid flash-attn unless verified.
- Max runtime ~12h/session; internet with `enable_internet: true`.
- Output only under `/kaggle/working/`.
- Queue waits are normal.

## Pitfalls
- Wrong GPU type -> re-push with `--accelerator NvidiaTeslaT4`.
- Missing outputs -> ensure files under `/kaggle/working/`.
- Editable install failure -> `sys.path.insert`.
- Gated model denied -> accept license on source site with the token account.
- No CLI cancel -> re-push a new version or stop from web UI.
