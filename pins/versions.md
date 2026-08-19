# BitFracture — Versions & Phase 0 Findings

Recorded from Kaggle T4 env-check run `bitfracture-env-check-t4-v3` (2026-08-19).
Note: loose pins caused pip to resolve NEWER versions than the original 4.51.3 plan.
These are the ACTUAL versions that load on the current Kaggle image.

## Installed stack (Kaggle T4, Python 3.12.13)
| Package | Version | Status |
|---|---|---|
| torch | 2.10.0+cu128 | OK (CUDA 12.8) |
| transformers | 5.0.0 | OK |
| accelerate | 1.13.0 | OK |
| bitsandbytes | 0.50.1 | OK |
| auto-gptq | null | **IMPORT FAILED — unavailable for this env** |
| bfcl-eval | 2026.3.23 | OK |
| numpy | 1.26.4 | OK |
| CUDA available | true | OK |

## Load/generate results
- **FP16**: Qwen3-1.7B loads + emits correct tool call `<get_weather city=Tokyo>`. 2.23s, 2.38GB peak.
- **NF4**: loads + emits correct tool call. 1.52s, 2.36GB peak.
- **W4A16 (AutoGPTQ)**: import failed — no wheel for this torch/CUDA/Python combo.

## Findings / decisions
1. FP16 and NF4 both proven on T4 with correct tool-calling output. -> Pilot can run these two.
2. **W4A16 is the open blocker.** auto-gptq has no build for torch 2.10 / CUDA 12.8 / Python 3.12 on PyPI's wheel set.
   Options (for Phase 1 decision):
   a. Drop W4A16, run **FP16 vs NF4** two-level comparison (still valid: 4-bit vs 16-bit error-profile shift).
   b. Use **AutoAWQ** instead (if it has a compatible wheel) — needs a quick probe.
   c. Use a pre-quantized GPTQ checkpoint from HF Hub (e.g., trust-remote-code GPTQ repos) with the transformers GPTQ integration (quant_method="gptq") — avoids auto-gptq pip package entirely.
3. Plan the Phase 1 matrix around this: either 2-formats (safe) or resolve W4A16 via option (c).
