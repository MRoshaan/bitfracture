# BitFracture — Phase 1 Pilot Results

Status: **CONDITIONAL GO** (see GATE.md). Data from results/phase1/results_pilot/pilot_results.json.

## Run metadata
- Kernel: `muhammadroshaan/bitfracture-pilot-fp16-nf4-v3` (2026-08-20)
- Model: Qwen/Qwen3-1.7B; formats fp16 vs nf4 (bitsandbytes)
- BFCL commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (bfcl-eval 2026.3.23)
- Categories: simple_python, parallel, multiple, live_simple, live_multiple (10 each)
- Decoding: greedy, temp=0, seed=0, enable_thinking=False, max_new_tokens=192
- Stack: torch 2.10.0+cu128, transformers 5.0.0, bitsandbytes 0.50.1, python 3.12.13
- n = 50 per format; identical prompt/decoding across formats (only quantization differs)

## Taxonomy breakdown (counts / proportions / 95% CI)
| Class | fp16 | nf4 |
|---|---|---|
| correct | 39 / 0.78 [0.66,0.88] | 35 / 0.70 [0.58,0.82] |
| missed_required | 1 / 0.02 [0.00,0.06] | **6 / 0.12 [0.04,0.22]** |
| unnecessary | 0 | 0 |
| wrong_tool | 0 | 0 |
| wrong_args | 10 / 0.20 [0.10,0.32] | 7 / 0.14 [0.06,0.24] |
| malformed | 0 | 0 |
| failed_exec | 0 | 0 |
| truncation | 0 | **2 / 0.04 [0.00,0.10]** |

## Two-sample bootstrap on Δ (fp − nf), 95% CI
| Class | Δ | 95% CI | includes 0? |
|---|---|---|---|
| correct | +0.080 | [−0.08, +0.26] | YES |
| missed_required | −0.100 | [−0.20, +0.00] | boundary |
| wrong_args | +0.060 | [−0.08, +0.20] | YES |
| truncation | −0.040 | [−0.10, +0.00] | YES |

## Reading (honest)
- Pipeline works end-to-end and produced real, comparable taxonomy output.
- Directional signal: NF4 increases `missed_required` (2%→12%) and adds `truncation`
  (0→4%); `wrong_args` decreases slightly.
- At n=50 per format the shift is NOT statistically significant (all Δ CIs include 0).
  It is a candidate worth a powered confirmation, not an established effect.

## Next
- Phase 2 (Aug 22-26): both models (1.7B + 4B) × both formats × full core categories,
  with a pre-registered powered decision threshold so the outcome is a confirmed shift
  or a precise powered null.
- If powered-null on missed_required/truncation -> pivot to Rank 1 (per ROADMAP.md:79).
