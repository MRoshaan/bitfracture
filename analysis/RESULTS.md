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

---

# BitFracture — Phase 2 Results

Status: **WEAK / MIXED per PREREGISTRATION.md — size-dependent effect** (see GATE.md).
Data from results/phase2/results_phase2/phase2_results.json (+ phase2_summary.txt).

## Run metadata
- Kernel: `muhammadroshaan/bitfracture-phase2-fp16-nf4-v1` (2026-08-22)
- Models: Qwen/Qwen3-1.7B + Qwen/Qwen3-4B; formats fp16 vs nf4
- Same pinned BFCL commit, categories, decoding, stack as Phase 1
- n = 150 per model-format (30/category x 5); pre-registration committed before run

## Headline numbers: missed_required (fp16 -> nf4)
| Model | fp16 | nf4 | Δ(fp−nf) | 95% CI | Verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B | 6.0% | **17.3%** | −11.3pp | [−19.3, −4.7] | **SOLID** |
| Qwen3-4B | 7.3% | 7.3% | +0.0pp | [−6.0, +6.0] | no effect |

## Full comparison highlights
- **1.7B:** correct 74.7%→67.3% (Δ CI crosses 0); wrong_args 17.3%→12.0%;
  truncation 0.7%→2.0%. The missed_required shift is spread across ALL five
  categories under nf4 (4/3/3/8/8) vs concentrated in live_* under fp16 (0/0/0/2/7).
- **4B:** NO class shifts (all Δ CIs include 0). NF4 is nominally *better*:
  correct 79.3%→84.0%, wrong_args 10.7%→6.7%, missed_required flat at 11/11.
  A powered null at n=150 bounds any 4B missed_required effect to ±6pp.

## Reading (honest, per pre-registered rule)
1. Pre-registered CONFIRMED required BOTH models to show a solid shift → NOT met.
2. Outcome is WEAK/MIXED: strong, replicated-in-magnitude effect in 1.7B
   (11.3pp, CI excludes 0) and a clean null in 4B.
3. Per-category breakdown supports a real interaction rather than noise: the 1.7B
   nf4 missed_required errors are distributed across every category, not one artifact.
4. Honest claim candidate: "4-bit NF4 quantization degrades tool-calling DISCIPLINE
   (missed required calls) in the 1.7B model but not the 4B — fragility is size-dependent."
   This must be framed as a size×format interaction finding, NOT a universal claim.
5. Throughput/memory recorded: 4B nf4 uses ~5.9GB peak vs ~7.8GB fp16 (~25% saving);
   all formats remain usable latency-wise (11-18 tok/s on T4).
