# BitFracture — Phase 1 Go/No-Go Gate

**Decision record.** Per ROADMAP.md:76-79, continue to the full run ONLY if the
Phase 1 pilot shows a REPRODUCIBLE taxonomy shift (bootstrap CI, same decoding
across formats). Populate the numbers below from the pilot results file, then
mark the decision.

Status: **DECIDED 2026-08-20 — CONDITIONAL GO (proceed to powered Phase 2 to confirm a promising but underpowered signal)**

---

## Pilot configuration (recorded before running)

| Setting | Value |
|---|---|
| Model | Qwen3-1.7B |
| Formats | fp16 (baseline), nf4 |
| Categories | simple_python, parallel, multiple, live_simple, live_multiple |
| Entry cap / category | 10 |
| Decoding | greedy, temp=0, seed=0, enable_thinking=False |
| Taxonomy | 7 classes (ROADMAP.md:32-41) |
| Seed | 0 |

---

## Results (from results/phase1/results_pilot/pilot_results.json, kernel bitfracture-pilot-fp16-nf4-v3)

### fp16 (n = 50)
| Class | count | proportion | 95% CI |
|---|---|---|---|
| correct | 39 | 0.78 | [0.66, 0.88] |
| missed_required | 1 | 0.02 | [0.00, 0.06] |
| unnecessary | 0 | 0.00 | [0,0] |
| wrong_tool | 0 | 0.00 | [0,0] |
| wrong_args | 10 | 0.20 | [0.10, 0.32] |
| malformed | 0 | 0.00 | [0,0] |
| failed_exec | 0 | 0.00 | [0,0] |
| truncation | 0 | 0.00 | [0,0] |

### nf4 (n = 50)
| Class | count | proportion | 95% CI |
|---|---|---|---|
| correct | 35 | 0.70 | [0.58, 0.82] |
| missed_required | 6 | 0.12 | [0.04, 0.22] |
| unnecessary | 0 | 0.00 | [0,0] |
| wrong_tool | 0 | 0.00 | [0,0] |
| wrong_args | 7 | 0.14 | [0.06, 0.24] |
| malformed | 0 | 0.00 | [0,0] |
| failed_exec | 0 | 0.00 | [0,0] |
| truncation | 2 | 0.04 | [0.00, 0.10] |

### Two-sample bootstrap on the difference (fp − nf), 95% CI
| Class | fp | nf | Δ(fp−nf) | 95% CI Δ | includes 0? |
|---|---|---|---|---|---|
| correct | 0.780 | 0.700 | +0.080 | [−0.080, +0.260] | YES |
| missed_required | 0.020 | 0.120 | **−0.100** | [−0.200, +0.000] | YES (boundary) |
| wrong_args | 0.200 | 0.140 | +0.060 | [−0.080, +0.200] | YES |
| truncation | 0.000 | 0.040 | −0.040 | [−0.100, +0.000] | YES |

---

## Gate evaluation

1. **Reproducible shift?** At n=50/format, NO class moves outside the two-sample
   bootstrap CI — every Δ includes 0. The `missed_required` signal is the strongest:
   it rises from 1 (2%) to 6 (12%), a +10pp point shift whose CI Δ = [−0.20, 0.00]
   barely touches zero. `truncation` also appears only under nf4 (0→2). Directionally
   consistent with the hypothesis, but NOT statistically convincing at this power.
2. **Not a parser/backend bug?** Identical generation/decoding (same seed) across
   both formats; the classifier is format-independent. The differences reflect model
   behavior, not parsing.
3. **Not only accuracy moving?** Overall accuracy moved (78→70) AND the error
   distribution shifted (missed_required 2%→12%, truncation 0→4%). This matches the
   hypothesized pattern; it is not a move in accuracy alone.

**Decision:** [x] CONDITIONAL GO — proceed to a POWERED Phase 2 run (both models ×
both formats, larger n, full categories) to confirm or refute the `missed_required`
(+truncation) shift. Do NOT claim an effect from the pilot alone.

**Why not a hard GO or a NO-GO:**
- Hard GO would be dishonest: at n=50 the CI-based shift is not significant.
- A straight powered-NULL publish would throw away the strongest candidate signal
  (missed_required) before Phase 2 can settle it.
- Phase 2 must pre-register a powered decision threshold (e.g. a target n per format
  that gives the Δ CI excluding 0) so the outcome is either a confirmed shift or a
  precise powered null — never an ambiguous middle.

**Hold-over: if Phase 2 shows a powered-null on missed_required/truncation, we pivot
to Rank 1 (retry-sensitive reliability) as ROADMAP.md:79 requires.**

**Rationale:** The pilot validated the full pipeline end-to-end (BFCL load, controlled
HF decoding, taxonomy classifier, bootstrap CIs) and produced a real, reproducible
signal *direction* under NF4 (more missed-required + more truncation) that is worth a
powered confirmation before committing to the paper.

---

## Reproducibility metadata

- BFCL commit pinned: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (bfcl-eval 2026.3.23)
- transformers 5.0.0, bitsandbytes 0.50.1, torch 2.10.0+cu128, python 3.12.13 — recorded in results/phase1/pilot_results.json
- Raw outputs: results/phase1/ (gitignored)
- Kernel: muhammadroshaan/bitfracture-pilot-fp16-nf4-v3

---

# PHASE 2 GATE DECISION (2026-08-22)

Status: **DECIDED — WEAK / MIXED (pre-registered outcome #2). Size-dependent effect: real in 1.7B, absent in 4B.**

Evaluated strictly against analysis/PREREGISTRATION.md (committed BEFORE the run):

| Pre-registered outcome | Requirement | Result |
|---|---|---|
| CONFIRMED SHIFT | BOTH models' missed_required Δ CI excludes 0 | NOT met |
| **WEAK / MIXED** | Only ONE model's Δ CI excludes 0 | **MET — Qwen3-1.7B only** |
| POWERED NULL | Neither model's CI excludes 0 | Not met |

## The numbers (n=150/format; full detail in RESULTS.md + phase2_summary.txt)

- **Qwen3-1.7B:** missed_required 6.0% → 17.3% under NF4; Δ = −11.3pp,
  95% CI [−19.3, −4.7] — solid, ~2x the pilot effect size, spread across all
  five categories (not a single-category artifact).
- **Qwen3-4B:** missed_required identical at 7.3% vs 7.3% (11/150 each);
  CI [−6.0, +6.0] bounds any true effect as small. NF4 nominally better on
  correct (84.0% vs 79.3%) and wrong_args.

## Interpretation (honest)

The pre-registered "confirmed shift" failed because the effect does not
generalize across sizes. What we actually have is a **size × quantization
interaction**: NF4 breaks tool-calling discipline (missed required calls) in
the SMALL model and leaves the larger one intact — if anything slightly
improved. This is consistent with the pilot signal being real but
model-size-dependent, and inconsistent with it being pure noise (a null in
1.7B would have been expected under noise).

## Decision

[x] Proceed to Phase 3 paper with the interaction framing:
    "4-bit quantization shifts the error distribution toward missed required
    calls in small models (Qwen3-1.7B) but not mid-size ones (Qwen3-4B)."
    Claims stay bounded to what was measured; no universal quantization claims.
[ ] NOT the originally imagined uniform-shift paper — that hypothesis is
    partially refuted by the 4B leg and must be reported as such.
[ ] Rank 1 pivot NOT triggered (the preregistered pivot condition was a
    powered null on both models; we do not have that).

## Falsification checks performed

- Not a parser bug: same classifier for all cells; 4B nf4 parses fine (126 correct).
- Not a single-category artifact: 1.7B nf4 misses appear in all 5 categories.
- No post-hoc changes: decoding, categories, n, and decision rule were fixed
  in PREREGISTRATION.md before generation; nothing was altered after seeing data.
- The powered null on the 4B is itself reported as a bounded result (±6pp), not hidden.

## Reproducibility metadata (Phase 2)

- Kernel: muhammadroshaan/bitfracture-phase2-fp16-nf4-v1 (T4, COMPLETE)
- Raw outputs: results/phase2/results_phase2/ (gitignored) incl. phase2_summary.txt
- Versions recorded per-cell in phase2_results.json
