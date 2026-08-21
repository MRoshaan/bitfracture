# BitFracture — Phase 2 Pre-Registration

**Written BEFORE running Phase 2.** This file is committed before any Phase 2
generation starts, so the success rule is fixed in advance (no peeking, no
moving goalposts).

Date written: 2026-08-22 · Before: kernel `bitfracture-phase2` run

---

## What we are testing (plain language)

Pilot observation (n=50/format, Qwen3-1.7B): the compressed model (NF4)
"forgot to call the tool" more often than the normal model (FP16):
1/50 vs 6/50 — and produced 2 truncated outputs vs 0.

Phase 2 repeats this with more data and a second model size to see whether
that pattern is real or was luck.

## Fixed design (decided in advance)

| Setting | Value |
|---|---|
| Models | Qwen/Qwen3-1.7B, Qwen/Qwen3-4B |
| Formats | fp16 (baseline), nf4 (bitsandbytes 4-bit) |
| Categories | simple_python, parallel, multiple, live_simple, live_multiple |
| Questions per category | 30 → n = 150 per model-format |
| Decoding | greedy, temp=0, seed=0, enable_thinking=False |
| Max new tokens | 192 |
| Primary endpoint | `missed_required` proportion difference (fp16 − nf4) |
| Secondary endpoints | `truncation` diff; per-category diffs |

## Decision rule (fixed now)

Compute the two-sample bootstrap 95% CI of Δ(fp16 − nf4) on `missed_required`,
per model, at n=150/format.

- **CONFIRMED SHIFT** — if for BOTH models the Δ CI excludes 0 with
  Δ < 0 (i.e., NF4 misses required calls more often), AND the direction of
  `truncation` does not contradict (NF4 ≥ FP16).
  → Proceed to paper writing (Phase 3). Claim: quantization shifts the error
  distribution toward missed required calls, replicated across two sizes.
- **WEAK / MIXED** — if only one model shows CI excluding 0.
  → Report honestly as partial replication; decide paper framing after seeing
  per-category breakdowns (do not pool silently to force significance).
- **POWERED NULL** — if neither model's Δ CI excludes 0 at n=150.
  → The pilot signal was noise. Do NOT write the taxonomy-shift paper.
  → Pivot to Rank 1 (retry-sensitive reliability) per ROADMAP.md gate rules;
    pilot infrastructure is reused there.

## Honesty notes

- No outcome-dependent changes to decoding, parsing, category mix, or n after
  results are seen. If a bug forces a change, document it in GATE.md.
- The classifier is format-independent and identical for both formats.
- All numbers land in results/phase2/*.json + a human-readable summary .txt,
  then analysis/RESULTS.md.
