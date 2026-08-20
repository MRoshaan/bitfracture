# BitFracture — Phase 1 Go/No-Go Gate

**Decision record.** Per ROADMAP.md:76-79, continue to the full run ONLY if the
Phase 1 pilot shows a REPRODUCIBLE taxonomy shift (bootstrap CI, same decoding
across formats). Populate the numbers below from the pilot results file, then
mark the decision.

Status: **PENDING (pilot not yet run)**

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

## Results (fill after pull of results/phase1/)

### fp16 (n = __)
| Class | count | proportion | 95% CI |
|---|---|---|---|
| correct | | | |
| missed_required | | | |
| unnecessary | | | |
| wrong_tool | | | |
| wrong_args | | | |
| malformed | | | |
| failed_exec | | | |
| truncation | | | |

### nf4 (n = __)
| Class | count | proportion | 95% CI |
|---|---|---|---|
| correct | | | |
| missed_required | | | |
| unnecessary | | | |
| wrong_tool | | | |
| wrong_args | | | |
| malformed | | | |
| failed_exec | | | |
| truncation | | | |

---

## Gate evaluation

1. **Reproducible shift?** Does any non-correct class move outside the bootstrapped
   CI between fp16 and nf4 (with disjoint intervals / pooled effect)?
2. **Not a parser/backend bug?** Confirm the shift is not an artifact of the Qwen3
   tool-call parser or a decode difference between the two formats.
3. **Not only accuracy moving?** Confirm the taxonomy distribution itself differs,
   not just the overall score.

**Decision:** [ ] GO — proceed to full run (both models × both formats)
                            [ ] NO-GO (powered null) — publish null explaining what it rules out
                            [ ] NO-GO (pivot to Rank 1) — retry-sensitive reliability

**Rationale:** <to be written from results>

---

## Reproducibility metadata

- BFCL commit pinned: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (bfcl-eval 2026.3.23)
- transformers / bitsandbytes versions: recorded in results/phase1/pilot_results.json
- Raw outputs: results/phase1/ (gitignored)
