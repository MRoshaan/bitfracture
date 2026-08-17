# SLM-Agents Rank 3 - Quantization Error-Profile Study

Research project for the **SLM-Agents: 1st Workshop on SLMs for Agentic Systems**
(NeurIPS 2026, Paris). See [`ROADMAP.md`](./ROADMAP.md) for the full plan.

## One-line goal
Measure whether quantizing small tool-calling models (FP16 → NF4 → W4A16)
shifts the *distribution of error types*, even when overall accuracy looks stable.

## Layout
| Path | Purpose |
|---|---|
| `ROADMAP.md` | Full phased roadmap + architecture + gate rules |
| `kernel/` | Kaggle CLI script-kernel metadata + push/monitor/pull helpers |
| `run/` | `gen_responses.py` (HF, BFCL-format output) + eval helper |
| `pins/` | pinned `requirements.txt` + version table |
| `results/` | raw BFCL result/score JSON (gitignored) |
| `analysis/` | taxonomy breakdown, charts, `RESULTS.md`, `GATE.md` |
| `paper/` | `main.tex` (6pp NeurIPS template) + references |

## Quick start
1. Read `ROADMAP.md` (especially the Go/No-Go gate).
2. Phase 0 = Kaggle env check. See `kernel/` for the push command.
3. Run pilot (Phase 1) before any full run. Decide the gate before writing the paper.

## Honesty rules (non-negotiable)
- Version-pin every dependency; record run metadata with every result.
- No "new metric" claims — extend BFCL/ACBench work, do not re-claim it.
- Negative results only if powered, fair, and explained.
- Never commit `results/`, `*.pdf`, or any credential.
