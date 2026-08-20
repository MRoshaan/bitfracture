# Roadmap — Rank 3: Quantization Changes the Error Profile of Small Tool-Calling Models

Target: SLM-Agents @ NeurIPS 2026 (Paris) · Long paper (6pp) · NeurIPS template
Deadline: Aug 29, 2026 (23:59 AoE) — HARD floor. Submit by Aug 28.
Status: PLANNED (not started) · Last updated: 2026-08-18

====================================================================
0. GOAL (one line)
====================================================================
Measure whether quantizing small tool-calling models (FP16 -> NF4 -> W4A16)
shifts the *distribution of error types*, even when overall accuracy looks stable.

====================================================================
1. FIXED EXPERIMENT DESIGN  (do not expand without reviewing the gate)
====================================================================
Models:      Qwen3-1.7B + Qwen3-4B (Apache-2.0, no HF token needed)
Formats:     FP16 (baseline) | NF4 (bitsandbytes) | W4A16 (AutoGPTQ)
Backend:     HuggingFace transformers (NOT vLLM/sglang)
             reason: sglang needs SM80+; T4 is SM75. HF gives exact decoding control.
Decoding:    greedy, temp=0, fixed seed, enable_thinking=False (ALL variants)
Eval data:   BFCL single-turn core: simple_python, parallel, multiple,
             live_simple, live_multiple, relevance + format-sensitivity subset
Scoring:     bfcl evaluate (official checker) + our error-taxonomy classifier
Token budget: ZERO API spend. All local on Kaggle T4.

Eval pipeline note:
  - BFCL's local generation expects sglang/vllm; we generate responses with our
    own HF script in BFCL result-JSON format, then run `bfcl evaluate` on them.
  - This keeps decoding identical across formats (only quantization differs).

====================================================================
2. ERROR TAXONOMY (7 classes — our core contribution)
====================================================================
 1. missed required call       (tool was required but never called)
 2. unnecessary call           (called a tool when none needed)
 3. wrong tool                 (called a tool that isn't the right one)
 4. right tool, wrong args     (correct tool, bad arguments)
 5. malformed output           (not parseable as a function call)
 6. valid output, failed exec  (parses but execution state fails)
 7. truncation                 (output cut off by max tokens)
Extra recorded: calibration time, throughput (tok/s), latency, memory, output length.

GO criteria:
 - reproducible error-distribution shift with CI, e.g. NF4/W4A16 amplifies
   "right tool wrong args" or "malformed" relative to FP16.
NO-GO criteria (must check pilot before any full run):
 - differences within bootstrap CI (noise)            -> publish powered null OR pivot
 - shift explained by parser/backend bug              -> fix, not paper
 - only accuracy moves, taxonomy identical            -> reframe or pivot

====================================================================
3. PHASES (13 days, backwards from Aug 29)
====================================================================
PHASE 0  Setup (1 day, Aug 18-19)  — DONE 2026-08-19
  [x] configure ~/.kaggle access token  (chmod 600; NEVER git-commit)
  [x] create bitfracture/ workspace + ROADMAP
  [x] kernel-metadata.json + push helper (per brother's gist)
  [x] env-check kernel: load Qwen3-1.7B FP16+NF4 (pass), W4A16 blocked
      -> pins/versions.md records stack + W4A16 resolution options
PHASE 1  Pilot / Go-No-Go (2-3 days, Aug 19-22)  — DONE 2026-08-20 (CONDITIONAL GO)
  [x] 50 BFCL single-turn entries x Qwen3-1.7B x {fp16, nf4}  (5 core categories x10)
  [x] error-taxonomy breakdown + bootstrap intervals  -> results/phase1/
  [x] GATE DECISION recorded in analysis/GATE.md      -> CONDITIONAL GO (powered Phase 2)
  note: W4A16 dropped; gate on 1.7B only (4B is the Phase 2 replication leg).
        Pipeline fixed during pilot: BFCL flat NDJSON layout, Qwen3 JSON tool-call
        parser, bitsandbytes>=0.46.1 upgrade on the Kaggle base image.
PHASE 2  Full run (4-5 days, Aug 22-26)
  [ ] full core categories x 2 models (1.7B + 4B) x 2 formats (fp16, nf4)
  [ ] pre-register powered decision threshold for the missed_required/truncation shift
  [ ] collect throughput/latency/memory/calibration-time
  [ ] charts: error-type bars, error-shift matrix, size x format interaction
PHASE 3  Paper + Submit (Aug 26-28)
  [ ] write 6pp NeurIPS-template paper (overleaf template URL in CFP)
  [ ] double-blind pass (no names, no repo links with real identity)
  [ ] create OpenReview profile if not active; submit by Aug 28, buffer 1 day

====================================================================
4. GO/NO-GO GATE DECISION RULES (binding)
====================================================================
Continue to full run ONLY if Phase 1 shows a REPRODUCIBLE taxonomy shift
(bootstrap CI, same decoding all formats). If ranking is stable and errors
within variance -> either publish a PRECISE powered null result or pivot to
Rank 1 (Retry-Sensitive Reliability). Do not let sunk time force a weak paper.

====================================================================
5. HONESTY / EVIDENCE RULES (from neurips/neurips2026_slm_agentic_ideas.md)
====================================================================
 - Version-pin: transformers, tokenizer, chat template, bitsandbytes,
   auto_gptq, bfcl commit, seeds, parser. Table in pins/versions.md.
 - No "new metric" claims. BFCL already owns format sensitivity + ACBench
   already measured quantized small models. Our claim = taxonomy-level audit
   of error-profile SHIFT in Qwen3 family, controlled decoding.
 - No absolute claims: "we found no prior study applying THIS protocol" (with
   search boundary) instead of "no one has done this".
 - Negative result OK only if powered + fair + explains what it rules out.
 - Do NOT list as achieved metric anywhere until measured and recorded.
 - All numbers go into results/ .json + analysis/RESULTS.md with run metadata.

====================================================================
6. REPO LAYOUT (created during Phase 0)
====================================================================
bitfracture/   (standalone project, outside resume-builder)
  ROADMAP.md            this file
  README.md             overview + how to run
  kernel/               kernel-metadata.json + kaggle push/monitor/pull scripts
  run/                  gen_responses.py (HF, BFCL-format output) + eval helper
  pins/                 pinned-requirements.txt + versions table
  results/              raw result/score JSON per (model,format)   [gitignored]
  analysis/             taxonomy breakdown, charts, stats, RESULTS.md, GATE.md
  paper/                main.tex (6pp) + references.bib
  .gitignore            ignores results/, *.pdf, token files

====================================================================
7. KAGGLE COMMANDS (per brother's gist dd72b04)
====================================================================
kaggle config view                                    # confirm account
kaggle kernels push -p ./kernel-dir/ --accelerator NvidiaTeslaT4
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p ./local-out/
- GPU: T4 (SM75). Use fp16 (NOT bf16 — T4 lacks native bf16).
- Runtime cap ~12h/session. Write outputs ONLY under /kaggle/working/.
- Secrets: attach a PRIVATE dataset, never env vars, never hardcode in script.
- Avoid `pip install -e .` in-kernel; clone repo + sys.path.insert.
- Avoid flash-attention unless verified on Kaggle T4 image.

====================================================================
8. OPEN QUESTIONS (resolve during Phase 0)
====================================================================
 [ ] exact BFCL category name for format-sensitivity subset (check TEST_CATEGORIES.md)
 [ ] does Kaggle image have a working auto-gptq wheel for Python 3.11/3.12 + CUDA on T4
 [ ] confirm Qwen3-4B fits 16GB at FP16 with room for KV cache (yes ~9-10GB)
 [ ] decide .gitignore approach for ROADMAP.txt if we keep .txt (default: use .md)

====================================================================
9. DECISIONS LOCKED (from 2026-08-18 conversation)
====================================================================
Model family:     Qwen3 (1.7B + 4B)          [chosen, matches doc]
Third format:     W4A16 via AutoGPTQ          [chosen by user]
Paper format:     Long, 6 pages               [chosen by user]
Submission path:  KEEP pilot-first sequence   [phases above unchanged]
Kaggle token:     stored ~/.kaggle/access_token, chmod 600, never committed
