# NeurIPS 2026 Workshop Ideas: Adversarially Revised Portfolio

> Revised 2026-08-17 after an adversarial literature audit. This document keeps only distinct projects, ranks them by the order in which they should be pursued, and removes ideas whose central novelty claims are no longer credible.

## Decision in one page

This is now a ranked portfolio, not a list of ten independent submissions.

| Rank | Project | Original ideas | Priority | Decision |
|---:|---|---|---|---|
| 1 | Retry-Sensitive Reliability and Budgeted Scoring for Stateful Tool Agents | 1 + 7 merged | P0 | Pursue first |
| 2 | Resume Semantics and Policy Drift in Stateful Tool Agents | 2 | P1 | Pursue only after a small conformance pilot |
| 3 | Quantization Changes the Error Profile of Small Tool-Calling Models | 5 | P1 | Pilot immediately; continue only if the effect is real |
| 4 | Failure-Aware Memory Under a Fixed Prompt-Token Budget | 8 | P2 | Later, high competition and high leakage risk |
| 5 | When Coordination Stops Paying for Stateful Tool Use | 9 | P2 | Later, narrow the matrix aggressively |
| 6 | Does QD Diversity Buy Robustness Under Semantic Tool Shifts? | 3 | P3 | Only if the higher-ranked projects are blocked |

The following original ideas are removed from the pursuit list:

- Idea 4, agentic aptitude probes: kill the flagship framing. Existing checkpoint-probing work makes the novelty claim false, and the proposed labels are ill-posed.
- Idea 6, tool-choice circuits across scales: kill as a discovery paper. A current paper directly studies readable and steerable multi-tool choice across model families and scales.
- Idea 10, probe-calibrated abstention: kill as a standalone project. Hidden-state routing, abstention, and tool-use control are already active research areas. Its only defensible fragment belongs inside a routing or reliability study, not as a separate paper.
- Idea 1 is not retained separately because it is fully merged into Rank 1.
- Idea 7 is not retained separately because it is fully merged into Rank 1.

## Constraints and evidence policy

- Current date for this plan: 2026-08-17. The NeurIPS-wide 29 August date is a suggested workshop-contribution date, not a universal hard deadline. Individual CFPs control. See the [official NeurIPS dates](https://neurips.cc/Conferences/2026/Dates) and [workshop guidance](https://nips.cc/Conferences/2026/WorkshopsGuidance).
- Available compute: Kaggle 2xT4, 16 GB each, approximately 30 hours per week.
- API budget: USD 50 maximum. Do not spend API budget before a local pilot establishes an effect.
- Use version-pinned open models, benchmark commits, inference backends, user simulators, seeds, and parsers.
- Do not call a metric new when an existing benchmark or paper already implements the same idea.
- Do not make absolute absence claims. Use bounded wording such as “we found no prior study applying this protocol to ...” and list the search boundary.
- A negative result is acceptable only when the null result is powered, the comparison is fair, and the paper explains what the null rules out.

## Rank 1. Retry-Sensitive Reliability and Budgeted Scoring for Stateful Tool Agents

### Decision

Pursue this first. This is the best combination of feasibility, fit with existing work, and a defensible empirical question.

### Corrected contribution

Do not claim to introduce success-at-budget or a missing retry protocol. The current [tau2-bench CLI](https://github.com/sierra-research/tau2-bench/blob/main/docs/cli-reference.md) already exposes repeated trials, retries, resume, and cost-related outputs. The original [tau-bench paper](https://arxiv.org/abs/2406.12045) introduced repeated-trial reliability, and [ReliabilityBench](https://arxiv.org/abs/2601.06112) defines a broader reliability surface over repeated execution, perturbation, and fault tolerance. [Exgentic](https://github.com/Exgentic/exgentic) also provides an open evaluation harness with retry controls.

The contribution should be narrower:

> Under a fixed total resource budget, do retry semantics reorder rankings of stateful tool agents?

This is an empirical audit, not a new leaderboard or metric paper.

### Minimal experiment

- Models: Qwen3-1.7B, Qwen3-4B, and Qwen3-8B. Keep the model family fixed for the main result.
- Scaffold: one pinned agent implementation. Do not compare multiple scaffolds until the core result is stable.
- Environment: tau2-bench retail and telecom. Treat airline as secondary because the current repository discussion raises concerns about whether its transactions are a reliable proxy for general transactional capability. See [issue 224](https://github.com/sierra-research/tau2-bench/issues/224).
- User simulator: pin one version and model, report it, and do not substitute a local simulator without an ablation. The simulator is part of the evaluation protocol.
- Retry policies:
  1. Blind resampling with a fresh attempt.
  2. Retry after exposing the tool error.
  3. Fresh-context restart.
  4. Model-generated repair after an observable failure.
- Reset the environment before every independent attempt. Count every attempt, token, tool call, simulator call, latency interval, and state-changing action.
- Report pass@k, pass^k, expected cost to first success, success-versus-budget curves, rank-inversion matrices, and collateral state damage.
- Use bootstrap intervals and Kendall rank stability. A rank flip without uncertainty intervals is not evidence.

### What not to do

- Do not build a public leaderboard website.
- Do not sweep every open model listed in the old document.
- Do not add BFCL, AppWorld, two APIs, five models, four retry policies, and three domains in the first pass.
- Do not compare monetary API cost with local T4 cost without reporting GPU-seconds and wall-clock time separately.

### Go or no-go gate

Run a local pilot on 30 to 50 tasks. Continue only if retry policy changes produce either a meaningful rank movement or a meaningful cost-quality separation with uncertainty intervals. If rankings remain stable, publish a rank-stability audit only if the null result is precise enough to rule out practically important changes.

### Deliverable

Release a version-pinned evaluator and traces as an extension to the existing evaluation ecosystem. This should reuse the deterministic machinery in the [Tool-Calling Reliability Benchmark](https://github.com/aaliyan1230/tool-calling-reliability-benchmark) and [Agent Watchtower](https://github.com/aaliyan1230/agent-watchtower), rather than creating a second independent harness.

## Rank 2. Resume Semantics and Policy Drift in Stateful Tool Agents

### Decision

This is the most interesting systems project, but it is higher risk than Rank 1. Do not start it as a large benchmark until the conformance pilot works.

### Corrected contribution

The broad gap is no longer open. [ACRFence](https://arxiv.org/abs/2603.20625) studies semantic rollback attacks. [Crab](https://arxiv.org/abs/2604.28138) studies semantics-aware checkpoint and restore for agent sandboxes. [Experience Graphs](https://arxiv.org/abs/2606.29823) presents Trellis, which treats agent experience as durable database state. The current tau2 runner already has atomic save and resume through checkpoint.py in its [runner documentation](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/runner/README.md). [Resume Means Resume](https://arxiv.org/abs/2608.03836), submitted on 2026-08-04, defines machine-checkable recovery properties and reports violations in deployed workflow frameworks.

The viable question is:

> How does logical checkpoint restoration change executable tool policy when tool effects are stateful?

### Required semantics

Do not treat exact restore, replay, and summarization as interchangeable.

- Exact restore changes execution state.
- Summary restore changes the information available to the model.
- Replay changes the trajectory and can duplicate effects.
- A hybrid changes both information and trajectory.

### Minimal experiment

- Start with a small transactional environment, not AppWorld.
- Use immutable event logs, state hashes, idempotency keys, and explicit read-only versus effectful tools.
- Inject failure at a turn boundary.
- Compare logical prefix resume, summary resume, read-only replay, and summary plus read-only replay.
- Run temperature-zero fixed-seed trials first. Measure stochastic divergence separately.
- Measure prefix equivalence, next-tool distribution, argument equivalence, state equivalence, exactly-once violations, collateral damage, tokens, latency, and final task success.

### Kill criteria

Stop if the apparent improvement disappears after matching context tokens, if drift is no larger than ordinary sampling variance, or if the restore method cannot state how it prevents duplicate external effects. A summary that makes the agent more successful is not evidence of safe restoration.

Keep this experiment disjoint from Rank 4. Rank 2 studies within-task restoration. Rank 4 studies cross-task transfer from completed prior tasks.

## Rank 3. Quantization Changes the Error Profile of Small Tool-Calling Models

### Decision

Run a pilot immediately, but do not commit to the paper until the pilot shows a reproducible method-by-size interaction.

### Corrected contribution

The old claim that small models had not been measured is false. [ACBench](https://arxiv.org/abs/2505.19433) includes small models such as Gemma-2B and Qwen2.5 1.5B/3B and evaluates tool-use behavior under quantization. BFCL also contains format-sensitivity evaluation. The hypothesis should therefore be:

> Quantization changes the distribution of tool-call errors in small models, even when general instruction-following appears stable.

### Minimal pilot

- Models: Qwen3-1.7B and Qwen3-4B, or Qwen2.5-1.5B and 3B.
- Formats: FP16, BNB-NF4, and one maintained W4A16 implementation. Do not use an unpinned AutoAWQ stack.
- Pin the Transformers version, inference backend, tokenizer, chat template, calibration data, group size, activation dtype, decoding, and parser.
- Use BFCL core categories and the format-sensitivity subset. Use IFEval only as a control.
- Record calibration time, throughput, latency, output length, and memory use.

Separate:

- missed required call;
- unnecessary call;
- wrong tool;
- right tool with wrong arguments;
- malformed output;
- valid output with failed executable state;
- truncation.

### Go or no-go gate

Continue only if the pilot shows a reproducible error-profile change, not merely noise or a backend/parser difference. The paper title should not say “quantization kills tool calling” unless a large effect survives all controls.

## Rank 4. Failure-Aware Memory Under a Fixed Prompt-Token Budget

### Decision

Defer until the higher-ranked projects are stable. This is salvageable but crowded and unusually vulnerable to leakage.

### Corrected contribution

The architecture is not unclaimed. [MERIT](https://arxiv.org/abs/2608.05906) already combines failure classification with positive and negative episodic memory and correction retrieval. [DIVE](https://arxiv.org/abs/2608.12486), submitted on 2026-08-12, studies persistent skill evolution in frozen models. [Memory Reward Inflation](https://arxiv.org/abs/2608.00017) shows that faulty verifiers and similarity retrieval can reinforce incorrect memories.

The defensible question is:

> Does typed failure retrieval improve cross-task repair for frozen small agents under a fixed memory-token budget?

### Protocol

- Use deterministic tool and state-diff failure signatures for the primary result.
- Do not use a local LLM judge as the main verifier.
- Separate cold start, online stream, and cross-domain transfer.
- Do not insert the current task’s final reward, hidden expected action, task ID, or post-completion trajectory into memory before the task ends.
- Use retail as development and telecom as held-out transfer. Use airline only as an additional domain, not the sole transfer test.
- Compare no memory, recent-context memory, untyped retrieval, typed retrieval, and a Reflexion-style baseline under equal prompt-token budgets.
- Measure recovery after first failure, cross-task transfer, harmful retrieval, stale-memory rate, negative transfer, memory tokens, and total tokens.

Do not reimplement MERIT and DIVE from scratch in the remaining time. Reproduce only comparable baseline abstractions. The TTCL challenge uses AgentOdyssey, resets memory between games, caps runs at 500 steps, and ranks Qwen3-4B or Qwen3.5-4B variants. See the [TTCL call](https://ttcl-agents.github.io/). A tau2-only paper belongs in the general track, not the challenge track.

## Rank 5. When Coordination Stops Paying for Stateful Tool Use

### Decision

Defer. The question is useful, but the original matrix is too large and the novelty claim is too broad.

### Corrected contribution

Prior work already includes [Can Small Agents Collaborate to Beat a Single Large Language Model?](https://arxiv.org/abs/2601.11327), [Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460), [RCWT](https://arxiv.org/abs/2607.12216), and [SALE](https://arxiv.org/abs/2602.02751).

The viable question is:

> When does coordination stop paying for stateful tool use after accounting for all communication and execution resources?

### Minimal experiment

- Compare a Qwen3-1.7B coordinator-worker team against Qwen3-8B single-agent and, optionally, Qwen3-4B single-agent baselines.
- Use tau2-bench retail and telecom, not BFCL parallel categories as a proxy for multi-agent work.
- Use 30 to 50 tasks and three seeds.
- Normalize separately by total tokens, GPU-seconds, and wall-clock time.
- Report communication tokens, tool calls, latency, state collisions, collateral damage, and task success.
- Do not include an API arm until the local comparison is clear.

“Equal output tokens” is not “equal cost.” A crossover result that disappears after accounting for prompt tokens or coordination latency is not a real crossover.

## Rank 6. Does QD Diversity Buy Robustness Under Semantic Tool Shifts?

### Decision

Only pursue this if higher-ranked projects are blocked. It is a small robustness study, not a new harness-evolution framework.

### Corrected contribution

QD prompt and harness evolution already has substantial prior art, including [QD-LLM](https://arxiv.org/abs/2605.09781), [Diverse Prompts](https://arxiv.org/abs/2504.14367), [EvoLattice](https://arxiv.org/abs/2512.13857), [Agentic Harness Engineering](https://arxiv.org/abs/2604.25850), and [Meta-Harness](https://arxiv.org/abs/2603.28052). [Rethinking Evaluation of Harness Evolution](https://arxiv.org/abs/2607.12227) reports that harness evolution does not consistently beat compute-matched test-time scaling and often generalizes weakly.

The viable question is:

> Does archive diversity improve robustness beyond repeated, compute-matched search under semantic tool shifts?

### Minimal experiment

- One frozen model.
- 30 to 50 search tasks.
- Exogenous descriptors such as tool count, interaction depth, argument complexity, and statefulness. Do not define diversity from success labels.
- Semantic-preserving shifts: tool renaming, parameter reordering, description paraphrasing, equivalent schema formatting, and distractor tools.
- Compare MAP-Elites, random search, single-objective search, and repeated sampling with equal rollout budgets.
- Hold out entire shift families.

Do not transplant MCPEvol operators onto BFCL without showing that the resulting mutation preserves task semantics. [MCPEvol-Bench](https://arxiv.org/abs/2607.14642) studies evolving MCP environments, not arbitrary BFCL prompt edits.

## Removed from pursuit

### Original Idea 4: Agentic Aptitude Probes

The flagship novelty claim is no longer credible. [Fast and Accurate Probing of In-Training LLMs’ Downstream Performances](https://arxiv.org/abs/2604.01025) already predicts later downstream behavior from intermediate representations. [Linear Probes Detect Task Format, Not Reasoning Mode](https://arxiv.org/abs/2606.02907) directly warns that probe accuracy can be format leakage. The proposed repeated final-instruct score labels are also not a valid checkpoint-level learning target. Revisit only as a tightly controlled study with multiple post-training descendants from the same base checkpoint.

### Original Idea 6: Tool-Choice Circuits Across Scales

Kill as a discovery paper. [Tool Calling is Linearly Readable and Steerable in Language Models](https://arxiv.org/abs/2605.07990) already studies tool identity readability and steering across multiple families and scales. Raw hidden-state cosine and cross-scale patching are not well-defined without learned alignment. A replication showing non-portability could be useful, but it is not a priority project.

### Original Idea 10: Probe-Calibrated Abstention

Kill as a standalone project. [ASA](https://arxiv.org/abs/2602.04935), [To Call or Not to Call](https://arxiv.org/abs/2605.00737), [AgentAbstain](https://arxiv.org/abs/2607.10059), [Multi-Head Latent Control](https://arxiv.org/abs/2607.14277), and [Doomed from the Start](https://arxiv.org/abs/2607.06503) already cover hidden-state tool-use control, abstention, routing, or early failure prediction. The only defensible fragment is an ablation inside Rank 1 or an independently justified routing study.

## Citation and data corrections

These corrections are mandatory before the document is used externally.

- Replace the malformed Crab citation with [2604.28138](https://arxiv.org/abs/2604.28138).
- Replace the malformed HyFunc citation with [2602.13665](https://arxiv.org/abs/2602.13665).
- Replace the malformed early-abstention citation with [2502.09054](https://arxiv.org/abs/2502.09054).
- Cite [2506.07982](https://arxiv.org/abs/2506.07982) as the tau2-bench paper. Do not call that paper the tau3-bench paper.
- Cite [2606.29823](https://arxiv.org/abs/2606.29823) under its paper title, *Experience Graphs: The Data Foundation for Self-Improving Agents*, and describe Trellis as the system proposed there.
- Pin a BFCL repository commit. The old count of 4,684 entries is stale. Use the exact files and counts in the experiment rather than a floating headline.
- Treat AppWorld licensing as data-dependent. Its protected task and API information has additional redistribution conditions beyond a casual Apache-2.0 summary. See the [AppWorld repository](https://github.com/StonyBrookNLP/appworld).
- Verify the exact Qwen3.5 checkpoint and runtime before including it in a text-only T4 model pool. The current [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) describes a multimodal model.
- Recalculate API costs from the exact provider, model, region, batch mode, and token counts. The old document mixes Gemini 2.5 and Gemini 3.5 pricing, lists Anthropic pricing despite a Gemini or Bedrock-only constraint, and uses unsupported DeepSeek V3.2 estimates. Check [Google pricing](https://ai.google.dev/gemini-api/docs/pricing), [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/), and [Anthropic pricing](https://www.anthropic.com/claude/sonnet).

## Execution order

1. Reuse the TCRB and Agent Watchtower infrastructure for the Rank 1 local pilot.
2. Do not start a second paper until the Rank 1 pilot has a measurable result or a clearly powered null result.
3. Run the Rank 3 quantization pilot as a quick independent go or no-go test if it does not interfere with Rank 1.
4. Start Rank 2 only if you intentionally choose the systems-risk path and can implement transactional restore semantics before scaling up.
5. Treat Ranks 4 to 6 as later projects, not simultaneous submissions.

The current recommendation is therefore simple: pursue Rank 1 first, keep Rank 2 as the serious backup, and use Rank 3 only as a gated pilot. Do not revive the removed ideas under their old novelty claims.
