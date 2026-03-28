# Week 12 Self-Critique (OODA) — History-Aware AirBench GEPA

## OBSERVE: Reading the Week 12 Report Critically

The Week 12 report stands on its own for **what the AirBench problem is** and **what we changed in code**, and it correctly anchors “new” claims in the **2026-03-22** history-aware commit rather than re-labeling older autoresearch or Week 8 work as fresh news.

We note three gaps:

1. **Outcome story vs. infrastructure story.** The report is clear that **the seed / base_recheck line remained best** in both committed runs. That honesty is good, but it reinforces the same tension flagged in [week8_self_critique.md](week8_self_critique.md): we are shipping **more capable proposal machinery** without yet showing **a better final program** than the strong baseline.

2. **The two runs are not a controlled comparison.** `history_aware_proposer` differs, but **GEPA trajectories, failure counts, and outer `total_metric_calls`** are not matched as in a randomized ablation. Wall-clock time (~1333 s vs. ~1087 s) therefore **cannot** be read as “history is faster”; it may reflect different candidate failures and early stopping behavior.

3. **Variance at the threshold still dominates interpretation.** Search-time scoring uses **`trials=1`** in [`run_config.json`](data/airbench/gepa_runs/20260322_175827/run_config.json) while verification uses **`verify_trials=3`**. Best verified means (~0.9405) sit **just above** the 0.94 target; differences of **0.0001** in mean accuracy between runs are compatible with **measurement noise**, not proof that one setting is better.

The report appropriately cites **no commits after 2026-03-22** in this repository snapshot, which avoids overstating recency for a Week 12 due date that falls later on the calendar.

## ORIENT: Analyzing Our Work

### Strengths

- **History is now first-class data, not an anecdote.** Persisted `round_*_history.json` and structured summaries (trajectory, best rows, derived lessons) make the proposer’s context **auditable**—a real improvement over opaque single-shot prompts.
- **Documentation reflects two execution paths** ([`docs/airbench_setup.md`](docs/airbench_setup.md)): GEPA vs. autoresearch/Gemini. That reduces confusion for anyone picking up the repo cold.
- **Evidence is tied to files.** Metrics and configs are cited from `data/airbench/gepa_runs/20260322_*`; readers can verify claims from `summary.json`, `eval_log.jsonl`, and `best_verified_eval.json`.

### Areas for improvement

- **We still have not run the ablation Week 8 asked for:** multi-agent **vs.** simpler proposer, or history **on** vs. **off**, under **matched evaluation budget and identical stopping rules**, with enough trials to see signal.
- **Proposal quality remains brittle:** committed logs still show **runtime failures**, **syntax errors**, and **catastrophically bad accuracies** on some GEPA candidates. History may help eventually, but these runs show **plenty of avoidable breakage** still reaching Modal.
- **Token and cost pressure increased slightly:** feeding JSON history into every team round **grows prompts**. Without measuring cost per successful target-meeting candidate, we do not know if this trade is net positive.
- **Single-trial search** still conflicts with a **hard 94%** gate when the incumbent sits near the boundary ([week8_self_critique.md](week8_self_critique.md), “Search-time evaluation is too noisy”).

### Critical risks and assumptions

We assume that **more context** (history) improves multi-agent coordination. These two runs **do not falsify** that, but they also **do not confirm** better *programs*—only that the harness runs.

We assume **Gemini** via LiteLLM is an acceptable substitute for the OpenAI-centric path described in Week 8. That may be fine, but it changes **latency, rate limits, and failure modes**, so cross-week comparisons of “how often quota killed the run” need explicit model tags.

## DECIDE: Concrete Next Actions (max 3; within a week)

1. **Run one matched ablation:** fixed seed, `max_metric_calls`, and reflection model; only toggle `--no-history-aware-proposer` vs. default on. Log **cost** (tokens or API spend if available) and **number of valid Modal evaluations** per arm.

2. **Raise search-time `trials` to at least 2** (or implement ranking that does not flip accept/reject on one noisy draw at 94%), as already urged in Week 8—so the optimizer is not chasing single-trial luck.

3. **Tighten pre-Modal gates** using the patterns already visible in `eval_log.jsonl` (syntax check, known bad accuracy bands) so fewer budget-eating remote runs are spent on doomed candidates.

## ACT: Resource Needs

Sustained **Gemini (or chosen reflection model) quota** for full team rounds **with** history JSON in every prompt, plus **Modal time** for repeated A100 evaluations if `trials` increases. A short **engineering sprint on local static checks** (CLI contract, syntax, import sanity) could reduce Modal spend more than adding another outer-loop iteration without those gates.
