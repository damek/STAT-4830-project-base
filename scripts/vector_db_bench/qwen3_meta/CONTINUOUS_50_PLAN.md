# Continuous 50-Round Harness Plan

## Objective
- Run `vector-db-bench` as a **continuous optimization loop**
- Start from the official empty scaffold in round 1
- Start from the **best incumbent so far** in rounds 2 through 50
- Keep the inner agent as close as possible to the official benchmark runtime

## Scope
This is a **separate condition** from the official benchmark baseline.

Official baseline:
- 3 independent attempts
- each starts from blank scaffold
- best of 3 wins

Continuous 50-round loop:
- 50 dependent rounds
- each round starts from the best solution found so far
- outer loop promotes or rejects each round based on measured benchmark results

## Inner Loop
Each round uses the upstream benchmark agent setup:
- scaffold layout from `third_party/vector-db-bench/skeleton/`
- system prompt from `agent/system_prompt.txt`
- opening user message: `Begin. Read the project files and start implementing.`
- tool schema from `agent/src/tools.rs`
- max tool calls: `50`
- same benchmark scoring and recall threshold

The inner loop should remain benchmark-faithful.

## Outer Loop
The outer harness is responsible for:
1. creating a fresh round workspace
2. seeding it from the incumbent
3. running the upstream agent
4. parsing the round result
5. promoting or rejecting the result
6. writing compact round summaries
7. repeating for 50 rounds

## Initialization Policy

### Round 1
- seed from official empty scaffold

### Rounds 2-50
- seed from the current incumbent snapshot

The incumbent snapshot must be a full buildable candidate, not just partial source files.

## Incumbent Snapshot
Canonical incumbent directory should include:
- `src/`
- `Cargo.toml`
- `Cargo.lock`

Recommended path:
- `data/vector_db_bench/qwen3_meta/continuous_50/incumbent/`

## Keep / Reject Rule
Promote the round result if:
- the round's **best strict-valid QPS** is greater than incumbent QPS
- recall remains valid
- anti-cheat remains valid

Otherwise:
- reject the round
- keep the previous incumbent unchanged

This is strict hill-climbing on measured benchmark performance.

## Round Outputs
Each round directory should contain:
- seeded workspace
- `agent_log.jsonl`
- `eval_log.json`
- benchmark outputs
- profiling outputs
- promoted candidate snapshot if applicable
- round summary

Recommended layout:

```text
data/vector_db_bench/qwen3_meta/continuous_50/
  incumbent/
  memory.md
  results.tsv
  summary.json
  round_001/
  round_002/
  ...
  round_050/
```

## Round Summary Fields
Per round, record:
- `round`
- `seed_source`
- `best_qps`
- `final_qps`
- `recall`
- `recall_passed`
- `tool_calls_used`
- `elapsed_secs`
- `promoted`
- `promotion_reason`
- `notes`

## Memory Carryover
Do **not** carry full transcripts between rounds.

Carry only compact research memory:
- current incumbent QPS / recall
- recent successful ideas
- recent failed ideas
- notable compile/runtime failures
- next likely moves

Recommended file:
- `data/vector_db_bench/qwen3_meta/continuous_50/memory.md`

Initial implementation can keep memory minimal.

## Prompt Policy
First version should minimize prompt drift:
- keep the same upstream system prompt
- keep the same upstream opening user message

Optional future extension:
- prepend a short research memo as an additional user message

But that should come after the seeded-incumbent loop works correctly.

## Benchmark Repo Choice
Use the **clean benchmark clone** on the Linux host, not the dirty submodule checkout.

Recommended host path:
- `/home/jonx/code/vector-db-bench-official-baseline`

Reason:
- the dirty checkout already contains scaffold modifications and is not baseline-safe

## Profiling Requirement
The inner loop uses `run_profiling`, so profiling must work.

Current status:
- fixed user-locally on the Linux host
- `perf`, `stackcollapse-perf.pl`, and `flamegraph.pl` are available
- FlameGraph Perl dependency was also fixed in user space

## Runtime Expectations
Observed from the first exact-format baseline run:
- approximately `26.6 minutes` for one full 50-call round

Estimated 50-round wall-clock cost if fully serial:
- roughly `22 hours`

This is long but feasible as a day-long experiment.

## Minimal Implementation Plan

### Step 1
Implement `run_continuous_loop.py`

Responsibilities:
- create round workspaces
- seed from incumbent
- invoke the upstream agent runtime
- parse `eval_log.json`
- compare against incumbent
- promote or reject
- append to `results.tsv`

### Step 2
Run a 3-round pilot

Success criteria:
- incumbent promotion works
- rejected rounds do not corrupt incumbent
- logs and artifacts are stored correctly

### Step 3
Scale to 50 rounds

## Non-Goals For V1
- no Codex outer-agent edits yet
- no tool-schema changes yet
- no custom prompt rewriting yet
- no benchmark rule changes

Version 1 should isolate the effect of:
- incumbent seeding
- outer keep/reject logic

## Future Extensions
After the seeded 50-round loop is stable:
- add Codex as an outer optimizer
- allow prompt adapter evolution
- allow tool-wrapper evolution
- add compact round-memory injection

## Immediate Next Step
Implement:
- `scripts/vector_db_bench/qwen3_meta/run_continuous_loop.py`

and validate it on:
- 3 continuous rounds
