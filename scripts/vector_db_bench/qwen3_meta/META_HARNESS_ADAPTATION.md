# Meta-Harness Adaptation Spec

## Objective
- Adapt the Meta-Harness idea to `vector-db-bench`
- Keep `qwen/qwen3-coder-next` fixed as the inner model
- Use an outer `codex exec` loop to optimize the **harness around Qwen**, not the benchmark itself
- Measure whether harness evolution improves Qwen's achieved strict-valid QPS over a continuous incumbent-seeded run

## Core Framing

### Inner system
- Model: `qwen/qwen3-coder-next`
- Task: optimize the official `vector-db-bench` Rust skeleton
- Runtime: official scaffold, official system prompt, official opening user message, official tool schema
- Budget: `50` tool calls per round for the benchmark-faithful setting

### Outer system
- Model: `codex exec`
- Task: inspect prior runs and improve the harness used by Qwen
- Search space: harness code, harness prompts, tool descriptions, wrappers, diagnostics, and round-policy logic
- Objective: maximize Qwen's downstream benchmark performance under a fixed round budget

## High-Level Hypothesis
- Qwen performance is sensitive not only to model quality, but also to:
  - what context it receives
  - how benchmark/profiling tools are described and scheduled
  - what diagnostics are available after failures
  - how incumbent state is surfaced across rounds
- Therefore, an outer coding agent can improve Qwen's achieved QPS by improving the harness

## Comparison Conditions

### Condition A: Official Baseline
- `3` independent official attempts
- blank scaffold each time
- no continuous carryover
- no outer-loop harness edits

### Condition B: Continuous Naive Loop
- `50` dependent rounds
- round 1 starts from blank scaffold
- later rounds start from current incumbent
- same inner tool schema and same official-style runtime
- no outer harness edits

### Condition C: Continuous Meta-Harness Loop
- same `50`-round incumbent-seeded loop
- outer `codex exec` periodically edits the harness
- Qwen remains the fixed inner model

This is the main experimental comparison:
- `B` tests whether incumbent-seeded continuity helps
- `C` tests whether outer harness optimization helps beyond continuity alone

## Fixed Surface
These must remain fixed to preserve task comparability:
- benchmark dataset
- recall threshold
- anti-cheat
- final strict evaluation logic
- protected API contract
- inner model identity for a given experiment

## Mutable Surface
Codex may edit only the harness layer.

### Allowed
- extra incumbent-aware handoff messages
- context summarization and context selection logic
- round memory formatting
- tool descriptions
- tool wrappers
- helper tools
- benchmark scheduling policy
- profiling policy
- compile-error summarization
- round termination / retry heuristics
- result logging and trace structuring

### Not allowed
- benchmark scorer
- benchmark client semantics
- anti-cheat behavior
- recall threshold
- dataset
- model provider or model ID during a fixed experiment

## What "Harness Optimization" Means Here
The outer loop is not trying to write better Rust solutions directly.

It is trying to improve:
- how Qwen understands the current state
- how Qwen chooses which tools to use
- how expensive tools are scheduled
- how failures are surfaced
- how prior success is carried forward

In short:
- optimize **Qwen's search process**
- not the benchmark definition

## Optimization Target

### Primary objective
- maximize **best strict-valid QPS** achieved by Qwen over the campaign

### Secondary objectives
- improve compile rate
- improve valid benchmark rate
- reduce wasted tool calls
- reduce time-to-first-valid candidate
- reduce time-to-first-strong-candidate

### Cost-aware diagnostics
Track:
- OpenRouter token cost
- wall-clock time per round
- benchmark calls per valid candidate
- profiling calls per promoted incumbent

## Outer-Loop Unit Of Change
The artifact being optimized is a **harness revision**.

Each harness revision consists of:
- code changes to the harness layer
- a unique revision ID
- a note describing the intended improvement

Suggested layout:

```text
data/vector_db_bench/qwen3_meta/meta_harness_50/
  harness_revisions/
    rev_001/
    rev_002/
    ...
  incumbent/
  results.tsv
  summary.json
  round_001/
  ...
  round_050/
```

## Recommended Cadence

### Default
- update the harness every `5` rounds

Reason:
- reduces overfitting to one noisy round
- improves attribution
- lowers Codex and OpenRouter cost
- gives each harness revision a meaningful evaluation window

### Initial schedule
- rounds `1-5`: harness revision `rev_001`
- rounds `6-10`: `rev_002`
- ...
- rounds `46-50`: `rev_010`

### Not recommended for v1
- changing the harness every single round
- branching multiple harness variants in parallel
- unlimited inner tool calls as the main comparison condition

These are interesting later, but they confound attribution and increase cost.

## Outer-Loop Inputs To Codex
For each harness update, Codex should be able to inspect:
- current harness code
- current incumbent code
- `results.tsv`
- per-round summaries
- benchmark outputs
- profiling outputs
- compile failures
- agent execution traces

This should mirror the Meta-Harness idea:
- give the proposer raw prior artifacts through the filesystem
- do not compress everything into a short prompt summary

## Outer-Loop Prompt To Codex
Codex should be instructed to:
- inspect prior revisions and round outcomes
- identify wasted tool usage or avoidable failures
- propose harness changes that improve Qwen's downstream performance
- keep edits within the allowed mutable surface
- explain the expected causal mechanism of improvement

Each proposal should include:
- hypothesis
- files changed
- expected benefit
- main risks

## Candidate Harness Evaluation Rule
Each new harness revision should be judged on the next `5`-round block.

### Suggested selection rule
Promote a harness revision if its block improves over the prior revision on a lexicographic objective:
1. valid round count
2. best strict-valid QPS
3. median strict-valid QPS across valid rounds
4. time-to-first-valid round
5. lower cost / lower wall-clock

If it does not improve, revert to the previous harness revision.

## Initial High-Value Search Directions
These are the most promising early harness targets.

### 1. Benchmark scheduling
- discourage early `max_queries=0` full benchmarks
- default to smaller benchmark budgets until correctness is established
- escalate to full benchmark only after promising proxy signal

### 2. Failure compression
- summarize compiler errors into short actionable diagnoses
- summarize benchmark timeout causes
- summarize diff vs incumbent

### 3. Profiling interpretation
- convert flamegraph/top-functions output into a short hotspot summary
- tell Qwen where time is actually being spent

### 4. Incumbent-aware context
- strengthen the handoff message
- expose incumbent QPS, recall, and current bottlenecks more clearly

### 5. Early stopping
- stop bad rounds earlier when repeated full-benchmark failures or compile churn indicate low value

## Non-Goals For V1
- unlimited inner tool calls
- changing the benchmark rules
- changing Qwen itself
- model fine-tuning
- multi-branch parallel harness evolution

V1 should isolate:
- incumbent-seeded continuous optimization
- plus periodic outer harness improvements

## Recommended Execution Plan

### Step 1
- finish the current `3`-round pilot
- patch resume safety for partial rounds

### Step 2
- run a `50`-round continuous naive baseline overnight

### Step 3
- run a second `50`-round campaign with harness updates every `5` rounds

### Step 4
- compare:
  - best strict-valid QPS
  - valid rate
  - compile rate
  - cost
  - time-to-first-strong incumbent

## Success Criterion
This adaptation is successful if the Meta-Harness-style outer loop produces a meaningfully stronger Qwen campaign than the naive continuous loop under comparable budget.

The cleanest win condition is:
- higher best strict-valid QPS
- with equal or better validity
- and reasonable added cost
