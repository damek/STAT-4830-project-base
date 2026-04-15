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

## State Model
Track two incumbents separately:

### Solution incumbent
- the best Rust implementation found so far
- carried forward into later Qwen rounds
- promoted only by measured benchmark improvement

### Harness incumbent
- the best harness revision found so far
- includes prompt construction, tool wrappers, scheduling policy, summaries, and diagnostics
- promoted only after a harness-evaluation block beats the current harness incumbent

This separation is required. Otherwise solution improvements and harness improvements get conflated.

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

### Condition D: Strong-Teacher Prompting
- same continuous loop structure as `B` or `C`
- a stronger model writes critique, summaries, or handoff guidance for Qwen
- this is a separate condition, not part of the first Meta-Harness experiment

Reason:
- otherwise we cannot tell whether gains came from the harness itself or from the stronger teacher model

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
- per-round `strategy.md` or equivalent strategy memo
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

In addition, each round may have a lightweight strategy artifact:
- `strategy.md`
- generated or refreshed before a Qwen round
- may be left unchanged if the outer loop decides no update is needed

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
- refresh the strategy memo every round
- update the actual harness every `5` rounds

Reason:
- reduces overfitting to one noisy round
- improves attribution
- lowers Codex and OpenRouter cost
- gives each harness revision a meaningful evaluation window

The outer loop can explicitly decide:
- `no_change` for the strategy memo on a given round
- `no_change` for the harness at a block boundary

### Optional exploratory schedule
- refresh the strategy memo every round
- update the actual harness every `3` rounds

Use this only after the `5`-round cadence is stable. It is more reactive, but noisier.

### Initial schedule
- rounds `1-5`: harness revision `rev_001`
- rounds `6-10`: candidate harness `rev_002`
- ...
- rounds `46-50`: candidate harness `rev_010`

Within each round:
- Codex may update `strategy.md`
- Qwen reads the current incumbent code plus current strategy memo

At each block boundary:
- Codex proposes at most `1` harness challenger in v1
- we do not run a population of `k` challengers yet
- this is a single-incumbent hill-climbing version of Meta-Harness

### Not recommended for v1
- changing the harness every single round
- branching multiple harness variants in parallel
- unlimited inner tool calls as the main comparison condition

These are interesting later, but they confound attribution and increase cost.

## Outer-Loop Inputs To Codex
For each harness update, Codex should be able to inspect:
- current harness code
- current incumbent code
- current `strategy.md`
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

For per-round strategy updates, Codex should produce a short memo with:
- incumbent QPS / recall
- current public target score
- current dominant bottleneck
- repeated failure modes to avoid
- 2-4 concrete next moves

## Candidate Harness Evaluation Rule
Each new harness revision should be judged on the next `5`-round block.

### Per-round strategy update rule
- the strategy memo may change every round
- this is cheap and intended to be responsive
- a strategy update does not become a new harness revision by itself

### Harness revision rule
- the harness revision changes only at block boundaries
- a new revision is a challenger against the current harness incumbent
- if the challenger does not beat the incumbent block, revert to the harness incumbent

### Suggested selection rule
Promote a harness revision if its block improves over the prior revision on a lexicographic objective:
1. valid round count
2. best strict-valid QPS
3. median strict-valid QPS across valid rounds
4. time-to-first-valid round
5. lower cost / lower wall-clock

If it does not improve, revert to the previous harness revision.

In v1, solution state is cumulative:
- the solution incumbent always carries forward
- the harness incumbent is accepted or rejected separately

If we need cleaner attribution later, add frozen-seed A/B blocks from the same solution incumbent.

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

### 4.5 Leaderboard-aware targeting
- expose the current incumbent score
- expose the current public target score or target band
- tell Qwen explicitly that the goal is to close the gap while preserving recall

Keep this short. The point is scale awareness, not a long leaderboard dump.

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

## Known Measurement Gap
Current continuous-loop promotion has an important evaluation mismatch:

- the inner official agent runtime defaults to `CPU_CORES=0-3`
- so its in-round benchmarks are CPU-pinned with `taskset`
- the current outer independent final evaluation path does not pin CPUs unless `--cpu-cores` is set explicitly

This means:
- the agent's own logged full benchmark and the outer promotion benchmark are not currently hardware-equivalent
- outer final-eval QPS can be materially higher than the in-round benchmark for the same final workspace

Observed example:
- round 11 agent full benchmark: `64.00 QPS` on `10000` queries
- round 11 outer final eval: `103.23 QPS` on `10000` queries

This is too large to treat as noise.

Implication:
- before the Meta-Harness experiment, align inner and outer evaluation settings
- most importantly, make CPU pinning consistent between the two paths

Until that is fixed, treat current promotion QPS as useful for search, but not as a perfectly fair comparison to the agent's own in-round metrics.

## Recommended Execution Plan

### Step 1
- finish the current `50`-round continuous naive baseline
- record the final incumbent, keep/discard rate, and major failure patterns

### Step 2
- align inner and outer evaluation settings
- fix the CPU pinning mismatch before using promotion QPS as a Meta-Harness comparison metric

### Step 3
- run a second `50`-round campaign with harness updates every `5` rounds

Within that campaign:
- update `strategy.md` every round
- allow harness revisions every `5` rounds
- include a short leaderboard target in the strategy context

### Step 3b
- optionally run a separate strong-teacher condition
- keep it separate from the first Meta-Harness run

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
