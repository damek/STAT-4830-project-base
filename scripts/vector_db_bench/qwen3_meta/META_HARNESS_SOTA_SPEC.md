# Meta-Harness SOTA Spec

## Objective
- Build a Meta-Harness that gets `qwen/qwen3-coder-next` to a **state-of-the-art** vector-db-bench solution.
- The concrete target is **`4000+ QPS`** at valid recall.
- The comparison target is the current strong public baseline around **GLM-5.1 / ~4000 QPS**.
- Codex should do everything in its power **short of directly writing the worker's Rust solution itself**.

## Framing Shift
This is **not** the same experiment as the benchmark-faithful Meta-Harness setup.

The old framing was:
- fixed 50-tool-call worker episodes
- improve best-of-3 benchmark score under official constraints
- stay close to the original benchmark protocol

The new framing is:
- maximize Qwen's chance of reaching a SOTA solution
- allow a **long-horizon worker run**
- allow Codex to build a much stronger teacher / harness / tooling layer
- optimize for final achieved performance, not benchmark-faithful minimality

This should be treated as a **separate condition** from the official-style baseline.

## Success Criterion
Primary success criterion:
- achieve **`>= 4000 QPS`** with recall at or above the benchmark threshold

Secondary success criteria:
- produce a stable valid implementation, not a lucky one-off run
- reduce empty-response deaths and wasted search loops
- reach a clear ANN / IVF-style regime rather than a brute-force local optimum
- document the causal role of the harness in getting Qwen there

## What Changes Relative To The Earlier Meta-Harness
### Removed constraint
- the **50-tool-call budget is no longer binding** in this SOTA condition

### New optimization target
- no longer "beat the last local incumbent by a bit"
- now "get Qwen to a state-of-the-art solution"

### New role for Codex
Codex is no longer just making small harness improvements.

Codex is now responsible for creating the strongest possible environment around Qwen, including:
- strategy packages
- design specs
- implementation roadmaps
- stronger helper tools
- richer diagnostics
- stronger state summaries
- benchmark control logic
- online research when useful
- progress gating and recovery logic

## Core Principle
Codex still does **not** directly write the final worker Rust solution.

Instead, Codex may:
- design the worker's search program
- provide architectural guidance
- structure intermediate state
- add tools that compress evaluation and diagnosis
- add tools that summarize incumbent progress and failure modes
- add tools that help Qwen manage long-horizon search
- shape the stopping criteria and milestone logic

So the system becomes:
- **Qwen writes the solution**
- **Codex designs the optimization environment**

## Worker Condition
### Inner worker
- model: `qwen/qwen3-coder-next`
- task: optimize the vector-db-bench Rust implementation
- scaffold: still starts from the benchmark scaffold
- evaluation target: valid recall plus maximum QPS

### Worker horizon
- worker is no longer artificially constrained to 50 tool calls
- instead, use a configurable large budget or open-ended budget with stop conditions

Recommended initial setting:
- `200-500` tool calls per worker run
- or stop on saturation criteria rather than a fixed cap

## New Harness Objective
The harness must maximize the probability that Qwen discovers and stabilizes an ANN solution class capable of SOTA throughput.

That means the harness should explicitly bias toward:
- escaping exact-scan local optima
- early architectural pivots
- disciplined candidate evaluation
- preserving the best known valid design state
- avoiding useless diagnostic churn
- avoiding unsafe low-level detours before the algorithmic structure is right

## Allowed Harness Powers
In this SOTA condition, Codex may change:
- extra prompt/context layers
- `strategy.md`
- `design_spec.md`
- `implementation_plan.md`
- `incumbent_record.md`
- teacher notes and milestone files
- helper tools
- helper tool descriptions
- build/correctness/benchmark/profiling summarizers
- candidate checkpointing tools
- worker recovery and rollback tools
- benchmark scheduling policy
- stop conditions
- retry policy
- long-horizon orchestration logic
- online research usage

## Suggested Mandatory Teacher Artifacts
Every serious harness revision should be allowed, and likely encouraged, to provide:
- `src/strategy.md`
- `src/design_spec.md`
- `src/implementation_plan.md`
- `src/incumbent_record.md`
- `src/milestones.md`

These should tell Qwen:
- what architecture class to pursue
- what order to implement components in
- what not to waste time on yet
- what the current best known valid state is
- what counts as a meaningful milestone

## Architecture Direction
The harness should assume that SOTA performance requires getting Qwen into an ANN regime.

Practical target class:
- IVF / coarse clustering
- shortlist generation by centroid or bucket scoring
- exact reranking over candidates
- compact cluster-local memory layout
- high but controlled probe count
- low scan ratio
- only then distance-kernel tuning / SIMD refinement

The harness should stop behaving as if exact search is the primary target.

## Required Helper Tools
In this SOTA condition, the harness should support richer tools than the minimal official set.

Important tool classes:
1. `review_run_state`
- summarize best valid result, current branch, recent failures, next move

2. `checkpoint_candidate`
- run build + correctness + cheap benchmark + optional profiling
- store the best valid state
- expose rollback handle

3. `restore_best_candidate`
- recover the last known good valid code state

4. `assess_architecture`
- classify current implementation as brute-force / partial-top-k / IVF-like / cluster-probe / etc.
- recommend next structural move

5. `profile_summary`
- compress profiling outputs into actionable text
- avoid raw SVG inspection by default

6. `benchmark_policy`
- mediate when a full benchmark is worth running

7. `research_notes`
- expose distilled online research results or allow live research where appropriate

## Online Research
Codex should be allowed to use online research when useful.

The purpose is to help Codex:
- understand current SOTA benchmark solutions
- inspect public reports like GLM-5.1 / Claude / GPT-5.4 runs
- identify architectural patterns associated with 3000-4000+ QPS
- understand provider/runtime quirks

This should remain optional, but available.

## State Model
The system should now track at least three distinct state objects.

### 1. Harness incumbent
- best harness revision so far

### 2. Worker incumbent
- best known valid worker code state found inside the current long-horizon search

### 3. Search trace state
- milestone log, failure history, architecture phase, best checkpoint handles, saturation indicators

This is not a pure fresh-start-only setup inside a single long-horizon run.

The intended policy is:
- **fresh worker start for each new harness revision**
- **worker state carryover allowed across episodes within that revision**

So the harness is evaluated on its ability to guide Qwen from the scaffold, while still allowing long-horizon progress once a revision is in flight.

## New Loop Design
### Outer loop
Codex should:
1. inspect search trace, current worker incumbent, and benchmark history
2. revise the harness
3. relaunch or resume the worker under the stronger harness
4. evaluate whether Qwen is moving toward SOTA architecture and score levels

### Inner loop
Qwen should:
1. read the current strategy/design package
2. inspect current worker code and incumbent notes
3. implement the next architectural milestone
4. checkpoint aggressively
5. benchmark cheaply during search
6. use full benchmarks only when justified
7. stop only when saturated or explicitly finished

## Milestone Ladder
The harness should structure progress around milestones rather than blind iteration.

Suggested milestones:
1. valid exact baseline
2. exact baseline with partial top-k selection
3. stable `50-100` QPS regime
4. first valid ANN / shortlist prototype
5. valid `500+` QPS regime
6. valid `1000+` QPS regime
7. valid `2000+` QPS regime
8. valid `3000+` QPS regime
9. valid `4000+` QPS regime

The harness should orient Qwen around the next milestone, not generic improvement.

## Stopping Policy
Do not stop simply because a fixed tool budget is exhausted.

Stop when one of these is true:
- `>= 4000 QPS` valid solution achieved
- no meaningful architectural improvement across a long saturation window
- repeated bounded attempts fail to improve the current ANN architecture
- explicit user stop

## Evaluation Protocol For This Condition
Because this is now a SOTA-seeking condition rather than a benchmark-faithful one, report separately:
- best valid QPS achieved
- recall at that point
- total tool calls used
- wall-clock time
- number of checkpoints
- number of full benchmarks
- number of profiling runs
- outer-loop revisions used

This condition should **not** be presented as the official leaderboard protocol.
It should be presented as:
- a stronger teacher/harness system for driving a weaker model toward SOTA

## Immediate Redesign Implications
The current system should be redesigned in these ways.

### 1. Remove the 50-call assumption from the SOTA path
- create a separate long-horizon runtime mode
- do not overload the official baseline mode

### 2. Promote worker state carryover
- preserve the best valid implementation across long-horizon search
- allow checkpoint restore

### 3. Make structured teacher artifacts first-class
- seed multiple worker-readable strategy/design files
- not just a short extra user message

### 4. Require richer helper tools
- checkpointing
- rollback
- architecture review
- benchmark policy
- profiling compression

### 5. Use explicit milestone tracking
- treat 4000 QPS as the top milestone
- record progress toward it

### 6. Use provider/runtime controls aggressively
- route away from flaky providers
- add targeted retries for zero-completion failures
- prefer stable provider paths during long runs

## Suggested Implementation Plan
### Phase 1
- create a dedicated SOTA-harness spec and runtime path
- separate from benchmark-faithful `H0`

### Phase 2
- implement long-horizon worker runtime
- add checkpoint / restore support
- add milestone tracking

### Phase 3
- let Codex author richer teacher artifacts and helper tools
- allow online research

### Phase 4
- run long-horizon SOTA-seeking experiments
- measure whether Qwen can cross `4000 QPS`

## Summary
This spec changes the question from:
- "Can Codex slightly improve Qwen under the official 50-call benchmark protocol?"

To:
- "Can Codex build a harness that lets Qwen reach a state-of-the-art vector-db-bench solution?"

That is the new target condition.
