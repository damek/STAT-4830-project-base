# Meta-Harness Adaptation Spec

## Objective
- Adapt the Meta-Harness idea to `vector-db-bench`
- Keep `qwen/qwen3-coder-next` fixed as the inner worker model
- Treat the **harness around the worker** as the editable outer-loop surface
- Measure whether `codex exec` can improve the worker by evolving:
  - prompting
  - strategy/context construction
  - tool descriptions
  - optional helper tools added around the official toolset
  - diagnostics and retry policy
  - benchmark scheduling policy
- Do **not** treat solution carryover as the primary experiment

## Core Framing

### Inner worker
- Model: `qwen/qwen3-coder-next`
- Task: optimize the official `vector-db-bench` Rust skeleton
- Runtime base: official scaffold, official system prompt, official opening user message, official tool schema
- Budget: `50` tool calls per worker attempt
- Start state: **fresh blank scaffold on every attempt**

### Outer optimizer
- Model: `codex exec`
- Task: inspect prior worker runs and improve the harness used by Qwen
- Search space: harness code, harness prompts, tool descriptions, wrappers, diagnostics, and attempt policy logic
- Objective: maximize Qwen's downstream benchmark performance under a fixed worker-attempt budget

## Key Decision
The primary Meta-Harness condition is **not**:
- "carry forward the best Rust solution and keep refining it"

The primary Meta-Harness condition **is**:
- "run the same fixed worker from scratch repeatedly while Codex improves the harness around it"

This is closer to the original Meta-Harness idea because gains are attributed to:
- better context
- better tooling
- better orchestration
- better diagnostics

rather than to accumulated solution state.

## What Counts As One Worker Attempt
One worker attempt is:
- a single fresh-start `vector-db-bench` run
- from the blank scaffold
- with the standard `50`-tool-call budget

So this is **not** plain one-shot text generation.

It is:
- one bounded iterative agent episode

That matches the official benchmark style.

## Comparison Conditions

### Condition A: Official Baseline
- `3` independent official attempts
- blank scaffold each time
- official scaffold
- official system prompt
- official opening user message
- official tool schema
- no custom outer-loop optimization

This is the benchmark-faithful reference condition.

### Condition B: Harness-Only Baseline (`H0`)
- worker still starts from blank scaffold every attempt
- same fixed worker model
- same task and scoring
- run under our harness implementation, but with no Codex evolution yet
- official tools remain unchanged

This establishes the baseline harness revision to improve from.

### Condition C: Meta-Harness Evolution
- worker still starts from blank scaffold every attempt
- same fixed worker model
- Codex periodically edits the harness
- each new harness revision is evaluated on a fresh block of worker attempts

### Condition D: Strong-Teacher Prompting
- separate optional condition
- a stronger model writes guidance, summaries, or critique for Qwen
- this is **not** part of the first Meta-Harness experiment

Reason:
- otherwise we cannot attribute gains cleanly to harness evolution

### Optional Separate Ablation
- incumbent-seeded continuous solution refinement

This may still be useful later, but it is no longer the primary Meta-Harness design.

## State Model
Track only one true incumbent in the primary Meta-Harness design:

### Harness incumbent
- the best harness revision found so far
- includes prompt construction, optional added helper tools, scheduling policy, summaries, retries, and diagnostics
- promoted only after a new harness revision beats it on a fresh evaluation block

Worker solution code is **not** carried between attempts in the primary condition.

This is intentional.

## High-Level Hypothesis
- Qwen performance is strongly affected by the harness around it
- the worker is currently losing efficiency through:
  - poor benchmark scheduling
  - weak strategy guidance
  - weak failure summarization
  - limited profiling interpretation
  - poor use of public target information
- therefore, an outer coding agent can improve the worker without changing the worker weights or carrying forward prior code

## Fixed Surface
These must remain fixed to preserve task comparability:
- benchmark dataset
- recall threshold
- anti-cheat
- final strict evaluation logic
- protected API contract
- worker model identity for a given experiment
- blank scaffold source
- official `50`-tool-call worker budget
- benchmark-faithful CPU pinning / runtime settings

## Mutable Surface
Codex may edit only the harness layer.

### Allowed
- extra strategy messages around the official opening prompt
- `strategy.md` or equivalent per-attempt strategy artifact
- context summarization and context selection logic
- leaderboard / target-score hints
- tool descriptions
- optional helper tools added around the official toolset
- benchmark scheduling policy
- profiling policy
- compile-error summarization
- retry policy
- empty-response handling
- logging and trace structuring
- attempt-selection policy for evaluation blocks

### Not allowed
- benchmark scorer
- benchmark client semantics
- anti-cheat behavior
- recall threshold
- dataset
- worker model/provider during a fixed experiment
- carry-forward of prior worker solution code in the primary Meta-Harness condition
- changing the semantics of the official benchmark tools in the baseline condition

## What "Harness Optimization" Means Here
The outer loop is not trying to write better Rust solutions directly.

It is trying to improve:
- how Qwen understands the task state
- how Qwen chooses which tools to use
- how expensive tools are scheduled
- how failures are surfaced
- how profiling is interpreted
- how global targets and constraints are communicated

In short:
- optimize **Qwen's search process**
- not the benchmark definition
- not the persistent solution state

## Optimization Target

### Primary objective
- maximize **best strict-valid QPS** achieved by Qwen under the current harness revision

### Secondary objectives
- improve valid-attempt count
- improve compile rate
- reduce wasted tool calls
- reduce repeated full-benchmark loops
- reduce time-to-first-valid candidate
- reduce time-to-first-strong candidate

### Cost-aware diagnostics
Track:
- OpenRouter token cost
- wall-clock time per attempt
- benchmark calls per valid attempt
- profiling calls per valid attempt

## Unit Of Evaluation
The artifact being optimized is a **harness revision**.

Each harness revision consists of:
- code changes to the harness layer
- a unique revision ID
- a note describing the intended improvement
- optional strategy artifacts used by the worker
- optional new helper tools added by Codex

Suggested layout:

```text
data/vector_db_bench/qwen3_meta/meta_harness/
  harness_revisions/
    rev_000/
    rev_001/
    rev_002/
    ...
  evaluations/
    rev_000/
      attempt_001/
      attempt_002/
      attempt_003/
    rev_001/
      attempt_001/
      attempt_002/
      attempt_003/
  harness_results.tsv
  summary.json
```

## Evaluation Block
Each harness revision is evaluated on a **fresh block of worker attempts**.

### Default block size
- `3` fresh attempts

Reason:
- directly mirrors the official "best of 3" benchmark framing
- lowers cost relative to larger blocks
- gives a cleaner bridge to the public leaderboard

### Optional larger block
- `5` fresh attempts

Use this only if the variance of the `3`-attempt block is too high.

## Outer-Loop Cadence

### Default
- one Codex harness proposal per evaluation cycle
- evaluate that harness on `3` fresh worker attempts
- accept or reject it
- then propose the next harness revision

This is a single-incumbent hill-climbing version of Meta-Harness.

### Not recommended for v1
- multiple parallel harness challengers per cycle
- solution carryover between worker attempts
- unlimited inner tool calls
- mixing strong-teacher prompting into the first Meta-Harness condition

## Outer-Loop Inputs To Codex
For each harness update, Codex should be able to inspect:
- current harness code
- prior harness revisions
- harness-level results ledger
- per-attempt summaries
- benchmark outputs
- profiling outputs
- compile failures
- agent execution traces

This should mirror the Meta-Harness idea:
- give the proposer raw prior artifacts through the filesystem
- do not compress everything into a tiny prompt summary

## Outer-Loop Prompt To Codex
Codex should be instructed to:
- inspect prior revisions and evaluation outcomes
- identify wasted tool usage or avoidable failures
- propose harness changes that improve Qwen's downstream performance
- stay within the allowed mutable surface
- explain the expected causal mechanism of improvement

Each proposal should include:
- hypothesis
- files changed
- expected benefit
- main risks

## Candidate Harness Evaluation Rule
Each new harness revision is judged against the current harness incumbent on a fresh evaluation block.

### Suggested selection rule
Promote a harness revision if it improves the incumbent on this lexicographic objective:
1. valid attempt count
2. best strict-valid QPS
3. median strict-valid QPS across valid attempts
4. time-to-first-valid attempt
5. lower cost / lower wall-clock

If it does not improve, revert to the previous harness incumbent.

## Initial High-Value Search Directions
These are the most promising early harness targets.

### 1. Benchmark scheduling
- discourage early `max_queries=0` full benchmarks
- default to smaller benchmark budgets until correctness is established
- escalate to full benchmark only after promising proxy signal

### 2. Strategy layer
- add a concise `strategy.md` or equivalent
- include:
  - current public target band
  - recall constraint
  - current dominant bottleneck
  - concrete next directions

### 3. Failure compression
- summarize compiler errors into short actionable diagnoses
- summarize benchmark timeout causes
- summarize repeated failure patterns

### 4. Profiling interpretation
- convert flamegraph/top-functions output into a short hotspot summary
- tell Qwen where time is actually being spent

### 5. Leaderboard-aware targeting
- expose the current public target score or target band
- tell Qwen explicitly that the goal is to close the gap while preserving recall

Keep this short. The point is scale awareness, not a long leaderboard dump.

### 6. Robustness
- handle empty LLM responses
- retry transient failures
- make sessions less brittle after tool errors

## Non-Goals For V1
- changing the benchmark rules
- changing Qwen itself
- model fine-tuning
- solution-incumbent carryover as the primary experiment
- multi-branch parallel harness evolution

V1 should isolate:
- fixed worker
- fresh-start attempts
- evolving harness only

## Operational Alignment Requirements
Before running the main Meta-Harness experiment, keep these aligned with the official benchmark setup:
- `CPU_CORES=0-3`
- `50` tool-call budget
- benchmark concurrency `4`
- worker attempts from blank scaffold
- official recall threshold `0.95`
- benchmark-faithful build / benchmark timeouts

## Recommended Execution Plan

### Step 1
- freeze an initial harness baseline `rev_000`
- run `3` fresh worker attempts
- record:
  - valid count
  - best strict-valid QPS
  - median valid QPS
  - major failure patterns

### Step 2
- let Codex inspect the raw artifacts from `rev_000`
- propose `rev_001`
- evaluate `rev_001` on `3` fresh worker attempts

### Step 3
- accept or reject `rev_001`
- repeat with `rev_002`, `rev_003`, ...

### Step 4
- after the harness-only experiment is stable, optionally run:
  - a strong-teacher prompting condition
  - an incumbent-seeded solution-carryover ablation

## Success Criterion
This adaptation is successful if the Meta-Harness-style outer loop produces a meaningfully stronger fresh-start Qwen worker than the fixed-harness baseline under comparable budget.

The cleanest win condition is:
- better best strict-valid QPS
- with equal or better valid-attempt count
- without changing the worker model or benchmark rules
