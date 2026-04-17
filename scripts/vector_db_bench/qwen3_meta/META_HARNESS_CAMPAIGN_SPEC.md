# Meta-Harness Campaign Spec

## Objective
- Build a long-running Codex-assisted optimization campaign for `qwen/qwen3-coder-next` on vector-db-bench.
- Target a valid **`4000+ QPS`** solution with recall at or above the benchmark threshold.
- Emphasize **persistent hill climbing over many iterations** rather than fresh-start evaluation blocks.

## Motivation
- The fresh-start harness condition is useful for measuring whether a revision helps Qwen from scratch.
- It is not the right structure for reproducing the kind of long-horizon optimization story reported for GLM-5.1.
- This campaign mode is meant to look more like an **autonomous engineering campaign**:
  - persistent code state
  - repeated experimentation
  - occasional strategy shifts
  - visible staircase progress over time

## High-Level Framing
This is a separate condition from both:
- the benchmark-faithful `H0` / best-of-3 setup
- the fresh-start SOTA harness setup

In campaign mode:
- the worker codebase is **persistent**
- Codex acts as an **intermittent supervisor**
- Qwen continues improving the same evolving implementation over a long horizon
- progress is judged by the **trajectory**, not just the final one-shot score

## Core Principle
- **Qwen writes and modifies the Rust solution**
- **Codex designs the optimization environment, campaign strategy, helper tools, and intervention logic**

Codex should do everything short of directly writing the worker Rust solution itself.

## Campaign State Model
Track three distinct persistent state layers.

### 1. Worker Mainline
- the current promoted worker code state
- always restorable
- should represent the best stable branch so far

### 2. Worker Experiments
- short-lived branch states for risky changes
- may temporarily violate recall or regress QPS
- only promoted into mainline if they beat the current mainline under the campaign's promotion rule

### 3. Harness State
- prompt layers
- teacher artifacts
- helper tools
- benchmark policy
- failure signatures
- research notes
- intervention history

## Worker Carryover Policy
This mode intentionally restores worker code carryover.

The policy is:
- **carry worker code across campaign iterations**
- **retain a promoted mainline snapshot**
- **allow temporary experiment branches**
- **allow Codex to update the harness around the same persistent worker codebase**

This is the defining difference from the fresh-start SOTA condition.

## Outer/Inner Loop Design
### Inner loop: Qwen
Qwen works on the persistent codebase and should:
1. read the current teacher package
2. inspect current mainline code and recent branch notes
3. implement the next structural move
4. checkpoint aggressively
5. benchmark and profile under campaign policy
6. promote or discard experiment branches
7. continue until a stop trigger or supervisor trigger fires

### Outer loop: Codex
Codex does not restart the worker from scratch.

Codex should:
1. inspect the campaign trace
2. identify whether Qwen is plateauing, thrashing, or making a structural transition
3. revise the harness around the same persistent worker codebase
4. update strategy, research notes, helper tools, and benchmark policy
5. send Qwen back into the same campaign with clearer guidance

## Supervisor Triggering
Codex should not intervene only once per run.

Codex should be triggered:
- every fixed interval such as `N` tool calls
- after each milestone promotion
- after repeated invalid results
- after repeated identical tool failures
- after a benchmark plateau
- after a major architectural transition

Recommended initial trigger rule:
- supervisor review every `75-150` worker tool calls
- immediate review after `k >= 3` repeated identical failure signatures
- immediate review after `m >= 5` benchmark attempts with no valid improvement

## Branch and Promotion Model
Campaign mode should make branching first-class.

Required operations:
1. `checkpoint_mainline`
- snapshot current promoted worker state

2. `fork_experiment`
- start a risky branch from mainline

3. `promote_experiment`
- replace mainline with the experiment if it satisfies promotion criteria

4. `discard_experiment`
- abandon the branch and return to mainline

5. `restore_mainline`
- hard return to the last promoted stable state

This is needed so the campaign can explore without losing accumulated progress.

## Promotion Rule
Do not promote every valid branch.

Suggested promotion policy:
- valid recall is required
- improvement must be meaningful relative to current mainline
- if a branch is architecturally superior but still slightly worse on QPS, it may be kept as a named experiment but should not automatically replace mainline

Initial default:
- promote when valid QPS exceeds current mainline by a minimum delta or when a new architectural phase is successfully entered

## Architecture Ladder
The campaign should explicitly model structural phases.

Recommended phase ladder:
1. `P0`: valid exact baseline
2. `P1`: exact baseline with partial top-k / cheap validation discipline
3. `P2`: first valid IVF / shortlist path
4. `P3`: contiguous list layout + online top-k
5. `P4`: large-`nlist` tuned IVF
6. `P5`: two-stage search / quantized coarse pass
7. `P6`: hierarchical routing / pruning / final leaderboard tuning

The objective is to move between phases, not spend the whole campaign polishing one local optimum.

## Required Teacher Artifacts
Campaign mode should keep these worker-facing files under `src/`:
- `src/strategy.md`
- `src/design_spec.md`
- `src/implementation_plan.md`
- `src/incumbent_record.md`
- `src/milestones.md`
- `src/research_notes.md`
- `src/campaign_journal.md`

Purpose:
- persistent memory
- current mainline description
- latest branch hypothesis
- next structural move
- benchmark policy
- failure-pattern reminders

## Required Helper Tools
Campaign mode should support at least:
1. `review_run_state`
- summarize current campaign state

2. `checkpoint_candidate`
- validate and optionally promote branch state

3. `restore_mainline`
- return to current stable promoted state

4. `fork_experiment`
- record a named experimental branch

5. `promote_experiment`
- promote the current branch when criteria are met

6. `assess_architecture`
- classify the current phase and recommend the next structural move

7. `benchmark_policy`
- decide whether to run smoke, quick, or full benchmark

8. `benchmark_history_summary`
- summarize recent benchmark trajectory and detect plateau

9. `profile_summary`
- compress profiling into actionable text

10. `research_notes`
- expose current distilled external ideas or supervisor guidance

11. `plan_next_step`
- turn current state into the next concrete implementation objective

## Anti-Thrash Controls
This mode must actively prevent wasting compute on identical failures.

Required controls:
- if the same tool fails `k` times with the same failure signature, block repeated execution until code or benchmark parameters change
- require a meaningful diff before re-running the same benchmark
- escalate to Codex supervisor after repeated identical failures
- forbid repeated full benchmarks during an invalid phase
- stop benchmarking entirely if correctness is still broken and no structural change has occurred

This is mandatory. Long campaigns without anti-thrash controls will burn compute uselessly.

## Benchmark Policy
Benchmarking should be staged.

Suggested classes:
1. `smoke`
- smallest cheap validation for liveness

2. `quick`
- short benchmark for directional signal

3. `full`
- only after benchmark policy approves it

Rules:
- no full benchmark before a valid checkpoint exists
- no repeated quick benchmark without a code or config change
- benchmark cadence should slow down during known-invalid phases and tighten near milestone confirmation

## Codex Research Role
Codex should be allowed to use online research when useful.

The purpose is to:
- study public strong VectorDBBench solutions
- extract architectural patterns
- keep `research_notes.md` current
- decide when the campaign should pivot from one phase to another

Codex should not dump generic advice. It should turn research into:
- design constraints
- concrete architecture ladders
- parameter ranges
- benchmark governance rules

## Logging Schema
Campaign mode should persist a structured trajectory log.

Each step or checkpoint should record:
- timestamp
- tool-call count
- wall-clock time
- campaign iteration id
- branch id
- architecture phase
- benchmark type
- QPS
- recall
- validity
- promotion/rollback action
- failure signature if any
- Codex intervention note if any

This should be machine-readable and append-only.

## Visualization Outputs
Campaign mode should generate artifacts that make hill climbing visible.

Required views:
1. QPS vs tool calls
2. QPS vs wall-clock time
3. recall violations marked explicitly
4. architecture phase transitions
5. branch promotions / rollbacks
6. Codex intervention points

The target pattern is a visible **staircase**:
- local tuning plateaus
- then structural jumps

## Success Criterion
Primary success:
- valid `>= 4000 QPS`

Secondary success:
- clear staircase campaign trajectory
- multiple structural transitions rather than one lucky jump
- evidence that Codex interventions materially changed the slope or direction of progress

## Comparison To Fresh-Start SOTA Mode
Fresh-start SOTA mode asks:
- can a harness revision help Qwen from the scaffold?

Campaign mode asks:
- can a persistent Codex-supervised Qwen campaign climb toward SOTA over a long horizon?

Both are useful, but they test different things.

## Recommendation
For the GLM-like experiment, use campaign mode as the primary path.

That means:
- persistent worker code
- periodic Codex supervision
- branch/promote/restore
- anti-thrash controls
- campaign logging and visualization

Do not judge it primarily by one revision's first episode. Judge it by the campaign trajectory.
