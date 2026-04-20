# Meta-Harness Codex Superagent Spec

## Objective
- Build a continuously running Codex-driven optimization campaign for vector-db-bench.
- Target a valid **`4000+ QPS`** solution with recall at or above the official threshold.
- Use the same official benchmark scaffold, dataset, recall rule, and CPU pinning policy.
- Remove the dependency on Qwen/OpenRouter and let Codex act as the full research-and-implementation agent.

## Framing
This is a new condition, separate from:
- the benchmark-faithful `H0` / best-of-3 setup
- the Qwen fresh-start Meta-Harness setup
- the Codex-conductor / Qwen-actuator campaign setup

In this condition:
- **Codex is both the conductor and the actuator**
- Codex may directly modify the Rust solution
- Codex may directly modify its own local harness inside the persistent workspace
- Codex may use web search, shell commands, and self-authored helper scripts inside the workspace

The only hard constraint is that the external benchmark spec stays official.

## What Must Stay Official
The following should remain fixed:
- blank scaffold origin
- official dataset
- official benchmark binary
- official recall threshold
- official CPU pinning policy such as `CPU_CORES=0-3`
- the same server / benchmark protocol used by vector-db-bench

Codex may not "win" by changing the benchmark target itself.

## What Codex May Change
Codex may change:
- Rust solution code in `src/`
- `Cargo.toml`
- local support scripts inside the persistent workspace
- local strategy and research files
- its own benchmark policy files
- local automation and orchestration scripts
- local profiling and analysis helpers

Codex should be explicitly allowed to create and evolve its own local harness under a reserved directory such as `.meta_codex/`.

## Official-Style Workspace Tools
Codex should be given benchmark-style tools inside the persistent workspace:
- `build_project`
- `run_correctness_test`
- `run_benchmark`
- `run_profiling`
- `get_status`

These do not need to be native API tools. Workspace wrapper scripts are sufficient if they provide the same operational surface and structured output.

`finish` is not required because the outer driver already controls cycle boundaries and promotion.

## Core Principle
This mode is meant to look like an autonomous long-running engineering campaign.

That means:
- persistent workspace
- persistent mainline snapshot
- repeated Codex bursts
- evaluation after each burst
- promotion of the best valid state
- continuous research and strategy revision inside the same run

## Persistent State Model
Track at least these state objects:

### 1. Live Workspace
- the current editable state Codex is working on
- may be ahead of or worse than the promoted mainline

### 2. Promoted Mainline Snapshot
- the best known valid state so far
- always restorable
- used as the stable point of reference

### 3. Campaign Memory
- progress state
- benchmark history
- milestone history
- research notes
- campaign journal
- self-authored support scripts

## Codex Burst Model
There is no artificial tool-call budget inside a Codex burst.

Instead, use:
- a wall-clock timeout per burst
- evaluation after each burst
- optional auto-restore after repeated invalid cycles

Recommended initial operating point:
- `20-40` cycles
- `20-30` minutes per cycle
- stop early if goal is reached

## Workspace Layout
The persistent workspace should contain:
- the official blank scaffold-derived Rust project
- `.meta_codex/README.md`
- `.meta_codex/official_constraints.md`
- `.meta_codex/strategy.md`
- `.meta_codex/design_spec.md`
- `.meta_codex/research_notes.md`
- `.meta_codex/benchmark_policy.md`
- `.meta_codex/campaign_journal.md`
- `.meta_codex/incumbent_record.md`
- `.meta_codex/milestones.md`
- `.meta_codex/progress_state.json`
- `.meta_codex/mainline_snapshot/`
- `.meta_codex/tools/`

Codex may edit any of these local harness files as the campaign evolves.

## Self-Modifying Harness
Codex should be told explicitly:
- you may change the solution
- you may change your own local strategy documents
- you may create helper scripts under `.meta_codex/tools/`
- those files will persist into later cycles
- use that persistent harness to improve your own future productivity

This is the key difference from a one-shot coding run.

## Evaluation Policy
Each cycle should end with external evaluation by the driver:

1. build
2. correctness test
3. quick benchmark
4. full benchmark only when the quick result is promising
5. promote if valid and better than current mainline

The driver, not Codex, should decide final promotion.

## Promotion Rule
Promote when:
- build succeeds
- correctness passes
- chosen benchmark result is valid
- QPS exceeds the current promoted mainline

Keep the promoted mainline snapshot separate from the current live workspace.

## Anti-Thrash Controls
The driver should support:
- optional auto-restore to mainline after repeated invalid cycles
- persistent benchmark history
- cycle summaries and trajectory logging

Codex should also be instructed to keep a campaign journal and not blindly repeat identical failures.

## Research
Codex should be allowed to use online research when useful.

Valid research targets:
- public vector-db-bench reports
- ANN / IVF design ideas
- data-layout and probing strategies
- optimization techniques used by strong public solutions

Research remains optional, but available.

## Success Criterion
Primary success criterion:
- achieve **`>= 4000 QPS`** with valid recall

Secondary success criteria:
- produce a visible staircase of progress across cycles
- let Codex measurably improve its own local harness over time
- reduce wasted evaluation churn

## Intended Experiment
The main question this mode asks is:

- can a persistent self-modifying Codex superagent, running against the official benchmark spec, climb to state-of-the-art performance through repeated research, coding, benchmarking, and harness improvement?
