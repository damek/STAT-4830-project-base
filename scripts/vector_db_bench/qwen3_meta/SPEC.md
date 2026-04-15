# Qwen VectorDBBench Meta Spec

## Objective
- Establish a **benchmark-faithful baseline** for `qwen/qwen3-coder-next` on `vector-db-bench`
- Use the **exact upstream scaffold and agent format** before introducing any custom harness changes
- After the baseline is stable, evaluate whether an outer `codex exec` loop can improve the inner agent by evolving tools, policies, and context handling

## Benchmark-Faithful Baseline

### Shared Upstream Assets
- Scaffold: `third_party/vector-db-bench/skeleton/`
- System prompt: [agent/system_prompt.txt](/Users/jonathanxu/Documents/Code/STAT-4830-AlphaGrad-project/third_party/vector-db-bench/agent/system_prompt.txt)
- Opening user message: `Begin. Read the project files and start implementing.`
- Tool schema: the exact tool definitions in [agent/src/tools.rs](/Users/jonathanxu/Documents/Code/STAT-4830-AlphaGrad-project/third_party/vector-db-bench/agent/src/tools.rs)
- Agent runtime: the upstream `vector-db-agent` binary and [scripts/run_eval.sh](/Users/jonathanxu/Documents/Code/STAT-4830-AlphaGrad-project/third_party/vector-db-bench/scripts/run_eval.sh)

### Standard Protocol
- `50` tool calls per attempt
- `3` independent attempts (`turn-1`, `turn-2`, `turn-3`)
- Final score = best strict-valid QPS across the three attempts
- Validity requires:
  - build success
  - benchmark run success
  - anti-cheat pass
  - recall `>= 0.95`

### Editable Surface
The upstream prompt emphasizes:
- `src/db.rs`
- `src/distance.rs`
- `Cargo.toml`

The runtime technically allows writes to writable `src/*` except read-only files, matching the official benchmark behavior.

### Read-Only Surface
- `src/main.rs`
- `src/api.rs`
- benchmark client
- scorer / anti-cheat

## Phases

### Phase 0: Exact Baseline
Run `qwen/qwen3-coder-next` through the upstream benchmark runtime with:
- the shared scaffold
- the shared system prompt
- the shared opening user message
- the shared tool set
- the shared `50`-tool-call budget
- `3` independent attempts

This phase is the reference point for every later comparison.

### Phase 1: Fixed-Tools Reimplementation
Reproduce the same benchmark conditions in our own harness while preserving:
- the same scaffold
- the same prompt structure
- the same tool schema
- the same budget and scoring rules

This phase exists only to verify we can faithfully mirror the benchmark runtime.

### Phase 2: Outer-Loop Evolution
Only after the exact baseline is working smoothly, allow `codex exec` to improve the inner agent environment between attempts.

Allowed outer-loop edits:
- tool implementations
- tool wrappers
- context summarization
- retry policy
- prompt adapters around the same baseline prompt
- logging / diagnostics

Not allowed:
- changing recall threshold
- changing anti-cheat
- changing benchmark scoring
- changing the protected API contract

## Conditions To Compare

### A. Official-Format Baseline
- Upstream scaffold
- Upstream prompt
- Upstream tools
- No custom outer-loop optimization

### B. Fixed Improved Harness
- Same benchmark task and budget
- Same prompt/task definition
- Our own runtime implementation, but behaviorally aligned to the official one

### C. Evolved Harness
- Same inner model
- Same benchmark task and budget
- Outer `codex exec` loop improves the inner runtime environment over time

## Metrics

### Primary
- best strict-valid QPS
- compile rate
- valid rate

### Secondary
- time-to-first-valid candidate
- number of benchmark evaluations before first valid candidate
- wall-clock time per attempt
- average strict QPS among valid attempts

### Meta
- tool usage frequency
- benchmark/profiling call counts
- outer-loop edits accepted vs rejected
- marginal gain after each accepted outer-loop change

## Keep/Revert Rule For Outer Changes
- Keep an outer-loop change only if it improves at least one of:
  - best strict-valid QPS
  - valid rate
  - compile rate
  - time-to-first-valid candidate
- Otherwise revert

## Immediate Next Step
1. Run a single official-format `turn-1` baseline with `qwen/qwen3-coder-next`
2. If the runtime works end-to-end, run the full `3` attempts
3. Only after that begin custom harness evolution
