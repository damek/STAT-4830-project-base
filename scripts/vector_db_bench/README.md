# Vector DB Bench Harness

Local Codex CLI harness for `KCORES/vector-db-bench`.

What it does:
- uses `codex exec` locally for coordinator, worker, and reviewer roles
- **Editable surface:** all `skeleton/**/*.rs` except `src/api.rs` and `src/main.rs`, plus `Cargo.toml` and `build.rs` (discovered from disk; see `CONTRACT.md`)
- **Protected:** `src/api.rs`, `src/main.rs` (HTTP contract + server shell)
- builds and benchmarks candidates locally against the repo's official benchmark binary
- ranks candidates by:
  1. validity (`recall >= 0.95` and anti-cheat pass)
  2. higher `qps`

Requirements:
- local clone of `KCORES/vector-db-bench`
- prepared benchmark data
- `codex` on PATH
- Rust toolchain installed

Typical command:

```bash
python scripts/vector_db_bench/codex_cli_harness.py \
  --bench-repo /path/to/vector-db-bench \
  --rounds 2 \
  --workers-per-round 3 \
  --strict-top-k 1
```

If your benchmark data files are not at the default paths, pass them explicitly:

```bash
python scripts/vector_db_bench/codex_cli_harness.py \
  --bench-repo /path/to/vector-db-bench \
  --base-vectors /path/to/base_vectors.json \
  --query-vectors /path/to/query_vectors.json \
  --ground-truth /path/to/ground_truth.json
```

### Modal (remote CPU eval)

`modal_vdb_eval.py` runs **`cargo build` + `vector-db-benchmark` on Modal** while you keep proposal generation local (e.g. Codex + `--oss`).

1. Install and log in: [Modal](https://modal.com) CLI (`modal token set …`).
2. Prepare `vector-db-bench` data (`base_vectors.json`, `query_vectors.json`, `ground_truth.json`) per upstream `USAGE.md`.
3. Run:

```bash
export VECTOR_DB_BENCH_ROOT=/abs/path/to/vector-db-bench
export VECTOR_DB_BENCH_DATA=/abs/path/to/dir/with/json   # or omit to use $VECTOR_DB_BENCH_ROOT/data

modal run scripts/vector_db_bench/modal_vdb_eval.py::main \
  --candidate-json /path/to/candidate_files.json \
  --max-queries 500
```

`candidate_files.json` is a single JSON object: `{ "Cargo.toml": "...", "src/db.rs": "...", ... }`.

The **multi-agent Codex harness** does not call Modal yet; wire `remote_evaluate.remote(...)` in if you want strict evals on Modal from that loop.

Notes:
- proposal generation is parallel; benchmarking stays sequential on one machine for cleaner measurements
- proxy evaluation uses a smaller query subset by default; strict uses the full query set unless `--strict-max-queries` is set
- if upstream data conversion produced `base_vectors_*.json` shards, the harness automatically merges them into one benchmark input file under the run directory
- run artifacts are written under `data/vector_db_bench/codex_cli_runs/<timestamp>/`
- `--apply-incumbent` writes the final winning mutable files back into the target repo's `skeleton/`

### Codex Teacher Rollouts For SFT

`codex_sft_rollouts.py` uses the same worker prompts as the RL pipeline, but
asks `codex exec` to act as a teacher model. Accepted outputs are written as
chat-format JSONL for `modal_sft_train.py`.

Typical workflow:

```bash
python scripts/vector_db_bench/generate_rl_prompts.py \
  --bench-repo third_party/vector-db-bench

python scripts/vector_db_bench/codex_sft_rollouts.py \
  --bench-repo third_party/vector-db-bench \
  --prompts-path data/vector_db_bench/rl_prompts.jsonl \
  --attempts-per-prompt 1 \
  --only-valid \
  --output data/vector_db_bench/codex_teacher_sft.jsonl
```

Useful flags:
- `--eval-phase proxy|strict`: choose which benchmark split filters teacher rollouts
- `--only-valid` / `--no-only-valid`: keep only fully valid candidates, or any build+runtime-clean candidate
- `--min-qps N`: reject weak but technically valid candidates
- `--max-prompts N`: smoke test on a subset before a full teacher pass

This is the cleanest way to bootstrap SFT from a stronger external coder before
running GRPO on the open-source model.
