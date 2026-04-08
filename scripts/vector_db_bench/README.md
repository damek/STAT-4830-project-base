# Vector DB Bench Harness

Local Codex CLI harness for `KCORES/vector-db-bench`.

What it does:
- uses `codex exec` locally for coordinator, worker, and reviewer roles
- keeps the mutable surface to:
  - `skeleton/Cargo.toml`
  - `skeleton/src/db.rs`
  - `skeleton/src/distance.rs`
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

Notes:
- proposal generation is parallel; benchmarking stays sequential on one machine for cleaner measurements
- proxy evaluation uses a smaller query subset by default; strict uses the full query set unless `--strict-max-queries` is set
- run artifacts are written under `data/vector_db_bench/codex_cli_runs/<timestamp>/`
- `--apply-incumbent` writes the final winning mutable files back into the target repo's `skeleton/`
