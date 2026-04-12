# Vector DB Bench Program

You are optimizing a Rust implementation for `KCORES/vector-db-bench`.

Goal:
- Keep `recall >= 0.95`
- Keep anti-cheat passing
- Among valid candidates, maximize benchmark `qps`

Editable surface (see `scripts/vector_db_bench/CONTRACT.md`):
- All `*.rs` under `skeleton/src/` **except** the protected API/server files
- `skeleton/Cargo.toml` and optional `skeleton/build.rs`

Protected (never edit in rollouts):
- `skeleton/src/api.rs`, `skeleton/src/main.rs` — HTTP/API contract and server entry

Each worker returns **full contents for every current editable file** and may **add** new `src/*.rs` modules (must compile and stay recall-safe).

Fixed outside the skeleton:
- benchmark client crate, scoring, anti-cheat, dataset files

Priorities:
1. Recall >= threshold and anti-cheat passing
2. Throughput (QPS) on the official client workload
3. Sound ANN / systems choices (IVF, HNSW, quantization, parallelism, SIMD, layout) as appropriate
4. Release/profile tuning (`Cargo.toml` profiles, features)

Prefer coherent, incremental improvements round-to-round; large architectural jumps are allowed when recall-safe.
