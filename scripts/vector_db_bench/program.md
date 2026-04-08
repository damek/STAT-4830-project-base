# Vector DB Bench Program

You are optimizing a Rust implementation for `KCORES/vector-db-bench`.

Goal:
- Keep `recall >= 0.95`
- Keep anti-cheat passing
- Among valid candidates, maximize benchmark `qps`

Mutable surface only:
- `Cargo.toml`
- `src/db.rs`
- `src/distance.rs`

Fixed surface:
- HTTP/API contract
- benchmark crate
- scoring logic
- anti-cheat logic
- dataset files

Priorities:
1. exact brute-force correctness first
2. memory layout and data access
3. L2 distance kernel efficiency
4. top-k selection efficiency
5. query-time parallelism
6. release/profile tuning
7. ANN structures only after the exact baseline is strong and stable

Do not rewrite unrelated infrastructure. Prefer coherent systems changes that can plausibly improve throughput while preserving recall.
