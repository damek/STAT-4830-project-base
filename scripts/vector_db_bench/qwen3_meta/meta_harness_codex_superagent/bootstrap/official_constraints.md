# Official Constraints

Keep these external benchmark rules fixed:

- start from the official blank scaffold
- use the official dataset
- use the official benchmark binary and protocol
- satisfy the official recall threshold of `0.95`
- preserve `CPU_CORES=0-3`

Do not try to win by changing the benchmark target itself.

Allowed changes:
- Rust solution code
- `Cargo.toml`
- local support scripts and notes inside this workspace
- your own local harness under `.meta_codex/`
