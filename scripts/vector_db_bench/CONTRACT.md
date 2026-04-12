# Vector DB Bench — harness contract

This document matches `scripts/vector_db_bench/codex_cli_harness.py`.

## Protected (read-only for the agent)

These paths under the upstream **`skeleton/`** tree are **never** written by workers or `--apply-incumbent` targets for agent output. They define the HTTP/API surface and server entrypoint expected by the official benchmark client:

- `src/api.rs`
- `src/main.rs`

Do not edit them in rollouts. The harness still **copies** them from the repo skeleton on every eval (full tree copy + overlay editable files).

## Editable (mutable surface)

Any **regular file** under `skeleton/` that is:

- not in the protected list above,
- not under `target/`, `.git/`, `.idea/`, `.vscode/`,
- and matches one of:
  - `Cargo.toml` (at skeleton root),
  - `build.rs` (at skeleton root),
  - `*.rs` anywhere under `skeleton/` (except protected paths),

is **discovered** at startup and treated as part of the **incumbent snapshot**. Workers must return **full contents** for every path in the current incumbent, and may add **new** allowed paths (e.g. `src/ivf.rs`) via the `files` object.

Paths must be **relative POSIX paths** (use `/`). `..`, absolute paths, and protected paths are rejected.

## What the benchmark tests

Unchanged from [KCORES/vector-db-bench](https://github.com/KCORES/vector-db-bench): the **benchmark client** measures **QPS**, **recall** (vs precomputed ground truth), and **anti-cheat** on your **running server**. The harness does not replace that logic.
