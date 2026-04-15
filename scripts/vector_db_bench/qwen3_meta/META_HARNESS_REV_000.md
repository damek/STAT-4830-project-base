# Meta-Harness `rev_000`

## Purpose
`rev_000` is the neutral starting point for the fresh-start Meta-Harness experiment.

It does **not** try to improve Qwen yet.
It exists so we can:
- run a benchmark-faithful `H0` baseline
- load harness revisions from disk
- add harness changes later in a controlled way

## Current state

### `h0`
- exact official worker setup
- fresh blank scaffold every attempt
- official system prompt
- official opening user message
- official tool schema
- official `50`-tool-call budget
- no extra messages
- no helper tools

### `rev_000`
- neutral placeholder revision
- same behavior as `h0` today
- no pre-baked helper tools
- no strategy template
- no extra prompt/context injected by us

## Implemented files
- `scripts/vector_db_bench/qwen3_meta/run_meta_harness_eval.py`
- `scripts/vector_db_bench/qwen3_meta/meta_harness_common.py`
- `scripts/vector_db_bench/qwen3_meta/meta_harness/revisions/h0/revision.toml`
- `scripts/vector_db_bench/qwen3_meta/meta_harness/revisions/rev_000/revision.toml`

## What the runtime supports today
The runtime can already load per-revision metadata and evaluate a revision over fresh attempts.

Supported revision knobs:
- `extra_user_messages`
- `seed_files_dir`
- `added_helper_tools` metadata

Current behavior:
- `extra_user_messages`: supported
- `seed_files_dir`: supported
- `added_helper_tools`: declared in config, but actual helper-tool execution is **not implemented yet**

This is intentional. We are not hard-coding custom tools before Codex decides to add them.

## Evaluation model
Each revision is evaluated as:
- `3` fresh attempts by default
- blank scaffold each time
- official `run_eval.sh`
- official result collection logic
- best-of-3 summary written to `summary.json`

## Next step
After `H0` is measured, Codex can propose the first real harness change by editing:
- prompt/context files
- extra user messages
- seed files visible to the worker
- optional helper-tool definitions and runtime support
