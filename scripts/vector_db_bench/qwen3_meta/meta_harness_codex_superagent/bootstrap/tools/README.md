# Workspace Tools

These scripts provide official-style benchmark actions inside the Codex workspace:

- `build_project`
- `run_correctness_test`
- `run_benchmark`
- `run_profiling`
- `get_status`

They are wrappers around the same benchmark helper logic used by the driver.

Examples:

```bash
.meta_codex/tools/build_project
.meta_codex/tools/run_correctness_test
.meta_codex/tools/run_benchmark
.meta_codex/tools/run_benchmark --full
.meta_codex/tools/run_profiling --duration 30
.meta_codex/tools/get_status
```

`finish` is intentionally absent because the outer driver owns cycle boundaries and promotion.
