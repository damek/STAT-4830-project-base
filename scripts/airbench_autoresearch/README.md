# AirBench Autoresearch

Minimal `autoresearch`-style AirBench loop.

Files:
- `candidate.py`: the only mutable training program
- `program.md`: high-level optimization instructions
- `memory.md`: short rolling memory updated by the loop
- `incumbent_record.json`: validated incumbent-of-record used to skip baseline revalidation
- `loop_core.py`: shared keep/discard loop logic
- `run_candidate.py`: evaluate the current candidate
- `run_loop.py`: propose -> evaluate -> keep/revert loop
- `codex_cli_harness.py`: local multi-agent harness that uses `codex exec` for coordinator/worker/reviewer roles
- `modal_autoresearch.py`: long-lived Modal-hosted loop plus artifact sync helpers

Typical commands:

```bash
conda activate airbench_gepa
python scripts/airbench_autoresearch/run_candidate.py --mode proxy --modal-show-output
```

```bash
conda activate airbench_gepa
python scripts/airbench_autoresearch/run_loop.py \
  --max-attempts 5 \
  --model gemini/gemini-3.1-flash-lite-preview \
  --modal-show-output
```

Codex CLI harness with one coordinator, three parallel workers, and one reviewer:

```bash
conda activate airbench_gepa
python scripts/airbench_autoresearch/codex_cli_harness.py \
  --rounds 2 \
  --workers-per-round 3 \
  --strict-top-k 1
```

If you want Codex to use a local OSS model provider:

```bash
conda activate airbench_gepa
python scripts/airbench_autoresearch/codex_cli_harness.py \
  --rounds 2 \
  --workers-per-round 3 \
  --strict-top-k 1 \
  --codex-oss \
  --codex-local-provider ollama
```

Background Modal-hosted loop on one long-lived worker:

```bash
modal run --detach scripts/airbench_autoresearch/modal_autoresearch.py::launch \
  --max-attempts 20
```

Wait for the detached/background run to finish, then sync its artifacts and apply the new incumbent:

```bash
modal run scripts/airbench_autoresearch/modal_autoresearch.py::pull \
  --run-name <timestamp> \
  --apply-incumbent
```

```bash
python scripts/airbench_autoresearch/plot_progress.py \
  --run-dir data/airbench/autoresearch_runs/<timestamp>
```

Notes:
- `candidate.py` is restored to the best incumbent after rejected attempts.
- Results are written under `data/airbench/autoresearch_runs/<timestamp>/`.
- The loop does not revalidate the starting incumbent on every run; it loads `incumbent_record.json` as the validated incumbent-of-record.
- The loop uses a cheap proxy evaluation during search, but any proxy improvement is promoted only after a stronger strict confirmation run.
- The optional final strict evaluation is mostly redundant now and can usually be disabled for longer runs.
- The Modal-hosted loop writes artifacts to a Modal Volume first; use `::pull` to sync them back into the local workspace.
- The Codex CLI harness keeps proposal generation local and deterministic scoring in Python; it writes run artifacts under `data/airbench/codex_cli_runs/<timestamp>/`.
- The Codex CLI harness does not rely on the API prompt loop directly; it shells out to `codex exec` for coordinator, worker, and reviewer steps.
