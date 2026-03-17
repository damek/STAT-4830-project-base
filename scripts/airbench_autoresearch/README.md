# AirBench Autoresearch

Minimal `autoresearch`-style AirBench loop.

Files:
- `candidate.py`: the only mutable training program
- `program.md`: high-level optimization instructions
- `memory.md`: short rolling memory updated by the loop
- `loop_core.py`: shared keep/discard loop logic
- `run_candidate.py`: evaluate the current candidate
- `run_loop.py`: propose -> evaluate -> keep/revert loop
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
- The loop uses a cheap proxy evaluation during search, but any proxy improvement is promoted only after a strict confirmation run.
- The optional final strict evaluation is mostly redundant now and can usually be disabled for longer runs.
- The Modal-hosted loop writes artifacts to a Modal Volume first; use `::pull` to sync them back into the local workspace.
