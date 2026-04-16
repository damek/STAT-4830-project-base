#!/usr/bin/env python3
"""Run fresh-start vector-db-bench Meta-Harness evaluations.

`h0` is the benchmark-faithful baseline:
- fresh blank scaffold every attempt
- official system prompt
- official opening user message
- official tool schema
- official 50-tool-call budget
- official `run_eval.sh` result collection

Later revisions may add extra context files, extra initial user messages, or
helper tools around the official worker, but the worker still starts fresh for
all attempts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from meta_harness_common import (
    DEFAULT_DOTENV_PATH,
    DEFAULT_REVISIONS_ROOT,
    AttemptOutcome,
    attempt_outcome_from_logs,
    attempt_row,
    build_results_writer,
    close_results_writer,
    ensure_blank_seed,
    flush_results_writer,
    load_dotenv,
    load_revision_config,
    prepare_workdir,
    run_revision_attempt,
    validate_revision_worker_contract,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_BLANK_SEED_SOURCE = REPO_ROOT / "third_party" / "vector-db-bench" / "skeleton"
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_runs"


def _default_run_root(revision_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_ROOT / f"{revision_id}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-id", type=str, default="h0", help="Harness revision id under meta_harness/revisions/.")
    parser.add_argument("--revisions-root", type=Path, default=DEFAULT_REVISIONS_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)

    parser.add_argument("--model-name", type=str, default="qwen3-coder-next")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-id", type=str, default="qwen/qwen3-coder-next")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--thinking-mode", type=str, default="openrouter-openai")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)

    parser.add_argument("--attempts", type=int, default=None, help="Override attempts_per_eval from the revision config.")
    parser.add_argument("--max-tool-calls", type=int, default=50)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)
    return parser.parse_args()


def _revision_payload(revision: object) -> dict:
    payload = asdict(revision)
    payload["revision_dir"] = str(payload["revision_dir"])
    payload["seed_files_dir"] = str(payload["seed_files_dir"]) if payload["seed_files_dir"] else None
    payload["helper_tools_module"] = str(payload["helper_tools_module"]) if payload["helper_tools_module"] else None
    return payload


def _attempt_summary_payload(outcome: AttemptOutcome) -> dict:
    return {
        "attempt": outcome.attempt_index,
        "revision_id": outcome.revision_id,
        "status": outcome.status,
        "process_returncode": outcome.process_returncode,
        "valid": outcome.valid,
        "qps": outcome.qps,
        "recall": outcome.recall,
        "recall_passed": outcome.recall_passed,
        "result_source": outcome.result_source,
        "tool_calls_used": outcome.tool_calls_used,
        "tool_calls_total": outcome.tool_calls_total,
        "best_qps": outcome.best_qps,
        "best_recall": outcome.best_recall,
        "last_qps": outcome.last_qps,
        "last_recall": outcome.last_recall,
        "elapsed_secs": outcome.elapsed_secs,
        "work_dir": outcome.work_dir,
        "notes": outcome.notes,
    }


def _summary_payload(*, revision_id: str, attempts_requested: int, outcomes: list[AttemptOutcome], run_root: Path) -> dict:
    valid_outcomes = [o for o in outcomes if o.valid]
    best_outcome = max(valid_outcomes, key=lambda o: o.qps, default=None)
    valid_qps = [o.qps for o in valid_outcomes]
    elapsed = [o.elapsed_secs for o in outcomes]
    return {
        "revision_id": revision_id,
        "attempts_requested": attempts_requested,
        "attempts_completed": len(outcomes),
        "valid_attempts": len(valid_outcomes),
        "best_qps": best_outcome.qps if best_outcome else 0.0,
        "best_attempt": best_outcome.attempt_index if best_outcome else None,
        "best_recall": best_outcome.recall if best_outcome else 0.0,
        "median_valid_qps": statistics.median(valid_qps) if valid_qps else 0.0,
        "mean_valid_qps": statistics.fmean(valid_qps) if valid_qps else 0.0,
        "mean_elapsed_secs": statistics.fmean(elapsed) if elapsed else 0.0,
        "run_root": str(run_root),
    }


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)

    revision = load_revision_config(args.revisions_root, args.revision_id)
    attempts = args.attempts if args.attempts is not None else revision.attempts_per_eval
    if attempts <= 0:
        raise SystemExit("--attempts must be positive")

    api_key = args.api_key or ""
    if not api_key:
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("API_KEY")
            or ""
        )
    if not api_key:
        raise SystemExit("API key missing. Pass --api-key or set OPENROUTER_API_KEY in the environment/.env.")

    bench_repo = args.bench_repo.resolve()
    blank_seed_source = args.blank_seed_source.resolve()
    data_dir = args.data_dir.resolve()
    run_root = (args.run_root.resolve() if args.run_root else _default_run_root(revision.revision_id))
    blank_seed_dir = run_root / "blank_seed"
    results_path = run_root / "results.tsv"
    summary_path = run_root / "summary.json"
    config_path = run_root / "run_config.json"

    if not bench_repo.is_dir():
        raise SystemExit(f"bench repo not found: {bench_repo}")
    if not blank_seed_source.is_dir():
        raise SystemExit(f"blank seed source not found: {blank_seed_source}")
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    run_root.mkdir(parents=True, exist_ok=True)
    ensure_blank_seed(blank_seed_dir, blank_seed_source)
    validate_revision_worker_contract(blank_seed_dir=blank_seed_dir, revision=revision)

    write_json(
        config_path,
        {
            "revision": _revision_payload(revision),
            "attempts": attempts,
            "bench_repo": str(bench_repo),
            "blank_seed_source": str(blank_seed_source),
            "blank_seed_dir": str(blank_seed_dir),
            "data_dir": str(data_dir),
            "model_name": args.model_name,
            "base_url": args.base_url,
            "model_id": args.model_id,
            "thinking_mode": args.thinking_mode,
            "reasoning_effort": args.reasoning_effort,
            "api_interval_ms": args.api_interval_ms,
            "max_tool_calls": args.max_tool_calls,
            "cpu_cores": args.cpu_cores,
            "run_root": str(run_root),
        },
    )

    writer = build_results_writer(results_path)
    outcomes: list[AttemptOutcome] = []
    try:
        for attempt_index in range(1, attempts + 1):
            attempt_dir = run_root / f"attempt_{attempt_index:03d}"
            work_dir = attempt_dir / "workdir"
            results_dir = attempt_dir / "results"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)

            prepare_workdir(
                bench_repo=bench_repo,
                blank_seed_dir=blank_seed_dir,
                work_dir=work_dir,
                revision=revision,
                max_tool_calls=args.max_tool_calls,
            )
            process = run_revision_attempt(
                revision=revision,
                bench_repo=bench_repo,
                work_dir=work_dir,
                results_dir=results_dir,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=api_key,
                model_id=args.model_id,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
                api_interval_ms=args.api_interval_ms,
                cpu_cores=args.cpu_cores,
                data_dir=data_dir,
                max_tool_calls=args.max_tool_calls,
            )
            outcome = attempt_outcome_from_logs(
                attempt_index=attempt_index,
                revision=revision,
                process=process,
                work_dir=work_dir,
                results_dir=results_dir,
            )
            outcomes.append(outcome)
            write_json(attempt_dir / "attempt_summary.json", _attempt_summary_payload(outcome))
            writer.writerow(attempt_row(outcome))
            flush_results_writer(writer)
            write_json(
                summary_path,
                _summary_payload(
                    revision_id=revision.revision_id,
                    attempts_requested=attempts,
                    outcomes=outcomes,
                    run_root=run_root,
                ),
            )
    finally:
        close_results_writer(writer)

    write_json(
        summary_path,
        _summary_payload(
            revision_id=revision.revision_id,
            attempts_requested=attempts,
            outcomes=outcomes,
            run_root=run_root,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
