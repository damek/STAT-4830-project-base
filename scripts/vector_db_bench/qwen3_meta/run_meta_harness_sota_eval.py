#!/usr/bin/env python3
"""Run long-horizon SOTA-seeking Meta-Harness evaluations.

This is a separate condition from the benchmark-faithful fresh-start best-of-3
path. It allows:
- long-horizon worker episodes
- worker incumbent carryover across episodes
- structured teacher artifacts under src/
- helper tools and recovery logic

The objective is to push Qwen toward a valid 4000+ QPS solution rather than to
stay close to the official 50-tool-call protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from meta_harness_common import (
    DEFAULT_DOTENV_PATH,
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
DEFAULT_SOTA_REVISIONS_ROOT = SCRIPT_PATH.with_name("meta_harness_sota") / "revisions"
DEFAULT_BLANK_SEED_SOURCE = REPO_ROOT / "third_party" / "vector-db-bench" / "skeleton"
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_GOAL_QPS = 4000.0
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_sota_runs"
META_STATE_DIRNAME = ".meta_harness"
BEST_CANDIDATE_DIRNAME = "best_candidate"
BEST_CANDIDATE_MANIFEST = "best_candidate_manifest.json"
PROGRESS_STATE_FILENAME = "progress_state.json"


def _default_run_root(revision_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_ROOT / f"{revision_id}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-id", type=str, default="sota_000")
    parser.add_argument("--revisions-root", type=Path, default=DEFAULT_SOTA_REVISIONS_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)

    parser.add_argument("--model-name", type=str, default="qwen3-coder-next")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-id", type=str, default="qwen/qwen3-coder-next")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--thinking-mode", type=str, default="none")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)

    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=300)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)
    parser.add_argument("--goal-qps", type=float, default=DEFAULT_GOAL_QPS)
    parser.add_argument(
        "--carryover-worker-incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carry the best worker code snapshot across episodes.",
    )
    parser.add_argument(
        "--stop-at-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop once the goal QPS is reached with valid recall.",
    )
    parser.add_argument("--worker-incumbent-snapshot", type=Path, default=None)
    return parser.parse_args()


def _revision_payload(revision: object) -> dict[str, Any]:
    payload = asdict(revision)
    payload["revision_dir"] = str(payload["revision_dir"])
    payload["seed_files_dir"] = str(payload["seed_files_dir"]) if payload["seed_files_dir"] else None
    payload["helper_tools_module"] = str(payload["helper_tools_module"]) if payload["helper_tools_module"] else None
    return payload


def _attempt_summary_payload(outcome: AttemptOutcome, *, progress_state: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "episode": outcome.attempt_index,
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
    if progress_state is not None:
        payload["progress_state"] = progress_state
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _overlay_worker_incumbent(snapshot_dir: Path, work_dir: Path) -> int:
    if not snapshot_dir.exists():
        return 0
    restored = 0
    for src in sorted(snapshot_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(snapshot_dir)
        dst = work_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored += 1
    return restored


def _current_milestone(best_qps: float) -> tuple[str, float | None]:
    ladder = [
        (0.0, "no_valid_result"),
        (1.0, "valid_exact_baseline"),
        (50.0, "qps_50"),
        (100.0, "qps_100"),
        (500.0, "qps_500"),
        (1000.0, "qps_1000"),
        (2000.0, "qps_2000"),
        (3000.0, "qps_3000"),
        (4000.0, "qps_4000"),
    ]
    current = ladder[0][1]
    next_target = None
    for threshold, label in ladder:
        if best_qps >= threshold:
            current = label
        elif next_target is None:
            next_target = threshold
            break
    return current, next_target


def _write_dynamic_teacher_files(
    *,
    work_dir: Path,
    episode_index: int,
    goal_qps: float,
    incumbent_best_qps: float,
    incumbent_best_recall: float,
    incumbent_manifest: dict[str, Any] | None,
) -> None:
    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    milestone, next_target = _current_milestone(incumbent_best_qps)
    best_note = "No valid incumbent yet."
    if incumbent_manifest:
        best_note = (
            f"Best snapshot note: {incumbent_manifest.get('note', 'n/a')} | "
            f"files: {len(incumbent_manifest.get('files', []))}"
        )
    incumbent_record = "\n".join(
        [
            "# Incumbent Record",
            "",
            f"- episode: {episode_index}",
            f"- incumbent best valid QPS: {incumbent_best_qps:.2f}",
            f"- incumbent best recall: {incumbent_best_recall:.4f}",
            f"- current milestone: {milestone}",
            f"- target QPS: {goal_qps:.2f}",
            f"- next milestone target: {next_target if next_target is not None else 'goal reached'}",
            f"- note: {best_note}",
            "",
            "Use this file as ground truth for the current best known valid state. If you break build or recall, restore the best candidate immediately.",
        ]
    )
    milestones = "\n".join(
        [
            "# Milestones",
            "",
            f"- current: {milestone}",
            f"- best valid QPS: {incumbent_best_qps:.2f}",
            f"- target: {goal_qps:.2f}",
            f"- next target: {next_target if next_target is not None else 'goal reached'}",
            "",
            "Milestone ladder:",
            "- valid_exact_baseline",
            "- qps_50",
            "- qps_100",
            "- qps_500",
            "- qps_1000",
            "- qps_2000",
            "- qps_3000",
            "- qps_4000",
        ]
    )
    (src_dir / "incumbent_record.md").write_text(incumbent_record + "\n", encoding="utf-8")
    (src_dir / "milestones.md").write_text(milestones + "\n", encoding="utf-8")


def _promote_worker_incumbent(
    *,
    work_dir: Path,
    incumbent_snapshot_dir: Path,
    incumbent_manifest_path: Path,
    incumbent_progress_path: Path,
    previous_best_qps: float,
) -> tuple[bool, float, float, dict[str, Any] | None]:
    source_state_dir = work_dir / META_STATE_DIRNAME
    source_snapshot_dir = source_state_dir / BEST_CANDIDATE_DIRNAME
    source_manifest_path = source_state_dir / BEST_CANDIDATE_MANIFEST
    source_progress_path = source_state_dir / PROGRESS_STATE_FILENAME
    manifest = _read_json_if_exists(source_manifest_path)
    if not source_snapshot_dir.exists() or manifest is None:
        return False, previous_best_qps, 0.0, None
    benchmark = manifest.get("best_benchmark") or {}
    qps = float(benchmark.get("qps", 0.0) or 0.0)
    recall = float(benchmark.get("recall", 0.0) or 0.0)
    if qps <= 0.0 or qps < previous_best_qps:
        return False, previous_best_qps, recall, manifest

    if incumbent_snapshot_dir.exists():
        shutil.rmtree(incumbent_snapshot_dir)
    shutil.copytree(source_snapshot_dir, incumbent_snapshot_dir)
    incumbent_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_progress_path.exists():
        shutil.copy2(source_progress_path, incumbent_progress_path)
    return True, qps, recall, manifest


def _summary_payload(
    *,
    revision_id: str,
    episodes_requested: int,
    outcomes: list[AttemptOutcome],
    run_root: Path,
    goal_qps: float,
    incumbent_snapshot_dir: Path,
    incumbent_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_outcomes = [o for o in outcomes if o.valid]
    best_outcome = max(valid_outcomes, key=lambda o: o.qps, default=None)
    valid_qps = [o.qps for o in valid_outcomes]
    elapsed = [o.elapsed_secs for o in outcomes]
    best_qps = best_outcome.qps if best_outcome else 0.0
    milestone, _ = _current_milestone(best_qps)
    return {
        "revision_id": revision_id,
        "episodes_requested": episodes_requested,
        "episodes_completed": len(outcomes),
        "valid_episodes": len(valid_outcomes),
        "best_qps": best_qps,
        "best_episode": best_outcome.attempt_index if best_outcome else None,
        "best_recall": best_outcome.recall if best_outcome else 0.0,
        "median_valid_qps": statistics.median(valid_qps) if valid_qps else 0.0,
        "mean_valid_qps": statistics.fmean(valid_qps) if valid_qps else 0.0,
        "mean_elapsed_secs": statistics.fmean(elapsed) if elapsed else 0.0,
        "goal_qps": goal_qps,
        "goal_reached": bool(best_qps >= goal_qps),
        "best_milestone": milestone,
        "incumbent_snapshot_dir": str(incumbent_snapshot_dir) if incumbent_snapshot_dir.exists() else None,
        "incumbent_manifest": incumbent_manifest,
        "run_root": str(run_root),
    }


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)

    revision = load_revision_config(args.revisions_root, args.revision_id)
    episodes = args.episodes if args.episodes is not None else revision.attempts_per_eval
    if episodes <= 0:
        raise SystemExit("--episodes must be positive")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY") or ""
    if not api_key:
        raise SystemExit("API key missing. Pass --api-key or set OPENROUTER_API_KEY in the environment/.env.")

    bench_repo = args.bench_repo.resolve()
    blank_seed_source = args.blank_seed_source.resolve()
    data_dir = args.data_dir.resolve()
    run_root = args.run_root.resolve() if args.run_root else _default_run_root(revision.revision_id)
    blank_seed_dir = run_root / "blank_seed"
    results_path = run_root / "results.tsv"
    summary_path = run_root / "summary.json"
    config_path = run_root / "run_config.json"
    incumbent_snapshot_dir = run_root / "incumbent_snapshot"
    incumbent_manifest_path = run_root / "incumbent_snapshot_manifest.json"
    incumbent_progress_path = run_root / "incumbent_progress_state.json"

    if not bench_repo.is_dir():
        raise SystemExit(f"bench repo not found: {bench_repo}")
    if not blank_seed_source.is_dir():
        raise SystemExit(f"blank seed source not found: {blank_seed_source}")
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    run_root.mkdir(parents=True, exist_ok=True)
    ensure_blank_seed(blank_seed_dir, blank_seed_source)
    validate_revision_worker_contract(blank_seed_dir=blank_seed_dir, revision=revision)

    seeded_incumbent_snapshot = None
    if args.worker_incumbent_snapshot is not None:
        seeded_incumbent_snapshot = args.worker_incumbent_snapshot.resolve()
        if not seeded_incumbent_snapshot.exists():
            raise SystemExit(f"worker incumbent snapshot not found: {seeded_incumbent_snapshot}")
        if incumbent_snapshot_dir.exists():
            shutil.rmtree(incumbent_snapshot_dir)
        shutil.copytree(seeded_incumbent_snapshot, incumbent_snapshot_dir)

    write_json(
        config_path,
        {
            "revision": _revision_payload(revision),
            "episodes": episodes,
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
            "goal_qps": args.goal_qps,
            "carryover_worker_incumbent": args.carryover_worker_incumbent,
            "worker_incumbent_snapshot": str(seeded_incumbent_snapshot) if seeded_incumbent_snapshot else None,
            "run_root": str(run_root),
        },
    )

    writer = build_results_writer(results_path)
    outcomes: list[AttemptOutcome] = []
    incumbent_best_qps = 0.0
    incumbent_best_recall = 0.0
    incumbent_manifest = _read_json_if_exists(incumbent_manifest_path)
    if incumbent_manifest is not None:
        incumbent_best_qps = float(((incumbent_manifest.get("best_benchmark") or {}).get("qps", 0.0)) or 0.0)
        incumbent_best_recall = float(((incumbent_manifest.get("best_benchmark") or {}).get("recall", 0.0)) or 0.0)

    try:
        for episode_index in range(1, episodes + 1):
            episode_dir = run_root / f"episode_{episode_index:03d}"
            work_dir = episode_dir / "workdir"
            results_dir = episode_dir / "results"
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
            episode_dir.mkdir(parents=True, exist_ok=True)

            prepare_workdir(
                bench_repo=bench_repo,
                blank_seed_dir=blank_seed_dir,
                work_dir=work_dir,
                revision=revision,
                max_tool_calls=args.max_tool_calls,
            )
            if args.carryover_worker_incumbent and incumbent_snapshot_dir.exists():
                _overlay_worker_incumbent(incumbent_snapshot_dir, work_dir)
            _write_dynamic_teacher_files(
                work_dir=work_dir,
                episode_index=episode_index,
                goal_qps=args.goal_qps,
                incumbent_best_qps=incumbent_best_qps,
                incumbent_best_recall=incumbent_best_recall,
                incumbent_manifest=incumbent_manifest,
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
                attempt_index=episode_index,
                revision=revision,
                process=process,
                work_dir=work_dir,
                results_dir=results_dir,
            )
            progress_state = _read_json_if_exists(work_dir / META_STATE_DIRNAME / PROGRESS_STATE_FILENAME)
            promoted, incumbent_best_qps, incumbent_best_recall, incumbent_manifest = _promote_worker_incumbent(
                work_dir=work_dir,
                incumbent_snapshot_dir=incumbent_snapshot_dir,
                incumbent_manifest_path=incumbent_manifest_path,
                incumbent_progress_path=incumbent_progress_path,
                previous_best_qps=incumbent_best_qps,
            )
            outcomes.append(outcome)
            write_json(episode_dir / "episode_summary.json", _attempt_summary_payload(outcome, progress_state=progress_state))
            writer.writerow(attempt_row(outcome))
            flush_results_writer(writer)
            write_json(
                summary_path,
                _summary_payload(
                    revision_id=revision.revision_id,
                    episodes_requested=episodes,
                    outcomes=outcomes,
                    run_root=run_root,
                    goal_qps=args.goal_qps,
                    incumbent_snapshot_dir=incumbent_snapshot_dir,
                    incumbent_manifest=incumbent_manifest,
                ),
            )
            if promoted and args.stop_at_goal and incumbent_best_qps >= args.goal_qps:
                break
    finally:
        close_results_writer(writer)

    write_json(
        summary_path,
        _summary_payload(
            revision_id=revision.revision_id,
            episodes_requested=episodes,
            outcomes=outcomes,
            run_root=run_root,
            goal_qps=args.goal_qps,
            incumbent_snapshot_dir=incumbent_snapshot_dir,
            incumbent_manifest=incumbent_manifest,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
