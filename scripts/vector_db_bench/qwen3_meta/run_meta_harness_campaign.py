#!/usr/bin/env python3
"""Run a persistent long-horizon Meta-Harness campaign for vector-db-bench.

This mode is distinct from fresh-start evaluation. It keeps a persistent worker
codebase and mainline/experiment state across cycles, while refreshing the chat
budget and teacher package each cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
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
DEFAULT_REVISIONS_ROOT = SCRIPT_PATH.with_name("meta_harness_campaign") / "revisions"
DEFAULT_BLANK_SEED_SOURCE = REPO_ROOT / "third_party" / "vector-db-bench" / "skeleton"
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_GOAL_QPS = 4000.0
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_campaign_runs"
META_STATE_DIRNAME = ".meta_harness"
MAINLINE_MANIFEST = "mainline_manifest.json"
BEST_CANDIDATE_MANIFEST = "best_candidate_manifest.json"
PROGRESS_STATE_FILENAME = "progress_state.json"
CAMPAIGN_STATE_FILENAME = "campaign_state.json"


def _default_run_root(revision_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_ROOT / f"{revision_id}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-id", type=str, default="campaign_000")
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
    parser.add_argument("--thinking-mode", type=str, default="none")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)

    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=300)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)
    parser.add_argument("--goal-qps", type=float, default=DEFAULT_GOAL_QPS)
    parser.add_argument("--mainline-snapshot", type=Path, default=None)
    parser.add_argument("--mainline-manifest", type=Path, default=None)
    parser.add_argument(
        "--stop-at-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop once the campaign reaches the goal QPS with valid recall.",
    )
    return parser.parse_args()


def _chat_message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


def _copy_seed_files(seed_files_dir: Path, work_dir: Path, mount_dir: str) -> None:
    for path in sorted(seed_files_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(seed_files_dir)
        dest = work_dir / mount_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _now_rfc3339() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _meta_state_dir(work_dir: Path) -> Path:
    return work_dir / META_STATE_DIRNAME


def _mainline_manifest(work_dir: Path) -> dict[str, Any] | None:
    return _load_json_if_exists(_meta_state_dir(work_dir) / MAINLINE_MANIFEST)


def _best_candidate_manifest(work_dir: Path) -> dict[str, Any] | None:
    return _load_json_if_exists(_meta_state_dir(work_dir) / BEST_CANDIDATE_MANIFEST)


def _progress_state(work_dir: Path) -> dict[str, Any] | None:
    return _load_json_if_exists(_meta_state_dir(work_dir) / PROGRESS_STATE_FILENAME)


def _campaign_state(work_dir: Path) -> dict[str, Any] | None:
    return _load_json_if_exists(_meta_state_dir(work_dir) / CAMPAIGN_STATE_FILENAME)


def _seed_mainline_snapshot(snapshot_dir: Path, work_dir: Path, manifest_path: Path | None = None) -> int:
    if not snapshot_dir.exists():
        return 0
    state_dir = _meta_state_dir(work_dir)
    mainline_dir = state_dir / "mainline"
    if mainline_dir.exists():
        shutil.rmtree(mainline_dir)
    mainline_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    relpaths: list[str] = []
    for src in sorted(snapshot_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(snapshot_dir)
        relpaths.append(rel.as_posix())
        for root in (work_dir, mainline_dir):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        restored += 1
    manifest = _load_json_if_exists(manifest_path) if manifest_path is not None else None
    if not isinstance(manifest, dict):
        manifest = {}
    manifest = {
        "timestamp": str(manifest.get("timestamp") or _now_rfc3339()),
        "note": str(manifest.get("note") or "seeded_mainline_snapshot"),
        "tool_calls_used": int(manifest.get("tool_calls_used", 0) or 0),
        "best_benchmark": manifest.get("best_benchmark"),
        "milestone": str(manifest.get("milestone") or "no_valid_result"),
        "files": relpaths,
    }
    (state_dir / MAINLINE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    campaign_state = {
        "active_branch": "mainline",
        "experiments": {},
        "last_promotion_at": _now_rfc3339(),
    }
    (state_dir / CAMPAIGN_STATE_FILENAME).write_text(
        json.dumps(campaign_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return restored


def _refresh_teacher_package(*, bench_repo: Path, work_dir: Path, revision: Any, cycle_index: int, goal_qps: float) -> None:
    if revision.seed_files_dir is not None and revision.seed_files_mount_dir:
        _copy_seed_files(revision.seed_files_dir, work_dir, revision.seed_files_mount_dir)

    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    progress = _progress_state(work_dir) or {}
    mainline = _mainline_manifest(work_dir) or {}
    best_benchmark = (mainline.get("best_benchmark") or progress.get("best_benchmark") or {})
    best_qps = float(best_benchmark.get("qps", 0.0) or 0.0)
    best_recall = float(best_benchmark.get("recall", 0.0) or 0.0)
    milestone, next_target = _current_milestone(best_qps)
    active_branch = str(((progress.get("campaign_state") or {}).get("active_branch")) or "mainline")
    failure_guard = progress.get("benchmark_failure_guard") or {}

    incumbent_record = "\n".join(
        [
            "# Incumbent Record",
            "",
            f"- cycle: {cycle_index}",
            f"- active branch: {active_branch}",
            f"- best valid QPS so far: {best_qps:.2f}",
            f"- best recall so far: {best_recall:.4f}",
            f"- current milestone: {milestone}",
            f"- next target: {next_target if next_target is not None else 'goal reached'}",
            f"- goal QPS: {goal_qps:.2f}",
            "",
            "Use mainline for stable progress. Use named experiments for risky architectural moves.",
        ]
    )
    (src_dir / "incumbent_record.md").write_text(incumbent_record + "\n", encoding="utf-8")

    milestones = "\n".join(
        [
            "# Milestones",
            "",
            f"- current milestone: {milestone}",
            f"- best valid QPS: {best_qps:.2f}",
            f"- goal QPS: {goal_qps:.2f}",
            "",
            "Ladder:",
            "- valid_exact_baseline",
            "- first_ivf",
            "- qps_500",
            "- qps_1000",
            "- qps_2000",
            "- qps_3000",
            "- qps_4000",
        ]
    )
    (src_dir / "milestones.md").write_text(milestones + "\n", encoding="utf-8")

    journal_path = src_dir / "campaign_journal.md"
    if not journal_path.exists():
        journal_path.write_text("# Campaign Journal\n\n", encoding="utf-8")
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Cycle {cycle_index}\n")
        handle.write(f"- active branch: {active_branch}\n")
        handle.write(f"- best valid QPS: {best_qps:.2f}\n")
        if failure_guard:
            handle.write(f"- repeated failure guard: {failure_guard.get('count', 0)} x {failure_guard.get('signature', '')[:160]}\n")


def _write_cycle_session_context(*, bench_repo: Path, work_dir: Path, revision: Any, max_tool_calls: int, cycle_index: int) -> None:
    system_prompt = (bench_repo / "agent" / "system_prompt.txt").read_text(encoding="utf-8")
    prior_context = _load_json_if_exists(work_dir / "session_context.json") or {}
    progress = _progress_state(work_dir) or {}
    milestone = ((progress.get("milestone") or {}).get("label")) or "no_valid_result"
    extra_messages = list(revision.extra_user_messages)
    extra_messages.append(
        f"Campaign cycle {cycle_index}. Continue improving the persistent codebase. Current milestone: {milestone}. Use mainline and experiment tools deliberately."
    )
    messages = [
        _chat_message("system", system_prompt),
        _chat_message("user", "Begin. Read the project files and start implementing."),
    ]
    messages.extend(_chat_message("user", content) for content in extra_messages)
    session_context = {
        "tool_calls_used": 0,
        "tool_calls_total": max_tool_calls,
        "messages": messages,
        "last_benchmark": prior_context.get("last_benchmark") or progress.get("last_benchmark"),
        "best_benchmark": prior_context.get("best_benchmark") or progress.get("best_benchmark"),
        "call_log": prior_context.get("call_log", []),
        "benchmark_failure_guard": prior_context.get("benchmark_failure_guard") or progress.get("benchmark_failure_guard") or {},
    }
    (work_dir / "session_context.json").write_text(json.dumps(session_context), encoding="utf-8")


def _cycle_summary_payload(outcome: AttemptOutcome, progress_state: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "cycle": outcome.attempt_index,
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


def _write_root_summary(*, revision_id: str, cycles_requested: int, outcomes: list[AttemptOutcome], run_root: Path, goal_qps: float) -> dict[str, Any]:
    valid_outcomes = [o for o in outcomes if o.valid]
    best_outcome = max(valid_outcomes, key=lambda o: o.qps, default=None)
    valid_qps = [o.qps for o in valid_outcomes]
    elapsed = [o.elapsed_secs for o in outcomes]
    workspace = run_root / "workspace"
    mainline_manifest = _mainline_manifest(workspace) or {}
    mainline_best = mainline_manifest.get("best_benchmark") or {}
    mainline_best_qps = float(mainline_best.get("qps", 0.0) or 0.0)
    mainline_best_recall = float(mainline_best.get("recall", 0.0) or 0.0)
    segment_best_qps = best_outcome.qps if best_outcome else 0.0
    if mainline_best_qps >= segment_best_qps:
        best_qps = mainline_best_qps
        best_recall = mainline_best_recall
        best_cycle = best_outcome.attempt_index if best_outcome and best_outcome.qps == best_qps else None
    else:
        best_qps = segment_best_qps
        best_recall = best_outcome.recall if best_outcome else 0.0
        best_cycle = best_outcome.attempt_index if best_outcome else None
    milestone, _ = _current_milestone(best_qps)
    mainline_snapshot_dir = _meta_state_dir(workspace) / "mainline"
    summary = {
        "revision_id": revision_id,
        "cycles_requested": cycles_requested,
        "cycles_completed": len(outcomes),
        "valid_cycles": len(valid_outcomes),
        "best_qps": best_qps,
        "best_cycle": best_cycle,
        "best_recall": best_recall,
        "median_valid_qps": statistics.median(valid_qps) if valid_qps else 0.0,
        "mean_valid_qps": statistics.fmean(valid_qps) if valid_qps else 0.0,
        "mean_elapsed_secs": statistics.fmean(elapsed) if elapsed else 0.0,
        "goal_qps": goal_qps,
        "goal_reached": bool(best_qps >= goal_qps),
        "best_milestone": milestone,
        "run_root": str(run_root),
        "workspace": str(workspace),
        "mainline_snapshot_dir": str(mainline_snapshot_dir) if mainline_snapshot_dir.exists() else None,
        "mainline_manifest_path": str(_meta_state_dir(workspace) / MAINLINE_MANIFEST) if (_meta_state_dir(workspace) / MAINLINE_MANIFEST).exists() else None,
        "mainline_manifest": mainline_manifest,
        "campaign_state": _campaign_state(workspace),
        "trajectory_path": str(run_root / "trajectory.jsonl"),
    }
    write_json(run_root / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)

    revision = load_revision_config(args.revisions_root, args.revision_id)
    run_root = args.run_root.resolve() if args.run_root else _default_run_root(revision.revision_id)
    blank_seed_dir = run_root / "blank_seed"
    workspace = run_root / "workspace"
    cycles_root = run_root / "cycles"
    results_path = run_root / "results.tsv"
    trajectory_path = run_root / "trajectory.jsonl"
    summary_path = run_root / "summary.json"
    config_path = run_root / "campaign_config.json"

    ensure_blank_seed(blank_seed_dir, args.blank_seed_source.resolve())
    validate_revision_worker_contract(blank_seed_dir=blank_seed_dir, revision=revision)

    run_root.mkdir(parents=True, exist_ok=True)
    cycles_root.mkdir(parents=True, exist_ok=True)
    write_json(
        config_path,
        {
            "revision_id": revision.revision_id,
            "cycles": args.cycles,
            "goal_qps": args.goal_qps,
            "max_tool_calls": args.max_tool_calls,
            "run_root": str(run_root),
            "workspace": str(workspace),
            "revisions_root": str(args.revisions_root.resolve()),
            "bench_repo": str(args.bench_repo.resolve()),
            "blank_seed_source": str(args.blank_seed_source.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "mainline_snapshot": str(args.mainline_snapshot.resolve()) if args.mainline_snapshot else None,
            "mainline_manifest": str(args.mainline_manifest.resolve()) if args.mainline_manifest else None,
        },
    )

    if not workspace.exists():
        prepare_workdir(
            bench_repo=args.bench_repo.resolve(),
            blank_seed_dir=blank_seed_dir,
            work_dir=workspace,
            revision=revision,
            max_tool_calls=args.max_tool_calls,
        )
        if args.mainline_snapshot is not None:
            _seed_mainline_snapshot(
                args.mainline_snapshot.resolve(),
                workspace,
                args.mainline_manifest.resolve() if args.mainline_manifest else None,
            )

    outcomes: list[AttemptOutcome] = []
    writer = build_results_writer(results_path)
    try:
        for cycle_index in range(1, args.cycles + 1):
            cycle_dir = cycles_root / f"cycle_{cycle_index:03d}"
            results_dir = cycle_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            _refresh_teacher_package(
                bench_repo=args.bench_repo.resolve(),
                work_dir=workspace,
                revision=revision,
                cycle_index=cycle_index,
                goal_qps=args.goal_qps,
            )
            _write_cycle_session_context(
                bench_repo=args.bench_repo.resolve(),
                work_dir=workspace,
                revision=revision,
                max_tool_calls=args.max_tool_calls,
                cycle_index=cycle_index,
            )

            process = run_revision_attempt(
                revision=revision,
                bench_repo=args.bench_repo.resolve(),
                work_dir=workspace,
                results_dir=results_dir,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY") or "",
                model_id=args.model_id,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
                api_interval_ms=args.api_interval_ms,
                cpu_cores=args.cpu_cores,
                data_dir=args.data_dir.resolve(),
                max_tool_calls=args.max_tool_calls,
            )
            outcome = attempt_outcome_from_logs(
                attempt_index=cycle_index,
                revision=revision,
                process=process,
                work_dir=workspace,
                results_dir=results_dir,
            )
            outcomes.append(outcome)
            writer.writerow(attempt_row(outcome))
            flush_results_writer(writer)

            progress_state = _progress_state(workspace)
            write_json(cycle_dir / "cycle_summary.json", _cycle_summary_payload(outcome, progress_state))
            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "cycle": cycle_index,
                    "status": outcome.status,
                    "valid": outcome.valid,
                    "qps": outcome.qps,
                    "recall": outcome.recall,
                    "elapsed_secs": outcome.elapsed_secs,
                    "milestone": ((progress_state or {}).get("milestone") or {}).get("label"),
                    "active_branch": (((progress_state or {}).get("campaign_state") or {}).get("active_branch")),
                    "tool_calls_used": outcome.tool_calls_used,
                    "process_returncode": outcome.process_returncode,
                }) + "\n")
            summary = _write_root_summary(
                revision_id=revision.revision_id,
                cycles_requested=args.cycles,
                outcomes=outcomes,
                run_root=run_root,
                goal_qps=args.goal_qps,
            )
            if args.stop_at_goal and summary["goal_reached"]:
                break
    finally:
        close_results_writer(writer)

    if not summary_path.exists():
        _write_root_summary(
            revision_id=revision.revision_id,
            cycles_requested=args.cycles,
            outcomes=outcomes,
            run_root=run_root,
            goal_qps=args.goal_qps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
