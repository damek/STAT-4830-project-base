#!/usr/bin/env python3
"""Run a persistent Codex-only vector-db-bench superagent campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from meta_harness_common import DEFAULT_DOTENV_PATH, ensure_blank_seed, load_dotenv, write_json

try:
    from meta_harness_runtime import (
        RECALL_THRESHOLD,
        _build_project,
        _run_benchmark_like,
        _run_correctness_test_like,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .meta_harness_runtime import (  # type: ignore[no-redef]
        RECALL_THRESHOLD,
        _build_project,
        _run_benchmark_like,
        _run_correctness_test_like,
    )


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_BLANK_SEED_SOURCE = DEFAULT_BENCH_REPO / "skeleton"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_BOOTSTRAP_DIR = SCRIPT_PATH.with_name("meta_harness_codex_superagent") / "bootstrap"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_GOAL_QPS = 4000.0
DEFAULT_CYCLES = 40
DEFAULT_CYCLE_TIMEOUT_SECONDS = 60 * 30
DEFAULT_QUICK_BENCH_QUERIES = 1000
DEFAULT_FULL_BENCH_THRESHOLD_RATIO = 0.97
DEFAULT_AUTO_RESTORE_INVALID_CYCLES = 2
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "data"
    / "vector_db_bench"
    / "qwen3_meta"
    / "codex_superagent_runs"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)
RESULT_COLUMNS = [
    "cycle",
    "codex_returncode",
    "codex_runtime_seconds",
    "build_success",
    "correctness_passed",
    "quick_qps",
    "quick_recall",
    "full_qps",
    "full_recall",
    "chosen_qps",
    "chosen_recall",
    "valid",
    "promoted",
    "auto_restored_before_cycle",
    "goal_reached",
    "notes",
]
SNAPSHOT_EXCLUDE_DIRS = {
    "target",
    "benchmarks",
    "profiling",
    "__pycache__",
}
SNAPSHOT_EXCLUDE_PREFIXES = (
    ".meta_codex/mainline_snapshot",
    ".meta_codex/recent_cycles",
)
SNAPSHOT_EXCLUDE_FILES = {
    "perf.data",
    "flamegraph.svg",
}


@dataclass(frozen=True)
class CodexExecResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    last_message: str
    runtime_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    build_success: bool
    build_error: str | None
    correctness: dict[str, Any]
    quick_benchmark: dict[str, Any]
    full_benchmark: dict[str, Any] | None
    chosen_result: dict[str, Any] | None
    valid: bool
    promoted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP_DIR)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)

    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--goal-qps", type=float, default=DEFAULT_GOAL_QPS)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)
    parser.add_argument("--quick-benchmark-queries", type=int, default=DEFAULT_QUICK_BENCH_QUERIES)
    parser.add_argument("--full-benchmark-threshold-ratio", type=float, default=DEFAULT_FULL_BENCH_THRESHOLD_RATIO)
    parser.add_argument("--auto-restore-after-invalid-cycles", type=int, default=DEFAULT_AUTO_RESTORE_INVALID_CYCLES)
    parser.add_argument(
        "--stop-at-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop once the promoted mainline reaches the goal QPS.",
    )

    parser.add_argument("--codex-executable", type=str, default="codex")
    parser.add_argument("--codex-timeout-seconds", type=int, default=DEFAULT_CYCLE_TIMEOUT_SECONDS)
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--codex-model", type=str, default="")
    parser.add_argument("--codex-oss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--codex-local-provider", choices=("", "ollama", "lmstudio"), default="")
    parser.add_argument("--codex-enable-web-search", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _require_codex(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Could not find Codex CLI executable {executable!r} on PATH")
    return resolved


def _run_codex_exec(
    *,
    executable: str,
    prompt: str,
    cwd: Path,
    output_path: Path,
    events_path: Path,
    timeout_seconds: int,
    sandbox: str,
    model: str,
    use_oss: bool,
    local_provider: str,
    enable_web_search: bool,
) -> CodexExecResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        executable,
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--json",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "--output-last-message",
        str(output_path),
    ]
    if model:
        argv.extend(["-m", model])
    if enable_web_search:
        argv.extend(["--enable", "web_search_request"])
    if use_oss:
        argv.append("--oss")
    if local_provider:
        argv.extend(["--local-provider", local_provider])
    argv.append("-")

    started_at = time.time()
    proc = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    events_path.write_text(proc.stdout, encoding="utf-8")
    last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    return CodexExecResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        last_message=last_message,
        runtime_seconds=time.time() - started_at,
    )


def _build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
    if path.stat().st_size == 0:
        writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def _flush_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.flush()


def _close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def _ensure_benchmark_binary(bench_repo: Path) -> Path:
    benchmark_dir = bench_repo / "benchmark"
    benchmark_bin = benchmark_dir / "target" / "release" / "vector-db-benchmark"
    if benchmark_bin.exists():
        return benchmark_bin.resolve()
    proc = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to build benchmark crate: {(proc.stderr or proc.stdout)[-2000:]}")
    if not benchmark_bin.exists():
        raise FileNotFoundError(f"benchmark binary not found after build: {benchmark_bin}")
    return benchmark_bin.resolve()


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


def _normalized_relpath(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _should_skip_snapshot_rel(rel: PurePosixPath) -> bool:
    if any(part in SNAPSHOT_EXCLUDE_DIRS for part in rel.parts):
        return True
    rel_str = rel.as_posix()
    if rel_str in SNAPSHOT_EXCLUDE_FILES:
        return True
    return any(rel_str == prefix or rel_str.startswith(f"{prefix}/") for prefix in SNAPSHOT_EXCLUDE_PREFIXES)


def _copy_filtered_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = PurePosixPath(_normalized_relpath(src, path))
        if _should_skip_snapshot_rel(rel):
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _overlay_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _restore_workspace_from_snapshot(snapshot_dir: Path, workspace: Path) -> None:
    preserved_meta: Path | None = None
    if workspace.exists() and (workspace / ".meta_codex").exists():
        preserved_meta = workspace.parent / ".meta_codex_preserved"
        if preserved_meta.exists():
            shutil.rmtree(preserved_meta)
        shutil.copytree(workspace / ".meta_codex", preserved_meta)
    if workspace.exists():
        shutil.rmtree(workspace)
    _copy_filtered_tree(snapshot_dir, workspace)
    if preserved_meta is not None and preserved_meta.exists():
        _overlay_tree(preserved_meta, workspace / ".meta_codex")
        shutil.rmtree(preserved_meta)


def _seed_bootstrap(bootstrap_dir: Path, workspace: Path) -> None:
    meta_dir = workspace / ".meta_codex"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = meta_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(bootstrap_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bootstrap_dir)
        dest = meta_dir / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for path in tools_dir.rglob("*"):
        if path.is_file():
            path.chmod(0o755)


def _sync_recent_cycles(run_root: Path, workspace: Path, keep: int = 5) -> None:
    recent_dir = workspace / ".meta_codex" / "recent_cycles"
    if recent_dir.exists():
        shutil.rmtree(recent_dir)
    recent_dir.mkdir(parents=True, exist_ok=True)
    cycle_summaries = sorted((run_root / "cycles").glob("cycle_*/cycle_summary.json"))
    for path in cycle_summaries[-keep:]:
        target = recent_dir / f"{path.parent.name}.json"
        shutil.copy2(path, target)


def _mirror_mainline_snapshot(run_root: Path, workspace: Path) -> None:
    source = run_root / "mainline_snapshot"
    mirror = workspace / ".meta_codex" / "mainline_snapshot"
    if mirror.exists():
        shutil.rmtree(mirror)
    if source.exists():
        shutil.copytree(source, mirror)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = _load_json_if_exists(path)
    if payload is not None:
        return payload
    return {
        "cycles_requested": 0,
        "cycles_completed": 0,
        "best_qps": 0.0,
        "best_recall": 0.0,
        "best_cycle": None,
        "best_result_source": "none",
        "goal_qps": DEFAULT_GOAL_QPS,
        "goal_reached": False,
        "invalid_cycles_since_promotion": 0,
    }


def _write_dynamic_files(*, workspace: Path, run_root: Path, summary: dict[str, Any], cycle_index: int, goal_qps: float) -> None:
    meta_dir = workspace / ".meta_codex"
    meta_dir.mkdir(parents=True, exist_ok=True)
    best_qps = float(summary.get("best_qps", 0.0) or 0.0)
    best_recall = float(summary.get("best_recall", 0.0) or 0.0)
    milestone, next_target = _current_milestone(best_qps)
    progress_state = {
        "cycle": cycle_index,
        "goal_qps": goal_qps,
        "goal_reached": bool(summary.get("goal_reached", False)),
        "best_qps": best_qps,
        "best_recall": best_recall,
        "best_cycle": summary.get("best_cycle"),
        "best_result_source": summary.get("best_result_source", "none"),
        "invalid_cycles_since_promotion": int(summary.get("invalid_cycles_since_promotion", 0) or 0),
        "current_milestone": milestone,
        "next_target_qps": next_target,
        "mainline_snapshot_dir": str(run_root / "mainline_snapshot"),
        "mainline_manifest_path": str(run_root / "mainline_manifest.json"),
        "trajectory_path": str(run_root / "trajectory.jsonl"),
        "results_path": str(run_root / "results.tsv"),
    }
    write_json(meta_dir / "progress_state.json", progress_state)

    incumbent_lines = [
        "# Incumbent Record",
        "",
        f"- cycle: {cycle_index}",
        f"- best valid QPS so far: {best_qps:.2f}",
        f"- best recall so far: {best_recall:.4f}",
        f"- required recall threshold: {RECALL_THRESHOLD:.2f}",
        f"- best cycle: {summary.get('best_cycle')}",
        f"- best source: {summary.get('best_result_source', 'none')}",
        f"- goal QPS: {goal_qps:.2f}",
        f"- current milestone: {milestone}",
        f"- next target: {next_target if next_target is not None else 'goal reached'}",
        f"- invalid cycles since promotion: {int(summary.get('invalid_cycles_since_promotion', 0) or 0)}",
        "",
        "If the live workspace is badly broken, restore from `.meta_codex/mainline_snapshot/` before continuing.",
    ]
    (meta_dir / "incumbent_record.md").write_text("\n".join(incumbent_lines) + "\n", encoding="utf-8")

    milestone_lines = [
        "# Milestones",
        "",
        f"- current milestone: {milestone}",
        f"- best valid QPS: {best_qps:.2f}",
        f"- goal QPS: {goal_qps:.2f}",
        "",
        "Ladder:",
        "- valid_exact_baseline",
        "- qps_50",
        "- qps_100",
        "- qps_500",
        "- qps_1000",
        "- qps_2000",
        "- qps_3000",
        "- qps_4000",
    ]
    (meta_dir / "milestones.md").write_text("\n".join(milestone_lines) + "\n", encoding="utf-8")

    manifest_payload = {
        "timestamp": _now_rfc3339(),
        "best_qps": best_qps,
        "best_recall": best_recall,
        "best_cycle": summary.get("best_cycle"),
        "best_result_source": summary.get("best_result_source", "none"),
        "goal_qps": goal_qps,
    }
    write_json(run_root / "mainline_manifest.json", manifest_payload)
    write_json(meta_dir / "mainline_manifest.json", manifest_payload)
    write_json(
        meta_dir / "tool_config.json",
        {
            "repo_root": str(REPO_ROOT),
            "run_root": str(run_root),
            "workspace": str(workspace),
            "bench_repo": str((run_root / "superagent_config.json") and json.loads((run_root / "superagent_config.json").read_text(encoding="utf-8")).get("bench_repo", "")),
            "data_dir": str((run_root / "superagent_config.json") and json.loads((run_root / "superagent_config.json").read_text(encoding="utf-8")).get("data_dir", "")),
            "cpu_cores": str((run_root / "superagent_config.json") and json.loads((run_root / "superagent_config.json").read_text(encoding="utf-8")).get("cpu_cores", "")),
            "goal_qps": goal_qps,
            "cycle": cycle_index,
            "tool_calls_total": 2147483647,
        },
    )

    _sync_recent_cycles(run_root, workspace)
    _mirror_mainline_snapshot(run_root, workspace)


def _build_cycle_prompt(*, workspace: Path, run_root: Path, goal_qps: float, cycle_index: int) -> str:
    return textwrap.dedent(
        f"""\
        You are running a persistent Codex superagent campaign for vector-db-bench.

        Objective:
        - reach a valid {goal_qps:.2f}+ QPS solution and beat the public target
        - keep the external benchmark spec official
        - improve both the Rust solution and your own local harness inside this workspace

        Hard constraints:
        - keep the benchmark target official
        - preserve the official dataset, recall threshold of {RECALL_THRESHOLD:.2f}, and CPU pinning policy
        - do not modify the benchmark repo or data outside this workspace
        - the driver evaluates after this cycle and promotes the best valid state

        Operating mode:
        - this workspace persists across cycles
        - there is no artificial tool-call budget for this cycle
        - use web research when useful
        - you may directly edit the Rust solution
        - you may directly edit your local harness under `.meta_codex/`
        - you may create helper scripts under `.meta_codex/tools/` and use them in later cycles
        - optimize for speed of convergence, not for tiny cosmetic changes

        Official-style workspace tools available now:
        - `.meta_codex/tools/build_project`
        - `.meta_codex/tools/run_correctness_test`
        - `.meta_codex/tools/run_benchmark`
        - `.meta_codex/tools/run_profiling`
        - `.meta_codex/tools/get_status`
        - use these first-class wrappers when you want benchmark-style actions during the cycle
        - there is no `finish` tool here because the driver ends and evaluates the cycle externally

        Read first:
        - .meta_codex/README.md
        - .meta_codex/official_constraints.md
        - .meta_codex/strategy.md
        - .meta_codex/design_spec.md
        - .meta_codex/research_notes.md
        - .meta_codex/benchmark_policy.md
        - .meta_codex/incumbent_record.md
        - .meta_codex/milestones.md
        - .meta_codex/campaign_journal.md
        - .meta_codex/progress_state.json
        - any recent cycle summaries under `.meta_codex/recent_cycles/`

        Workspace guidance:
        - if the current workspace is badly broken, restore from `.meta_codex/mainline_snapshot/` or repair directly
        - keep `.meta_codex/campaign_journal.md` current with major decisions
        - prefer architecture changes that move toward real ANN / IVF-style shortlist generation
        - avoid fake ANN designs that still depend on global full-scan behavior
        - create local scripts when they will make future cycles faster

        Cycle:
        - this is cycle {cycle_index}
        - work directly in this workspace
        - end with a concise summary of what changed, current score expectations, and the next best move
        """
    )


def _is_valid_benchmark(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("type") == "RunBenchmark"
        and bool(payload.get("recall_passed", False))
        and float(payload.get("qps", 0.0) or 0.0) > 0.0
    )


def _evaluate_workspace(
    *,
    workspace: Path,
    bench_repo: Path,
    benchmark_bin: Path,
    data_dir: Path,
    cpu_cores: str,
    quick_queries: int,
    best_qps: float,
    full_threshold_ratio: float,
    run_root: Path,
) -> EvaluationResult:
    build_error = _build_project(workspace, profiling=False)
    build_success = build_error is None
    if not build_success:
        return EvaluationResult(
            build_success=False,
            build_error=build_error,
            correctness={"type": "Error", "message": build_error},
            quick_benchmark={"type": "Error", "message": build_error},
            full_benchmark=None,
            chosen_result=None,
            valid=False,
            promoted=False,
        )

    correctness = _run_correctness_test_like(
        work_dir=workspace,
        benchmark_bin=benchmark_bin,
        data_dir=data_dir,
        cpu_cores=cpu_cores,
    )
    quick = _run_benchmark_like(
        work_dir=workspace,
        benchmark_bin=benchmark_bin,
        data_dir=data_dir,
        cpu_cores=cpu_cores,
        concurrency=4,
        warmup=100,
        max_queries=quick_queries,
        save_history=True,
    )
    full = None
    quick_valid = _is_valid_benchmark(quick)
    quick_qps = float(quick.get("qps", 0.0) or 0.0) if quick_valid else 0.0
    should_run_full = quick_valid and (
        best_qps <= 0.0 or quick_qps >= max(1.0, best_qps * full_threshold_ratio)
    )
    if should_run_full:
        full = _run_benchmark_like(
            work_dir=workspace,
            benchmark_bin=benchmark_bin,
            data_dir=data_dir,
            cpu_cores=cpu_cores,
            concurrency=4,
            warmup=100,
            max_queries=0,
            save_history=True,
        )

    chosen = full if _is_valid_benchmark(full) else (quick if quick_valid else None)
    valid = bool(correctness.get("passed", False)) and chosen is not None
    return EvaluationResult(
        build_success=True,
        build_error=None,
        correctness=correctness,
        quick_benchmark=quick,
        full_benchmark=full,
        chosen_result=chosen,
        valid=valid,
        promoted=False,
    )


def _promote_workspace(workspace: Path, run_root: Path) -> None:
    snapshot_dir = run_root / "mainline_snapshot"
    _copy_filtered_tree(workspace, snapshot_dir)
    _mirror_mainline_snapshot(run_root, workspace)


def _save_cycle_summary(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)
    codex_executable = _require_codex(args.codex_executable)
    if not args.bootstrap_dir.is_dir():
        raise SystemExit(f"bootstrap_dir not found: {args.bootstrap_dir}")

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    blank_seed_dir = run_root / "blank_seed"
    workspace = run_root / "workspace"
    cycles_root = run_root / "cycles"
    outputs_root = run_root / "codex_outputs"
    prompts_root = run_root / "codex_prompts"
    results_path = run_root / "results.tsv"
    trajectory_path = run_root / "trajectory.jsonl"
    summary_path = run_root / "summary.json"
    config_path = run_root / "superagent_config.json"

    ensure_blank_seed(blank_seed_dir, args.blank_seed_source.resolve())
    cycles_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    prompts_root.mkdir(parents=True, exist_ok=True)

    write_json(
        config_path,
        {
            "run_root": str(run_root),
            "workspace": str(workspace),
            "cycles": args.cycles,
            "goal_qps": args.goal_qps,
            "cpu_cores": args.cpu_cores,
            "quick_benchmark_queries": args.quick_benchmark_queries,
            "full_benchmark_threshold_ratio": args.full_benchmark_threshold_ratio,
            "auto_restore_after_invalid_cycles": args.auto_restore_after_invalid_cycles,
            "bench_repo": str(args.bench_repo.resolve()),
            "blank_seed_source": str(args.blank_seed_source.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "bootstrap_dir": str(args.bootstrap_dir.resolve()),
            "codex_enable_web_search": args.codex_enable_web_search,
        },
    )
    _log(f"[superagent] run_root={run_root}")
    _log(f"[superagent] cycles={args.cycles} goal_qps={args.goal_qps:.2f} cpu_cores={args.cpu_cores}")

    summary = _load_summary(summary_path)
    summary["goal_qps"] = args.goal_qps
    if not workspace.exists():
        shutil.copytree(blank_seed_dir, workspace)
    if not (workspace / "Cargo.toml").exists():
        raise SystemExit(f"workspace is missing Cargo.toml: {workspace}")

    _seed_bootstrap(args.bootstrap_dir.resolve(), workspace)
    if not (run_root / "mainline_snapshot").exists():
        _promote_workspace(workspace, run_root)

    benchmark_bin = _ensure_benchmark_binary(args.bench_repo.resolve())
    writer = _build_results_writer(results_path)
    try:
        start_cycle = int(summary.get("cycles_completed", 0) or 0) + 1
        for cycle_index in range(start_cycle, args.cycles + 1):
            _log(
                f"[cycle {cycle_index:03d}] starting mainline_qps={float(summary.get('best_qps', 0.0) or 0.0):.2f} "
                f"invalid_since_promotion={int(summary.get('invalid_cycles_since_promotion', 0) or 0)}"
            )
            auto_restored = False
            if (
                args.auto_restore_after_invalid_cycles > 0
                and int(summary.get("invalid_cycles_since_promotion", 0) or 0) >= args.auto_restore_after_invalid_cycles
                and (run_root / "mainline_snapshot").exists()
            ):
                _restore_workspace_from_snapshot(run_root / "mainline_snapshot", workspace)
                _seed_bootstrap(args.bootstrap_dir.resolve(), workspace)
                auto_restored = True
                _log(f"[cycle {cycle_index:03d}] restored workspace from promoted mainline snapshot")

            _write_dynamic_files(
                workspace=workspace,
                run_root=run_root,
                summary=summary,
                cycle_index=cycle_index,
                goal_qps=args.goal_qps,
            )
            prompt = _build_cycle_prompt(
                workspace=workspace,
                run_root=run_root,
                goal_qps=args.goal_qps,
                cycle_index=cycle_index,
            )
            prompt_path = prompts_root / f"cycle_{cycle_index:03d}.md"
            prompt_path.write_text(prompt, encoding="utf-8")

            output_path = outputs_root / f"cycle_{cycle_index:03d}_last_message.txt"
            events_path = outputs_root / f"cycle_{cycle_index:03d}_events.jsonl"
            codex_result = _run_codex_exec(
                executable=codex_executable,
                prompt=prompt,
                cwd=workspace,
                output_path=output_path,
                events_path=events_path,
                timeout_seconds=args.codex_timeout_seconds,
                sandbox=args.codex_sandbox,
                model=args.codex_model,
                use_oss=args.codex_oss,
                local_provider=args.codex_local_provider,
                enable_web_search=args.codex_enable_web_search,
            )
            _log(
                f"[cycle {cycle_index:03d}] codex_done returncode={codex_result.returncode} "
                f"runtime_seconds={codex_result.runtime_seconds:.2f}"
            )
            write_json(
                outputs_root / f"cycle_{cycle_index:03d}_exec.json",
                {
                    "argv": codex_result.argv,
                    "returncode": codex_result.returncode,
                    "runtime_seconds": codex_result.runtime_seconds,
                    "stderr": codex_result.stderr,
                    "last_message": codex_result.last_message,
                    "events_path": str(events_path),
                },
            )

            evaluation = _evaluate_workspace(
                workspace=workspace,
                bench_repo=args.bench_repo.resolve(),
                benchmark_bin=benchmark_bin,
                data_dir=args.data_dir.resolve(),
                cpu_cores=args.cpu_cores,
                quick_queries=args.quick_benchmark_queries,
                best_qps=float(summary.get("best_qps", 0.0) or 0.0),
                full_threshold_ratio=args.full_benchmark_threshold_ratio,
                run_root=run_root,
            )
            promoted = False
            chosen = evaluation.chosen_result or {}
            chosen_qps = float(chosen.get("qps", 0.0) or 0.0)
            chosen_recall = float(chosen.get("recall", 0.0) or 0.0)
            if evaluation.valid and chosen_qps > float(summary.get("best_qps", 0.0) or 0.0):
                _promote_workspace(workspace, run_root)
                promoted = True
                summary["best_qps"] = chosen_qps
                summary["best_recall"] = chosen_recall
                summary["best_cycle"] = cycle_index
                summary["best_result_source"] = "full" if _is_valid_benchmark(evaluation.full_benchmark) else "quick"
                summary["invalid_cycles_since_promotion"] = 0
            else:
                summary["invalid_cycles_since_promotion"] = int(summary.get("invalid_cycles_since_promotion", 0) or 0) + (0 if evaluation.valid else 1)

            goal_reached = float(summary.get("best_qps", 0.0) or 0.0) >= args.goal_qps
            _log(
                f"[cycle {cycle_index:03d}] eval build_success={evaluation.build_success} "
                f"correctness_passed={bool(evaluation.correctness.get('passed', False))} "
                f"quick_qps={float(evaluation.quick_benchmark.get('qps', 0.0) or 0.0):.2f} "
                f"full_qps={float((evaluation.full_benchmark or {}).get('qps', 0.0) or 0.0):.2f} "
                f"chosen_qps={chosen_qps:.2f} valid={evaluation.valid} promoted={promoted} "
                f"mainline_qps={float(summary.get('best_qps', 0.0) or 0.0):.2f}"
            )
            summary.update(
                {
                    "run_root": str(run_root),
                    "workspace": str(workspace),
                    "cycles_requested": args.cycles,
                    "cycles_completed": cycle_index,
                    "goal_qps": args.goal_qps,
                    "goal_reached": goal_reached,
                    "mainline_snapshot_dir": str(run_root / "mainline_snapshot"),
                    "trajectory_path": str(trajectory_path),
                    "results_path": str(results_path),
                    "last_cycle": cycle_index,
                    "last_codex_returncode": codex_result.returncode,
                    "last_codex_runtime_seconds": round(codex_result.runtime_seconds, 2),
                    "last_evaluation": {
                        "build_success": evaluation.build_success,
                        "build_error": evaluation.build_error,
                        "correctness": evaluation.correctness,
                        "quick_benchmark": evaluation.quick_benchmark,
                        "full_benchmark": evaluation.full_benchmark,
                        "chosen_result": evaluation.chosen_result,
                        "valid": evaluation.valid,
                        "promoted": promoted,
                    },
                }
            )

            cycle_dir = cycles_root / f"cycle_{cycle_index:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            cycle_summary = {
                "cycle": cycle_index,
                "auto_restored_before_cycle": auto_restored,
                "codex_returncode": codex_result.returncode,
                "codex_runtime_seconds": codex_result.runtime_seconds,
                "codex_last_message": codex_result.last_message,
                "evaluation": {
                    "build_success": evaluation.build_success,
                    "build_error": evaluation.build_error,
                    "correctness": evaluation.correctness,
                    "quick_benchmark": evaluation.quick_benchmark,
                    "full_benchmark": evaluation.full_benchmark,
                    "chosen_result": evaluation.chosen_result,
                    "valid": evaluation.valid,
                    "promoted": promoted,
                },
                "summary_after_cycle": summary,
            }
            _save_cycle_summary(cycle_dir / "cycle_summary.json", cycle_summary)

            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "cycle": cycle_index,
                            "timestamp": _now_rfc3339(),
                            "auto_restored_before_cycle": auto_restored,
                            "codex_returncode": codex_result.returncode,
                            "codex_runtime_seconds": codex_result.runtime_seconds,
                            "build_success": evaluation.build_success,
                            "correctness_passed": bool(evaluation.correctness.get("passed", False)),
                            "quick_qps": float(evaluation.quick_benchmark.get("qps", 0.0) or 0.0)
                            if _is_valid_benchmark(evaluation.quick_benchmark)
                            else 0.0,
                            "full_qps": float((evaluation.full_benchmark or {}).get("qps", 0.0) or 0.0)
                            if _is_valid_benchmark(evaluation.full_benchmark)
                            else 0.0,
                            "chosen_qps": chosen_qps,
                            "chosen_recall": chosen_recall,
                            "valid": evaluation.valid,
                            "promoted": promoted,
                            "mainline_qps_after": float(summary.get("best_qps", 0.0) or 0.0),
                            "goal_reached": goal_reached,
                        }
                    )
                    + "\n"
                )

            writer.writerow(
                {
                    "cycle": cycle_index,
                    "codex_returncode": codex_result.returncode,
                    "codex_runtime_seconds": f"{codex_result.runtime_seconds:.2f}",
                    "build_success": str(evaluation.build_success).lower(),
                    "correctness_passed": str(bool(evaluation.correctness.get("passed", False))).lower(),
                    "quick_qps": f"{float(evaluation.quick_benchmark.get('qps', 0.0) or 0.0):.2f}",
                    "quick_recall": f"{float(evaluation.quick_benchmark.get('recall', 0.0) or 0.0):.4f}",
                    "full_qps": f"{float((evaluation.full_benchmark or {}).get('qps', 0.0) or 0.0):.2f}",
                    "full_recall": f"{float((evaluation.full_benchmark or {}).get('recall', 0.0) or 0.0):.4f}",
                    "chosen_qps": f"{chosen_qps:.2f}",
                    "chosen_recall": f"{chosen_recall:.4f}",
                    "valid": str(evaluation.valid).lower(),
                    "promoted": str(promoted).lower(),
                    "auto_restored_before_cycle": str(auto_restored).lower(),
                    "goal_reached": str(goal_reached).lower(),
                    "notes": codex_result.last_message[:500],
                }
            )
            _flush_results_writer(writer)
            _write_summary(summary_path, summary)

            if args.stop_at_goal and goal_reached:
                _log(f"[cycle {cycle_index:03d}] goal reached; stopping")
                break
    finally:
        _close_results_writer(writer)

    _write_summary(summary_path, summary)
    _log(
        f"[superagent] completed cycles={int(summary.get('cycles_completed', 0) or 0)} "
        f"best_qps={float(summary.get('best_qps', 0.0) or 0.0):.2f} "
        f"goal_reached={bool(summary.get('goal_reached', False))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
