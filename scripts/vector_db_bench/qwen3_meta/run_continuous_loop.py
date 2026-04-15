#!/usr/bin/env python3
"""Run a continuous incumbent-seeded vector-db-bench loop.

This is an autoresearch-style outer harness:
- round 1 starts from the official blank scaffold
- later rounds start from the best measured incumbent so far
- each round uses the upstream benchmark agent/runtime unchanged
- promotion is based on an independent final evaluation of the round's final
  workspace, so the kept incumbent is always a buildable, measured candidate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex_cli_harness import (  # noqa: E402
    BenchEvalResult,
    EvalInputs,
    RunConfig,
    _discover_editable_skeleton_files,
    _ensure_benchmark_binary,
    _evaluate_candidate,
    _maybe_make_subset,
    _parse_server_port,
    _resolve_base_vectors_file,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "data"
    / "vector_db_bench"
    / "qwen3_meta"
    / f"continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
RESULT_COLUMNS = [
    "round",
    "seed_source",
    "status",
    "promoted",
    "agent_best_qps",
    "agent_best_recall",
    "agent_last_qps",
    "agent_last_recall",
    "final_eval_valid",
    "final_eval_qps",
    "final_eval_recall",
    "final_eval_failure_type",
    "incumbent_qps_before",
    "incumbent_qps_after",
    "tool_calls_used",
    "tool_calls_total",
    "elapsed_secs",
    "work_dir",
    "notes",
]


@dataclass(frozen=True)
class IncumbentState:
    qps: float
    recall: float
    valid: bool
    source_round: int
    status: str
    notes: str


@dataclass(frozen=True)
class RoundOutcome:
    round_index: int
    seed_source: str
    status: str
    promoted: bool
    agent_best_qps: float
    agent_best_recall: float
    agent_last_qps: float
    agent_last_recall: float
    final_eval: BenchEvalResult | None
    incumbent_qps_before: float
    incumbent_qps_after: float
    tool_calls_used: int
    tool_calls_total: int
    elapsed_secs: float
    work_dir: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True, help="Path to a clean vector-db-bench clone on the benchmark host.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--blank-seed-source",
        type=Path,
        default=None,
        help="Optional path to the official blank skeleton. Defaults to <bench-repo>/skeleton on first run.",
    )
    parser.add_argument("--model-name", type=str, default="qwen3-coder-next")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-id", type=str, default="qwen/qwen3-coder-next")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--thinking-mode", type=str, default="openrouter-openai")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--max-tool-calls", type=int, default=50)
    parser.add_argument("--recall-threshold", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-max-queries", type=int, default=0, help="0 means the full query set.")
    parser.add_argument("--build-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    parser.add_argument("--round-timeout-seconds", type=int, default=60 * 60)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--server-bin-name", type=str, default="vector-db-skeleton")
    parser.add_argument("--benchmark-bin-name", type=str, default="vector-db-benchmark")
    parser.add_argument("--cpu-cores", type=str, default="")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--base-vectors", type=Path, default=None)
    parser.add_argument("--query-vectors", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--proxy-max-queries", type=int, default=2000)
    parser.add_argument("--results-dir-name", type=str, default="results")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue later rounds after a crashed/failed round.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an existing run-root if present.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> list[str]:
    if not path.exists():
        return []
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _build_results_writer(path: Path, *, append: bool) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
    if not append:
        writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def _close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def _flush_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.flush()


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_blank_seed(args: argparse.Namespace, blank_seed_dir: Path) -> None:
    if blank_seed_dir.exists():
        return
    source = (args.blank_seed_source or (args.bench_repo / "skeleton")).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"blank seed source not found: {source}")
    _copy_tree(source, blank_seed_dir)
    db_stub = (blank_seed_dir / "src" / "db.rs").read_text(encoding="utf-8")
    dist_stub = (blank_seed_dir / "src" / "distance.rs").read_text(encoding="utf-8")
    if "todo!(" not in db_stub or "todo!(" not in dist_stub:
        raise RuntimeError(
            "blank seed source does not look like the official empty scaffold; "
            "pass --blank-seed-source explicitly."
        )


def _initialize_incumbent(blank_seed_dir: Path, incumbent_dir: Path, state_path: Path) -> IncumbentState:
    if not incumbent_dir.exists():
        _copy_tree(blank_seed_dir, incumbent_dir)
    state = IncumbentState(
        qps=0.0,
        recall=0.0,
        valid=False,
        source_round=0,
        status="blank_seed",
        notes="Initialized from official blank scaffold.",
    )
    _write_json(state_path, asdict(state))
    return state


def _load_incumbent_state(state_path: Path) -> IncumbentState:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return IncumbentState(
        qps=float(payload.get("qps", 0.0) or 0.0),
        recall=float(payload.get("recall", 0.0) or 0.0),
        valid=bool(payload.get("valid", False)),
        source_round=int(payload.get("source_round", 0) or 0),
        status=str(payload.get("status", "")),
        notes=str(payload.get("notes", "")),
    )


def _next_round_index(run_root: Path) -> int:
    existing: list[int] = []
    for path in run_root.glob("round_*"):
        try:
            existing.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(existing, default=0) + 1


def _seed_round_skeleton(seed_dir: Path, skeleton_dir: Path) -> None:
    _copy_tree(seed_dir, skeleton_dir)


def _chat_message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


def _build_handoff_message(incumbent: IncumbentState) -> str:
    source_text = (
        "the current incumbent implementation"
        if incumbent.source_round > 0
        else "the current blank/seed incumbent"
    )
    round_text = f" from round {incumbent.source_round}" if incumbent.source_round > 0 else ""
    return (
        f"You are starting from {source_text}{round_text}. "
        f"The incumbent's current best kept score is {incumbent.qps:.2f} QPS at recall {incumbent.recall:.4f}. "
        "Your objective is to improve QPS while keeping recall at or above 0.95. "
        "Only changes that beat the incumbent on the final benchmark will be kept. "
        "Build on the existing implementation rather than starting over."
    )


def _prepare_resumed_workdir(
    *,
    seed_dir: Path,
    work_dir: Path,
    system_prompt_path: Path,
    max_tool_calls: int,
    extra_user_message: str,
) -> None:
    _copy_tree(seed_dir, work_dir)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    session_context = {
        "tool_calls_used": 0,
        "tool_calls_total": max_tool_calls,
        "messages": [
            _chat_message("system", system_prompt),
            _chat_message("user", "Begin. Read the project files and start implementing."),
            _chat_message("user", extra_user_message),
        ],
        "last_benchmark": None,
        "best_benchmark": None,
        "call_log": [],
    }
    (work_dir / "session_context.json").write_text(
        json.dumps(session_context),
        encoding="utf-8",
    )


def _run_eval_round(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    work_dir: Path,
    results_dir: Path,
    api_key: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": args.model_name,
            "API_URL": args.base_url,
            "API_KEY": api_key,
            "MODEL_ID": args.model_id,
            "THINKING_MODE": args.thinking_mode,
            "REASONING_EFFORT": args.reasoning_effort,
            "API_INTERVAL_MS": str(args.api_interval_ms),
            "CPU_CORES": args.cpu_cores,
            "WORK_DIR": str(work_dir),
            "DATA_DIR": str((args.data_dir or (args.bench_repo / "data")).resolve()),
            "RESULTS_DIR": str(results_dir),
            "MAX_TOOL_CALLS": str(args.max_tool_calls),
        }
    )
    command = ["bash", str(args.bench_repo / "scripts" / "run_eval.sh")]
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=args.bench_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.round_timeout_seconds,
    )
    elapsed = time.time() - started
    (round_dir / "run_eval.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (round_dir / "run_eval.stderr.log").write_text(
        proc.stderr + f"\n[wall_clock_seconds]={elapsed:.2f}\n", encoding="utf-8"
    )
    return proc


def _load_eval_log(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "eval_log.json"
    if not path.exists():
        raise FileNotFoundError(f"expected eval log at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_fields(payload: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(payload, dict):
        return 0.0, 0.0
    return (
        float(payload.get("qps", 0.0) or 0.0),
        float(payload.get("recall", 0.0) or 0.0),
    )


def _resolve_base_vectors_for_data_dir(data_dir: Path, run_root: Path) -> Path:
    single = data_dir / "base_vectors.json"
    if single.exists():
        return single.resolve()

    shard_paths = sorted(data_dir.glob("base_vectors_*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            f"Could not find base_vectors.json or base_vectors_*.json under {data_dir}"
        )

    merged_path = run_root / "benchmark_inputs" / "base_vectors_merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        return merged_path.resolve()

    with merged_path.open("w", encoding="utf-8") as out:
        out.write("[")
        first = True
        for shard_path in shard_paths:
            shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard_payload, list):
                raise ValueError(f"expected JSON array in shard file {shard_path}")
            for row in shard_payload:
                if not first:
                    out.write(",")
                json.dump(row, out, separators=(",", ":"))
                first = False
        out.write("]\n")
    return merged_path.resolve()


def _prepare_eval_inputs(args: argparse.Namespace, run_root: Path, data_dir: Path) -> tuple[EvalInputs, EvalInputs]:
    if args.base_vectors is not None:
        base_vectors = args.base_vectors.resolve()
    elif data_dir == (args.bench_repo / "data").resolve():
        base_vectors = _resolve_base_vectors_file(args.bench_repo, run_root).resolve()
    else:
        base_vectors = _resolve_base_vectors_for_data_dir(data_dir, run_root)

    query_vectors = (args.query_vectors or (data_dir / "query_vectors.json")).resolve()
    ground_truth = (args.ground_truth or (data_dir / "ground_truth.json")).resolve()
    for path in (base_vectors, query_vectors, ground_truth):
        if not path.exists():
            raise FileNotFoundError(f"required benchmark data file not found: {path}")

    input_dir = run_root / "benchmark_inputs"
    proxy_inputs = EvalInputs(
        base_vectors=base_vectors,
        query_vectors=_maybe_make_subset(
            query_vectors,
            input_dir / f"query_vectors_proxy_{args.proxy_max_queries}.json",
            args.proxy_max_queries,
        ).resolve(),
        ground_truth=_maybe_make_subset(
            ground_truth,
            input_dir / f"ground_truth_proxy_{args.proxy_max_queries}.json",
            args.proxy_max_queries,
        ).resolve(),
    )
    strict_inputs = EvalInputs(
        base_vectors=base_vectors,
        query_vectors=_maybe_make_subset(
            query_vectors,
            input_dir / f"query_vectors_strict_{args.strict_max_queries}.json",
            args.strict_max_queries,
        ).resolve(),
        ground_truth=_maybe_make_subset(
            ground_truth,
            input_dir / f"ground_truth_strict_{args.strict_max_queries}.json",
            args.strict_max_queries,
        ).resolve(),
    )
    return proxy_inputs, strict_inputs


def _build_run_config(args: argparse.Namespace, strict_inputs: Any, benchmark_bin: Path) -> RunConfig:
    benchmark_dir = args.bench_repo / "benchmark"
    return RunConfig(
        bench_repo=args.bench_repo.resolve(),
        skeleton_dir=(args.run_root / "blank_seed").resolve(),
        benchmark_dir=benchmark_dir.resolve(),
        benchmark_bin=benchmark_bin.resolve(),
        server_bin_name=args.server_bin_name,
        server_url=args.server_url,
        server_port=_parse_server_port(args.server_url),
        cpu_cores=args.cpu_cores,
        build_timeout_seconds=args.build_timeout_seconds,
        benchmark_timeout_seconds=args.benchmark_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        proxy_inputs=strict_inputs,
        strict_inputs=strict_inputs,
        concurrency=args.concurrency,
        warmup=args.warmup,
        recall_threshold=args.recall_threshold,
        seed=args.seed,
        codex_executable="",
        codex_timeout_seconds=0,
        codex_sandbox="workspace-write",
        codex_model="",
        codex_oss=False,
        codex_local_provider="",
        modal_show_output=False,
    )


def _evaluate_final_workspace(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    round_dir: Path,
    strict_inputs: Any,
    benchmark_bin: Path,
) -> BenchEvalResult:
    config = _build_run_config(args, strict_inputs, benchmark_bin)
    candidate_files = {
        relpath: content.rstrip() + "\n"
        for relpath, content in _discover_editable_skeleton_files(work_dir).items()
    }
    return _evaluate_candidate(
        candidate_files=candidate_files,
        workspace_dir=round_dir / "final_eval_workspace",
        eval_dir=round_dir / "final_eval",
        config=config,
        inputs=strict_inputs,
    )


def _snapshot_candidate(blank_seed_dir: Path, workspace_dir: Path, incumbent_dir: Path) -> None:
    _copy_tree(blank_seed_dir, incumbent_dir)
    editable_files = _discover_editable_skeleton_files(workspace_dir)
    for relpath, content in editable_files.items():
        dest = incumbent_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    cargo_lock = workspace_dir / "Cargo.lock"
    if cargo_lock.exists():
        shutil.copy2(cargo_lock, incumbent_dir / "Cargo.lock")
    else:
        lock_dest = incumbent_dir / "Cargo.lock"
        if lock_dest.exists():
            lock_dest.unlink()


def _write_round_summary(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _round_row(outcome: RoundOutcome) -> dict[str, Any]:
    final_eval = outcome.final_eval
    return {
        "round": outcome.round_index,
        "seed_source": outcome.seed_source,
        "status": outcome.status,
        "promoted": str(outcome.promoted).lower(),
        "agent_best_qps": f"{outcome.agent_best_qps:.2f}",
        "agent_best_recall": f"{outcome.agent_best_recall:.4f}",
        "agent_last_qps": f"{outcome.agent_last_qps:.2f}",
        "agent_last_recall": f"{outcome.agent_last_recall:.4f}",
        "final_eval_valid": "" if final_eval is None else str(final_eval.valid).lower(),
        "final_eval_qps": "" if final_eval is None else f"{final_eval.qps:.2f}",
        "final_eval_recall": "" if final_eval is None else f"{final_eval.recall:.4f}",
        "final_eval_failure_type": "" if final_eval is None else (final_eval.failure_type or ""),
        "incumbent_qps_before": f"{outcome.incumbent_qps_before:.2f}",
        "incumbent_qps_after": f"{outcome.incumbent_qps_after:.2f}",
        "tool_calls_used": outcome.tool_calls_used,
        "tool_calls_total": outcome.tool_calls_total,
        "elapsed_secs": f"{outcome.elapsed_secs:.2f}",
        "work_dir": outcome.work_dir,
        "notes": outcome.notes,
    }


def _update_summary(
    *,
    run_root: Path,
    args: argparse.Namespace,
    incumbent: IncumbentState,
    completed_rounds: int,
) -> None:
    payload = {
        "bench_repo": str(args.bench_repo.resolve()),
        "run_root": str(run_root.resolve()),
        "model_name": args.model_name,
        "model_id": args.model_id,
        "base_url": args.base_url,
        "data_dir": str((args.data_dir or (args.bench_repo / "data")).resolve()),
        "thinking_mode": args.thinking_mode,
        "reasoning_effort": args.reasoning_effort,
        "rounds_requested": args.rounds,
        "rounds_completed": completed_rounds,
        "incumbent": asdict(incumbent),
        "updated_at": datetime.now().isoformat(),
    }
    _write_json(run_root / "summary.json", payload)


def main() -> int:
    args = parse_args()
    args.bench_repo = args.bench_repo.resolve()
    args.run_root = args.run_root.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    load_dotenv(DEFAULT_DOTENV_PATH)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("missing API key: pass --api-key or set OPENROUTER_API_KEY in the environment/.env")

    blank_seed_dir = args.run_root / "blank_seed"
    incumbent_dir = args.run_root / "incumbent"
    incumbent_state_path = args.run_root / "incumbent_state.json"
    results_path = args.run_root / "results.tsv"

    args.run_root.mkdir(parents=True, exist_ok=True)
    _ensure_blank_seed(args, blank_seed_dir)

    if incumbent_state_path.exists():
        if not args.resume:
            raise RuntimeError(f"run root already exists and --no-resume was provided: {args.run_root}")
        incumbent = _load_incumbent_state(incumbent_state_path)
        append_results = results_path.exists()
    else:
        incumbent = _initialize_incumbent(blank_seed_dir, incumbent_dir, incumbent_state_path)
        append_results = False

    data_dir = (args.data_dir or (args.bench_repo / "data")).resolve()
    _proxy_inputs, strict_inputs = _prepare_eval_inputs(args, args.run_root, data_dir)
    benchmark_bin = _ensure_benchmark_binary(
        benchmark_dir=args.bench_repo / "benchmark",
        benchmark_bin_name=args.benchmark_bin_name,
        timeout_seconds=args.build_timeout_seconds,
    )
    writer = _build_results_writer(results_path, append=append_results)
    try:
        start_round = _next_round_index(args.run_root)
        for round_index in range(start_round, args.rounds + 1):
            round_dir = args.run_root / f"round_{round_index:03d}"
            work_dir = round_dir / "workdir"
            results_dir = round_dir / args.results_dir_name
            round_dir.mkdir(parents=True, exist_ok=True)

            seed_source = (
                "blank_seed" if incumbent.source_round == 0 else f"incumbent_round_{incumbent.source_round:03d}"
            )
            _seed_round_skeleton(incumbent_dir, args.bench_repo / "skeleton")
            if round_index > 1:
                _prepare_resumed_workdir(
                    seed_dir=incumbent_dir,
                    work_dir=work_dir,
                    system_prompt_path=args.bench_repo / "agent" / "system_prompt.txt",
                    max_tool_calls=args.max_tool_calls,
                    extra_user_message=_build_handoff_message(incumbent),
                )

            round_started = time.time()
            final_eval: BenchEvalResult | None = None
            status = "discard"
            promoted = False
            tool_calls_used = 0
            tool_calls_total = args.max_tool_calls
            agent_best_qps = 0.0
            agent_best_recall = 0.0
            agent_last_qps = 0.0
            agent_last_recall = 0.0
            notes = ""
            incumbent_qps_before = incumbent.qps

            try:
                proc = _run_eval_round(
                    args=args,
                    round_dir=round_dir,
                    work_dir=work_dir,
                    results_dir=results_dir,
                    api_key=api_key,
                )
                if proc.returncode != 0:
                    status = "crash"
                    notes = f"run_eval.sh exited with code {proc.returncode}"
                else:
                    eval_log = _load_eval_log(work_dir)
                    _write_json(round_dir / "eval_log.copy.json", eval_log)
                    agent_best_qps, agent_best_recall = _benchmark_fields(eval_log.get("best_benchmark"))
                    agent_last_qps, agent_last_recall = _benchmark_fields(eval_log.get("last_benchmark"))
                    tool_calls_used = int(eval_log.get("tool_calls_used", 0) or 0)
                    tool_calls_total = int(eval_log.get("tool_calls_total", args.max_tool_calls) or args.max_tool_calls)
                    final_eval = _evaluate_final_workspace(
                        args=args,
                        work_dir=work_dir,
                        round_dir=round_dir,
                        strict_inputs=strict_inputs,
                        benchmark_bin=benchmark_bin,
                    )
                    _write_json(round_dir / "final_eval.json", asdict(final_eval))
                    if final_eval.valid and final_eval.qps > incumbent.qps:
                        _snapshot_candidate(blank_seed_dir, work_dir, incumbent_dir)
                        incumbent = IncumbentState(
                            qps=final_eval.qps,
                            recall=final_eval.recall,
                            valid=final_eval.valid,
                            source_round=round_index,
                            status="keep",
                            notes=f"Promoted from round {round_index}.",
                        )
                        _write_json(incumbent_state_path, asdict(incumbent))
                        status = "keep"
                        promoted = True
                        notes = (
                            f"Promoted on independent final evaluation: "
                            f"{final_eval.qps:.2f} QPS @ recall {final_eval.recall:.4f}"
                        )
                    else:
                        status = "discard"
                        if final_eval is None:
                            notes = "No final evaluation produced."
                        elif not final_eval.valid:
                            notes = f"Final evaluation invalid: {final_eval.failure_type or 'constraint_failed'}"
                        else:
                            notes = (
                                f"Final evaluation {final_eval.qps:.2f} QPS did not beat incumbent "
                                f"{incumbent.qps:.2f} QPS"
                            )
            except subprocess.TimeoutExpired:
                status = "crash"
                notes = f"Round timed out after {args.round_timeout_seconds}s"
            except Exception as exc:  # noqa: BLE001
                status = "crash"
                notes = f"{type(exc).__name__}: {exc}"
                (round_dir / "outer_error.txt").write_text(notes + "\n", encoding="utf-8")

            outcome = RoundOutcome(
                round_index=round_index,
                seed_source=seed_source,
                status=status,
                promoted=promoted,
                agent_best_qps=agent_best_qps,
                agent_best_recall=agent_best_recall,
                agent_last_qps=agent_last_qps,
                agent_last_recall=agent_last_recall,
                final_eval=final_eval,
                incumbent_qps_before=incumbent_qps_before,
                incumbent_qps_after=incumbent.qps,
                tool_calls_used=tool_calls_used,
                tool_calls_total=tool_calls_total,
                elapsed_secs=time.time() - round_started,
                work_dir=str(work_dir),
                notes=notes,
            )
            writer.writerow(_round_row(outcome))
            _flush_results_writer(writer)
            _write_round_summary(
                round_dir / "round_summary.json",
                {
                    "round": round_index,
                    "seed_source": seed_source,
                    "status": status,
                    "promoted": promoted,
                    "agent_best_qps": agent_best_qps,
                    "agent_best_recall": agent_best_recall,
                    "agent_last_qps": agent_last_qps,
                    "agent_last_recall": agent_last_recall,
                    "final_eval": None if final_eval is None else asdict(final_eval),
                    "incumbent": asdict(incumbent),
                    "elapsed_secs": time.time() - round_started,
                    "notes": notes,
                },
            )
            _update_summary(
                run_root=args.run_root,
                args=args,
                incumbent=incumbent,
                completed_rounds=round_index,
            )
            if status == "crash" and not args.continue_on_error:
                break
    finally:
        _close_results_writer(writer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
