#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _find_meta_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool_config(meta_dir: Path) -> dict[str, Any]:
    path = meta_dir / "tool_config.json"
    if not path.exists():
        raise SystemExit(f"tool config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid tool config payload: {path}")
    return payload


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_runtime_symbols(repo_root: Path):
    _ensure_repo_on_path(repo_root)
    from scripts.vector_db_bench.qwen3_meta.meta_harness_runtime import (  # type: ignore[import-not-found]
        BUILD_TIMEOUT_SECS,
        DEFAULT_BENCHMARK_BIN_NAME,
        RECALL_THRESHOLD,
        _build_project,
        _run_benchmark_like,
        _run_correctness_test_like,
        _run_profiling_like,
    )

    return {
        "BUILD_TIMEOUT_SECS": BUILD_TIMEOUT_SECS,
        "DEFAULT_BENCHMARK_BIN_NAME": DEFAULT_BENCHMARK_BIN_NAME,
        "RECALL_THRESHOLD": RECALL_THRESHOLD,
        "_build_project": _build_project,
        "_run_benchmark_like": _run_benchmark_like,
        "_run_correctness_test_like": _run_correctness_test_like,
        "_run_profiling_like": _run_profiling_like,
    }


def _ensure_benchmark_binary(bench_repo: Path, runtime: dict[str, Any]) -> Path:
    benchmark_dir = bench_repo / "benchmark"
    benchmark_bin = benchmark_dir / "target" / "release" / str(runtime["DEFAULT_BENCHMARK_BIN_NAME"])
    if benchmark_bin.exists():
        return benchmark_bin.resolve()
    proc = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        timeout=int(runtime["BUILD_TIMEOUT_SECS"]),
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-4000:])
    if not benchmark_bin.exists():
        raise RuntimeError(f"benchmark binary not found after build: {benchmark_bin}")
    return benchmark_bin.resolve()


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _error(message: str) -> dict[str, Any]:
    return {"type": "Error", "message": message}


def _progress_state(meta_dir: Path) -> dict[str, Any]:
    path = meta_dir / "progress_state.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _mainline_manifest(meta_dir: Path) -> dict[str, Any]:
    path = meta_dir / "mainline_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _tool_state_path(meta_dir: Path) -> Path:
    return meta_dir / "tool_state.json"


def _load_tool_state(meta_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _tool_state_path(meta_dir)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    payload = {
        "start_time": time.time(),
        "tool_calls_used": 0,
        "tool_calls_total": int(config.get("tool_calls_total", 2147483647) or 2147483647),
        "server_running": False,
        "last_benchmark": None,
        "best_benchmark": None,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _save_tool_state(meta_dir: Path, payload: dict[str, Any]) -> None:
    _tool_state_path(meta_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_tool(meta_dir: Path, state: dict[str, Any], result: dict[str, Any]) -> None:
    state["tool_calls_used"] = int(state.get("tool_calls_used", 0) or 0) + 1
    if result.get("type") == "RunBenchmark":
        state["last_benchmark"] = result
        prev = state.get("best_benchmark")
        if result.get("recall_passed"):
            if not isinstance(prev, dict) or float(result.get("qps", 0.0) or 0.0) > float(prev.get("qps", 0.0) or 0.0):
                state["best_benchmark"] = result
    _save_tool_state(meta_dir, state)


def main() -> int:
    meta_dir = _find_meta_dir()
    workspace = meta_dir.parent
    config = _load_tool_config(meta_dir)
    repo_root = Path(str(config.get("repo_root", ""))).resolve()
    bench_repo = Path(str(config.get("bench_repo", ""))).resolve()
    data_dir = Path(str(config.get("data_dir", ""))).resolve()
    cpu_cores = str(config.get("cpu_cores", "0-3"))
    runtime = _load_runtime_symbols(repo_root)
    tool_state = _load_tool_state(meta_dir, config)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build_project")
    sub.add_parser("run_correctness_test")

    bench = sub.add_parser("run_benchmark")
    bench.add_argument("--full", action="store_true", default=False)
    bench.add_argument("--max-queries", type=int, default=None)
    bench.add_argument("--concurrency", type=int, default=4)
    bench.add_argument("--warmup", type=int, default=100)

    prof = sub.add_parser("run_profiling")
    prof.add_argument("--duration", type=int, default=20)

    sub.add_parser("get_status")

    args = parser.parse_args()

    if args.cmd == "build_project":
        err = runtime["_build_project"](workspace, profiling=False)
        result = {
            "type": "BuildProject",
            "success": err is None,
            "message": "Build succeeded." if err is None else f"Build failed: {err}",
        }
        _record_tool(meta_dir, tool_state, result)
        _print(result)
        return 0

    if args.cmd == "run_correctness_test":
        try:
            benchmark_bin = _ensure_benchmark_binary(bench_repo, runtime)
            result = runtime["_run_correctness_test_like"](
                work_dir=workspace,
                benchmark_bin=benchmark_bin,
                data_dir=data_dir,
                cpu_cores=cpu_cores,
            )
        except Exception as exc:  # noqa: BLE001
            result = _error(str(exc))
        _record_tool(meta_dir, tool_state, result)
        _print(result)
        return 0

    if args.cmd == "run_benchmark":
        max_queries = 0 if args.full else (args.max_queries if args.max_queries is not None else 1000)
        try:
            benchmark_bin = _ensure_benchmark_binary(bench_repo, runtime)
            result = runtime["_run_benchmark_like"](
                work_dir=workspace,
                benchmark_bin=benchmark_bin,
                data_dir=data_dir,
                cpu_cores=cpu_cores,
                concurrency=int(args.concurrency),
                warmup=int(args.warmup),
                max_queries=int(max_queries),
                save_history=True,
            )
        except Exception as exc:  # noqa: BLE001
            result = _error(str(exc))
        _record_tool(meta_dir, tool_state, result)
        _print(result)
        return 0

    if args.cmd == "run_profiling":
        try:
            benchmark_bin = _ensure_benchmark_binary(bench_repo, runtime)
            result = runtime["_run_profiling_like"](
                work_dir=workspace,
                benchmark_bin=benchmark_bin,
                data_dir=data_dir,
                cpu_cores=cpu_cores,
                duration=int(args.duration),
            )
        except Exception as exc:  # noqa: BLE001
            result = _error(str(exc))
        _record_tool(meta_dir, tool_state, result)
        _print(result)
        return 0

    if args.cmd == "get_status":
        tool_state["tool_calls_used"] = int(tool_state.get("tool_calls_used", 0) or 0) + 1
        tool_calls_total = int(tool_state.get("tool_calls_total", 2147483647) or 2147483647)
        result = {
            "type": "GetStatus",
            "tool_calls_used": int(tool_state.get("tool_calls_used", 0) or 0),
            "tool_calls_remaining": max(0, tool_calls_total - int(tool_state.get("tool_calls_used", 0) or 0)),
            "tool_calls_total": tool_calls_total,
            "elapsed_time_secs": max(0.0, time.time() - float(tool_state.get("start_time", time.time()))),
            "server_running": False,
            "last_benchmark": tool_state.get("last_benchmark"),
            "best_benchmark": tool_state.get("best_benchmark"),
        }
        _save_tool_state(meta_dir, tool_state)
        _print(result)
        return 0

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
