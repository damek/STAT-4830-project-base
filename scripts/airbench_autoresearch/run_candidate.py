#!/usr/bin/env python3
"""Evaluate the current AirBench autoresearch candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import modal

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1] / "airbench_gepa"))
    from airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from modal_airbench import app, run_airbench_candidate
else:
    raise RuntimeError("run_candidate.py is intended to be executed as a script")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_CANDIDATE_PATH = Path(__file__).with_name("candidate.py")


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
        if not key:
            continue
        if os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def normalize_target_accuracy(raw_value: float) -> float:
    return raw_value / 100.0 if raw_value > 1.0 else raw_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--mode", choices=("proxy", "strict"), default="proxy")
    parser.add_argument("--target-accuracy", type=float, default=94.0)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=60 * 15)
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream Modal and candidate logs while the app is active.",
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the candidate preflight path before warmup/trials.",
    )
    return parser.parse_args()


def result_to_payload(result: Any, *, candidate_path: Path, mode: str) -> dict[str, Any]:
    side = result.as_side_info()
    solver_code = candidate_path.read_text(encoding="utf-8")
    return {
        "mode": mode,
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(solver_code.encode("utf-8")).hexdigest(),
        "valid": result.valid,
        "failure_type": result.failure_type,
        "message": result.message,
        "score": result.score,
        "mean_accuracy": result.mean_accuracy,
        "mean_time_seconds": result.mean_time_seconds,
        "trials": result.trials,
        "runtime_seconds": result.runtime_seconds,
        "meets_target": side.get("meets_target"),
        "accuracy_margin": side.get("accuracy_margin"),
        "actual_device_name": side.get("actual_device_name"),
        "requested_gpu": side.get("requested_gpu"),
        "remote_runtime_seconds": side.get("remote_runtime_seconds"),
        "gpu_mismatch_attempts": side.get("gpu_mismatch_attempts", 0),
        "side_info": side,
    }


def main() -> int:
    args = parse_args()
    loaded_keys = load_dotenv(DEFAULT_DOTENV_PATH)
    if loaded_keys:
        print(f"[setup] loaded env keys from {DEFAULT_DOTENV_PATH}: {', '.join(sorted(loaded_keys))}", file=sys.stderr)

    if not args.candidate_path.exists():
        raise FileNotFoundError(f"Candidate not found: {args.candidate_path}")

    trials = args.trials
    if trials is None:
        trials = 1 if args.mode == "proxy" else 5

    eval_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
        preflight=args.preflight,
    )
    if args.modal_show_output:
        eval_cfg = replace(eval_cfg, candidate_verbose=True, stream_subprocess_logs=True)

    solver_code = args.candidate_path.read_text(encoding="utf-8")

    def _run_once() -> dict[str, Any]:
        result = evaluate_solver_code(solver_code, eval_cfg, run_airbench_candidate)
        return result_to_payload(result, candidate_path=args.candidate_path, mode=args.mode)

    if args.modal_show_output:
        with modal.enable_output(), app.run():
            payload = _run_once()
    else:
        with app.run():
            payload = _run_once()

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
