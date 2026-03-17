#!/usr/bin/env python3
"""Minimal autoresearch-style edit/evaluate loop for AirBench candidate.py."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import modal

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "airbench_gepa"))
    from airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from modal_airbench import app, run_airbench_candidate

    sys.path.append(str(Path(__file__).resolve().parent))
    from loop_core import (
        AutoresearchLoopConfig,
        ensure_auth,
        load_dotenv,
        normalize_target_accuracy,
        run_autoresearch_loop,
    )
else:
    from ..airbench_gepa.airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from ..airbench_gepa.modal_airbench import app, run_airbench_candidate
    from .loop_core import (
        AutoresearchLoopConfig,
        ensure_auth,
        load_dotenv,
        normalize_target_accuracy,
        run_autoresearch_loop,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "airbench" / "autoresearch_runs"
DEFAULT_CANDIDATE_PATH = Path(__file__).with_name("candidate.py")
DEFAULT_PROGRAM_PATH = Path(__file__).with_name("program.md")
DEFAULT_MEMORY_PATH = Path(__file__).with_name("memory.md")
DEFAULT_RECORD_PATH = Path(__file__).with_name("incumbent_record.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--program-path", type=Path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--incumbent-record-path", type=Path, default=DEFAULT_RECORD_PATH)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--model", type=str, default="gemini/gemini-3.1-flash-lite-preview")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=94.0)
    parser.add_argument("--proxy-trials", type=int, default=1)
    parser.add_argument("--strict-trials", type=int, default=5)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=60 * 15)
    parser.add_argument(
        "--final-strict-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one final strict evaluation on the final incumbent.",
    )
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream Modal and candidate logs while the app is active.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded_keys = load_dotenv(DEFAULT_DOTENV_PATH)
    if loaded_keys:
        print(f"[setup] loaded env keys from {DEFAULT_DOTENV_PATH}: {', '.join(sorted(loaded_keys))}")
    ensure_auth(args.model)

    if not args.candidate_path.exists():
        raise FileNotFoundError(f"Candidate not found: {args.candidate_path}")
    if not args.program_path.exists():
        raise FileNotFoundError(f"Program instructions not found: {args.program_path}")
    if not args.memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {args.memory_path}")
    if not args.incumbent_record_path.exists():
        raise FileNotFoundError(f"Incumbent record not found: {args.incumbent_record_path}")

    proxy_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=args.proxy_trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
        preflight=True,
    )
    strict_cfg = replace(proxy_cfg, trials=args.strict_trials)
    if args.modal_show_output:
        proxy_cfg = replace(proxy_cfg, candidate_verbose=True, stream_subprocess_logs=True)
        strict_cfg = replace(strict_cfg, candidate_verbose=True, stream_subprocess_logs=True)

    loop_cfg = AutoresearchLoopConfig(
        candidate_path=args.candidate_path,
        program_path=args.program_path,
        memory_path=args.memory_path,
        incumbent_record_path=args.incumbent_record_path,
        run_dir=args.run_dir,
        model=args.model,
        max_attempts=args.max_attempts,
        final_strict_eval=args.final_strict_eval,
    )

    def evaluate_proxy(code: str):
        return evaluate_solver_code(code, proxy_cfg, run_airbench_candidate)

    def evaluate_strict(code: str):
        return evaluate_solver_code(code, strict_cfg, run_airbench_candidate)

    if args.modal_show_output:
        with modal.enable_output(), app.run():
            return run_autoresearch_loop(loop_cfg, evaluate_proxy, evaluate_strict)
    with app.run():
        return run_autoresearch_loop(loop_cfg, evaluate_proxy, evaluate_strict)


if __name__ == "__main__":
    raise SystemExit(main())
