#!/usr/bin/env python3
"""Optimize the AirBench 94-percent CIFAR-10 script with GEPA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import gepa.optimize_anything as oa
import modal

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from agent_team_proposer import build_agent_team_candidate_proposer
    from airbench_evaluator import AirbenchEvalConfig, AirbenchEvalResult, evaluate_solver_code
    from modal_airbench import app, run_airbench_candidate
else:
    from .agent_team_proposer import build_agent_team_candidate_proposer
    from .airbench_evaluator import AirbenchEvalConfig, AirbenchEvalResult, evaluate_solver_code
    from .modal_airbench import app, run_airbench_candidate


DEFAULT_SEED_PATH = Path(__file__).with_name("seeds") / "airbench94_baseline.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-solver-path",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="Seed candidate script to optimize.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/airbench/gepa_runs") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=94.0,
        help="Target CIFAR-10 TTA accuracy. Values >1 are treated as percentages.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of scored AirBench trials per evaluation.")
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=1,
        help="Warmup trials to run before scored trials to amortize compilation.",
    )
    parser.add_argument(
        "--verify-trials",
        type=int,
        default=3,
        help="Number of scored trials for the final best-candidate verification pass.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60 * 15)
    parser.add_argument("--max-metric-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="openai/gpt-5.4",
        help="LiteLLM model used for GEPA reflection.",
    )
    parser.add_argument(
        "--proposal-strategy",
        choices=("single", "team"),
        default="team",
        help="How GEPA proposes candidate updates. 'team' uses a coordinator/worker/reviewer workflow.",
    )
    parser.add_argument(
        "--team-rounds",
        type=int,
        default=2,
        help="Maximum internal coordinator/worker/reviewer rounds per proposal when using team strategy.",
    )
    parser.add_argument(
        "--history-aware-proposer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed structured prior evaluation history into the team proposer prompts.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="Number of recent evaluation rows to summarize for history-aware team prompts.",
    )
    parser.add_argument(
        "--refiner-model",
        type=str,
        default=None,
        help="Optional LiteLLM model for inner-loop refinement retries. Defaults to --reflection-model.",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=2,
        help="Maximum immediate refinement retries per evaluated candidate. Use 0 to disable.",
    )
    parser.add_argument(
        "--cache-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GEPA evaluator caching.",
    )
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show Modal app/image logs while the run is active.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate the seed candidate once and stop.",
    )
    return parser.parse_args()


def normalize_target_accuracy(raw_value: float) -> float:
    return raw_value / 100.0 if raw_value > 1.0 else raw_value


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
        existing_value = os.environ.get(key)
        if existing_value:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def ensure_reflection_auth(
    reflection_model: str,
    *,
    dry_run: bool,
    refiner_model: str | None = None,
    max_refinements: int = 0,
) -> None:
    if dry_run:
        return
    required_models = [reflection_model]
    if max_refinements > 0 and refiner_model:
        required_models.append(refiner_model)
    if any(model.startswith("openai/") for model in required_models) and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for reflection model "
            f"{reflection_model!r}. Put it in the shell environment or .env before launching the run."
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def write_progress_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "metric_call",
        "candidate_kind",
        "same_as_seed",
        "score",
        "best_score_so_far",
        "mean_accuracy",
        "best_accuracy_so_far",
        "mean_time_seconds",
        "best_target_mean_time_so_far",
        "meets_target",
        "is_new_best_score",
        "is_first_target_hit",
        "elapsed_wall_clock_seconds",
        "runtime_seconds",
        "remote_runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def load_seed_solver(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Seed solver not found: {path}")
    return path.read_text(encoding="utf-8")


def write_candidate_snapshot(
    run_dir: Path,
    *,
    metric_call: int,
    candidate_kind: str,
    solver_code: str,
    eval_result: AirbenchEvalResult,
) -> None:
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    stem = f"metric_{metric_call:04d}_{candidate_kind}"
    (candidate_dir / f"{stem}.py").write_text(solver_code, encoding="utf-8")
    write_json(candidate_dir / f"{stem}.json", eval_result.as_side_info())


def build_objective(target_accuracy: float) -> str:
    return (
        "Optimize `solver_code` to train CIFAR-10 on a single NVIDIA A100-40GB GPU "
        f"to at least {100.0 * target_accuracy:.2f}% test-time-augmented accuracy as quickly as possible."
    )


def build_background(target_accuracy: float, trials: int, warmup_trials: int) -> str:
    return (
        "Runtime environment:\n"
        "- Evaluator runs the script remotely on Modal using Python 3.11, torch 2.4.1, torchvision 0.19.1.\n"
        "- Hardware target is one NVIDIA A100-40GB GPU.\n"
        "- CIFAR-10 is cached on a Modal Volume mounted at /vol/cifar10.\n\n"
        "Candidate format:\n"
        "- Candidate is full `solver_code`, not a config dict.\n"
        "- You may rewrite any part of the program: model, optimizer, schedule, augmentation, compilation, or evaluation logic.\n"
        "- Return a complete standalone Python script, not a diff or patch.\n"
        "- The script must remain executable as-is under the evaluator contract below.\n\n"
        "Output contract:\n"
        "- Script must accept CLI flags: --data-dir, --trials, --warmup-trials, --target-accuracy, --json-only, --preflight.\n"
        "- If --preflight is passed, run a cheap smoke-check before full warmup/trials to catch runtime or compile issues early.\n"
        "- Final stdout line must be a JSON object containing at least: mean_accuracy, mean_time_seconds, trials.\n"
        "- mean_accuracy may be reported as fraction or percentage; evaluator normalizes it.\n"
        "- mean_time_seconds must be positive.\n\n"
        "Scoring:\n"
        f"- Target accuracy is {100.0 * target_accuracy:.2f}%.\n"
        "- Any candidate that meets the target beats every candidate that misses it.\n"
        "- Among target-meeting candidates, lower mean_time_seconds is better.\n\n"
        "Evaluation protocol:\n"
        f"- Each score uses {trials} measured trial(s) after {warmup_trials} warmup trial(s).\n"
        "- Do not hardcode outputs, skip training, or fake metrics.\n"
        "- Prefer meaningful improvements to architecture, optimizer, augmentation, batch sizing, compile behavior, "
        "or schedule.\n"
        "- Preserve dtype consistency across normalized inputs, model weights, and biases, especially for compiled half-precision convolutions.\n"
        "- Avoid adding non-standard dependencies.\n"
        "- Preserve the CLI/JSON contract even if you substantially rewrite the program."
    )


def summarize_result(result: AirbenchEvalResult, metric_call: int, candidate_kind: str) -> dict[str, Any]:
    row = {
        "metric_call": metric_call,
        "candidate_kind": candidate_kind,
        "score": result.score,
        "valid": result.valid,
        "failure_type": result.failure_type,
        "message": result.message,
        "runtime_seconds": result.runtime_seconds,
        "mean_accuracy": result.mean_accuracy,
        "mean_time_seconds": result.mean_time_seconds,
        "trials": result.trials,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    if "remote_runtime_seconds" in result.extra:
        row["remote_runtime_seconds"] = result.extra["remote_runtime_seconds"]
    if "actual_device_name" in result.extra:
        row["actual_device_name"] = result.extra["actual_device_name"]
    if "requested_gpu" in result.extra:
        row["requested_gpu"] = result.extra["requested_gpu"]
    if "gpu_mismatch_attempts" in result.extra:
        row["gpu_mismatch_attempts"] = result.extra["gpu_mismatch_attempts"]
    if "meets_target" in result.extra:
        row["meets_target"] = result.extra["meets_target"]
    if "accuracy_margin" in result.extra:
        row["accuracy_margin"] = result.extra["accuracy_margin"]
    return row


def classify_candidate_kind(
    *,
    metric_call: int,
    solver_code: str,
    seed_solver_code: str,
    known_hashes: set[str],
) -> tuple[str, str, bool]:
    candidate_hash = hashlib.sha256(solver_code.encode("utf-8")).hexdigest()
    same_as_seed = solver_code == seed_solver_code
    if metric_call == 1:
        candidate_kind = "seed"
    elif same_as_seed and metric_call == 2:
        candidate_kind = "base_recheck"
    elif candidate_hash in known_hashes:
        candidate_kind = "repeat"
    else:
        candidate_kind = "gepa"
    known_hashes.add(candidate_hash)
    return candidate_kind, candidate_hash, same_as_seed


def print_eval_summary(prefix: str, result: AirbenchEvalResult) -> None:
    remote_runtime = result.extra.get("remote_runtime_seconds")
    remote_runtime_str = "n/a" if remote_runtime is None else f"{float(remote_runtime):.2f}s"
    accuracy_str = "n/a" if result.mean_accuracy is None else f"{100.0 * float(result.mean_accuracy):.3f}%"
    bench_time_str = "n/a" if result.mean_time_seconds is None else f"{float(result.mean_time_seconds):.4f}s"
    actual_device_name = result.extra.get("actual_device_name")
    mismatch_attempts = int(result.extra.get("gpu_mismatch_attempts", 0) or 0)
    device_str = "" if not actual_device_name else f" device={actual_device_name}"
    retry_str = "" if mismatch_attempts <= 0 else f" gpu_retries={mismatch_attempts}"
    print(
        f"{prefix} valid={result.valid} score={result.score:.6f} "
        f"accuracy={accuracy_str} benchmark_time={bench_time_str} "
        f"eval_wall={result.runtime_seconds:.2f}s remote_wall={remote_runtime_str} "
        f"failure={result.failure_type}{device_str}{retry_str}"
    )


def annotate_progress(
    row: dict[str, Any],
    eval_history: list[dict[str, Any]],
    *,
    elapsed_wall_clock_seconds: float,
) -> dict[str, Any]:
    best_score_so_far = row["score"]
    best_accuracy_so_far = row["mean_accuracy"]
    best_target_mean_time_so_far = row["mean_time_seconds"] if row.get("meets_target") else None
    previous_target_seen = False

    if eval_history:
        last = eval_history[-1]
        best_score_so_far = max(float(last["best_score_so_far"]), float(row["score"]))
        prev_best_accuracy = last.get("best_accuracy_so_far")
        if prev_best_accuracy is None:
            best_accuracy_so_far = row["mean_accuracy"]
        elif row["mean_accuracy"] is None:
            best_accuracy_so_far = prev_best_accuracy
        else:
            best_accuracy_so_far = max(float(prev_best_accuracy), float(row["mean_accuracy"]))

        prev_best_target_time = last.get("best_target_mean_time_so_far")
        if prev_best_target_time is not None:
            best_target_mean_time_so_far = float(prev_best_target_time)
        if row.get("meets_target") and row.get("mean_time_seconds") is not None:
            curr_time = float(row["mean_time_seconds"])
            if best_target_mean_time_so_far is None or curr_time < best_target_mean_time_so_far:
                best_target_mean_time_so_far = curr_time
        previous_target_seen = any(bool(prev.get("meets_target")) for prev in eval_history)

    row["elapsed_wall_clock_seconds"] = elapsed_wall_clock_seconds
    row["best_score_so_far"] = best_score_so_far
    row["best_accuracy_so_far"] = best_accuracy_so_far
    row["best_target_mean_time_so_far"] = best_target_mean_time_so_far
    row["is_new_best_score"] = (
        not eval_history or float(row["score"]) > float(eval_history[-1]["best_score_so_far"])
    )
    row["is_first_target_hit"] = bool(row.get("meets_target")) and not previous_target_seen
    return row


def build_milestones(eval_history: list[dict[str, Any]]) -> dict[str, Any]:
    milestones: dict[str, Any] = {
        "total_metric_calls_observed": len(eval_history),
        "total_wall_clock_seconds": eval_history[-1]["elapsed_wall_clock_seconds"] if eval_history else 0.0,
    }
    if not eval_history:
        return milestones

    best_score_row = max(eval_history, key=lambda row: float(row["score"]))
    best_accuracy_row = max(
        (row for row in eval_history if row.get("mean_accuracy") is not None),
        key=lambda row: float(row["mean_accuracy"]),
        default=None,
    )
    first_target_row = next((row for row in eval_history if row.get("meets_target")), None)
    fastest_target_row = min(
        (row for row in eval_history if row.get("meets_target") and row.get("mean_time_seconds") is not None),
        key=lambda row: float(row["mean_time_seconds"]),
        default=None,
    )

    milestones.update(
        {
            "best_score_metric_call": best_score_row["metric_call"],
            "best_score_elapsed_wall_clock_seconds": best_score_row["elapsed_wall_clock_seconds"],
            "best_score": best_score_row["score"],
        }
    )
    if best_accuracy_row is not None:
        milestones.update(
            {
                "best_accuracy_metric_call": best_accuracy_row["metric_call"],
                "best_accuracy_elapsed_wall_clock_seconds": best_accuracy_row["elapsed_wall_clock_seconds"],
                "best_accuracy": best_accuracy_row["mean_accuracy"],
            }
        )
    if first_target_row is not None:
        milestones.update(
            {
                "first_target_metric_call": first_target_row["metric_call"],
                "first_target_elapsed_wall_clock_seconds": first_target_row["elapsed_wall_clock_seconds"],
                "first_target_score": first_target_row["score"],
                "first_target_mean_accuracy": first_target_row["mean_accuracy"],
                "first_target_mean_time_seconds": first_target_row["mean_time_seconds"],
            }
        )
    else:
        milestones.update(
            {
                "first_target_metric_call": None,
                "first_target_elapsed_wall_clock_seconds": None,
                "first_target_score": None,
                "first_target_mean_accuracy": None,
                "first_target_mean_time_seconds": None,
            }
        )
    if fastest_target_row is not None:
        milestones.update(
            {
                "fastest_target_metric_call": fastest_target_row["metric_call"],
                "fastest_target_elapsed_wall_clock_seconds": fastest_target_row["elapsed_wall_clock_seconds"],
                "fastest_target_mean_accuracy": fastest_target_row["mean_accuracy"],
                "fastest_target_mean_time_seconds": fastest_target_row["mean_time_seconds"],
            }
        )
    else:
        milestones.update(
            {
                "fastest_target_metric_call": None,
                "fastest_target_elapsed_wall_clock_seconds": None,
                "fastest_target_mean_accuracy": None,
                "fastest_target_mean_time_seconds": None,
            }
        )
    return milestones


def get_best_solver_code(result_obj: Any) -> str:
    best = result_obj.best_candidate
    if isinstance(best, str):
        return best
    if isinstance(best, dict):
        if "solver_code" in best and isinstance(best["solver_code"], str):
            return best["solver_code"]
        for value in best.values():
            if isinstance(value, str):
                return value
    raise ValueError("Could not resolve solver code from result.best_candidate.")


def _run_with_modal_context(args: argparse.Namespace, eval_cfg: AirbenchEvalConfig) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.perf_counter()

    seed_solver_code = load_seed_solver(args.seed_solver_path)
    objective = build_objective(eval_cfg.normalized_target_accuracy())
    background = build_background(
        target_accuracy=eval_cfg.normalized_target_accuracy(),
        trials=eval_cfg.trials,
        warmup_trials=eval_cfg.warmup_trials,
    )
    refiner_model = args.refiner_model or args.reflection_model

    write_json(
        args.run_dir / "run_config.json",
        {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "args": vars(args),
            "eval_config": asdict(eval_cfg),
            "objective": objective,
            "background": background,
            "refiner_model": refiner_model,
            "max_refinements": args.max_refinements,
            "history_aware_proposer": args.history_aware_proposer,
            "history_window": args.history_window,
        },
    )
    (args.run_dir / "seed_solver.py").write_text(seed_solver_code, encoding="utf-8")

    eval_history: list[dict[str, Any]] = []
    known_candidate_hashes: set[str] = set()

    print("[seed] evaluating seed candidate on Modal")
    seed_eval = evaluate_solver_code(seed_solver_code, eval_cfg, run_airbench_candidate)
    print_eval_summary("[seed] finished", seed_eval)
    seed_kind, seed_hash, seed_same_as_seed = classify_candidate_kind(
        metric_call=1,
        solver_code=seed_solver_code,
        seed_solver_code=seed_solver_code,
        known_hashes=known_candidate_hashes,
    )
    seed_row = summarize_result(seed_eval, metric_call=1, candidate_kind=seed_kind)
    seed_row["candidate_sha256"] = seed_hash
    seed_row["same_as_seed"] = seed_same_as_seed
    write_candidate_snapshot(
        args.run_dir,
        metric_call=1,
        candidate_kind=seed_kind,
        solver_code=seed_solver_code,
        eval_result=seed_eval,
    )
    eval_history.append(
        annotate_progress(
            seed_row,
            eval_history=[],
            elapsed_wall_clock_seconds=time.perf_counter() - run_started_at,
        )
    )
    write_json(args.run_dir / "seed_eval.json", seed_eval.as_side_info())

    if args.dry_run:
        write_jsonl(args.run_dir / "eval_log.jsonl", eval_history)
        write_progress_csv(args.run_dir / "progress_curve.csv", eval_history)
        write_json(args.run_dir / "milestones.json", build_milestones(eval_history))
        print(
            f"[dry-run] seed accuracy={seed_eval.mean_accuracy} "
            f"time={seed_eval.mean_time_seconds} score={seed_eval.score}"
        )
        print(f"[dry-run] artifacts: {args.run_dir}")
        return 0

    metric_call_counter = len(eval_history)

    def evaluator(candidate: dict[str, str], opt_state: oa.OptimizationState | None = None):
        nonlocal metric_call_counter
        metric_call_counter += 1
        solver_code = candidate.get("solver_code", "")
        candidate_kind, candidate_hash, same_as_seed = classify_candidate_kind(
            metric_call=metric_call_counter,
            solver_code=solver_code,
            seed_solver_code=seed_solver_code,
            known_hashes=known_candidate_hashes,
        )
        print(
            f"[eval {metric_call_counter:03d}] submitting candidate to Modal "
            f"kind={candidate_kind} same_as_seed={same_as_seed}"
        )
        eval_result = evaluate_solver_code(solver_code, eval_cfg, run_airbench_candidate)
        print_eval_summary(f"[eval {metric_call_counter:03d}] finished", eval_result)
        row = summarize_result(eval_result, metric_call_counter, candidate_kind=candidate_kind)
        row["candidate_sha256"] = candidate_hash
        row["same_as_seed"] = same_as_seed
        write_candidate_snapshot(
            args.run_dir,
            metric_call=metric_call_counter,
            candidate_kind=candidate_kind,
            solver_code=solver_code,
            eval_result=eval_result,
        )
        if opt_state is not None:
            row["best_example_evals_seen"] = len(opt_state.best_example_evals)
        row = annotate_progress(
            row,
            eval_history=eval_history,
            elapsed_wall_clock_seconds=time.perf_counter() - run_started_at,
        )
        eval_history.append(row)
        oa.log(
            "metric_call=", metric_call_counter,
            "score=", eval_result.score,
            "mean_accuracy=", eval_result.mean_accuracy,
            "mean_time_seconds=", eval_result.mean_time_seconds,
            "failure_type=", eval_result.failure_type,
        )
        return eval_result.score, eval_result.as_side_info()

    config = oa.GEPAConfig(
        engine=oa.EngineConfig(
            run_dir=str(args.run_dir),
            seed=args.seed,
            display_progress_bar=True,
            use_cloudpickle=False,
            max_metric_calls=args.max_metric_calls,
            cache_evaluation=args.cache_evaluation,
            capture_stdio=False,
        ),
        reflection=oa.ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_minibatch_size=1,
            module_selector="all" if args.proposal_strategy == "team" else "round_robin",
            custom_candidate_proposer=(
                build_agent_team_candidate_proposer(
                    lm=oa.make_litellm_lm(args.reflection_model),
                    objective=objective,
                    background=background,
                    eval_history_ref=eval_history,
                    run_dir=args.run_dir,
                    max_rounds=args.team_rounds,
                    history_window=args.history_window,
                    history_enabled=args.history_aware_proposer,
                )
                if args.proposal_strategy == "team"
                else None
            ),
        ),
        refiner=(
            oa.RefinerConfig(
                refiner_lm=refiner_model,
                max_refinements=args.max_refinements,
            )
            if args.max_refinements > 0
            else None
        ),
    )

    result = oa.optimize_anything(
        seed_candidate={"solver_code": seed_solver_code},
        evaluator=evaluator,
        objective=objective,
        background=background,
        config=config,
    )

    best_solver_code = get_best_solver_code(result)
    verify_cfg = replace(eval_cfg, trials=args.verify_trials)
    best_verified = evaluate_solver_code(best_solver_code, verify_cfg, run_airbench_candidate)

    (args.run_dir / "best_solver.py").write_text(best_solver_code, encoding="utf-8")
    write_json(args.run_dir / "best_verified_eval.json", best_verified.as_side_info())
    write_json(args.run_dir / "gepa_result.json", result.to_dict())
    write_jsonl(args.run_dir / "eval_log.jsonl", eval_history)
    write_progress_csv(args.run_dir / "progress_curve.csv", eval_history)
    milestones = build_milestones(eval_history)
    write_json(args.run_dir / "milestones.json", milestones)

    summary = {
        "num_candidates": result.num_candidates,
        "best_idx": result.best_idx,
        "best_aggregate_score": result.val_aggregate_scores[result.best_idx],
        "best_verified_score": best_verified.score,
        "best_verified_valid": best_verified.valid,
        "best_verified_mean_accuracy": best_verified.mean_accuracy,
        "best_verified_mean_time_seconds": best_verified.mean_time_seconds,
        "reflection_model": args.reflection_model,
        "proposal_strategy": args.proposal_strategy,
        "team_rounds": args.team_rounds,
        "history_aware_proposer": args.history_aware_proposer,
        "history_window": args.history_window,
        "refiner_model": refiner_model if args.max_refinements > 0 else None,
        "max_refinements": args.max_refinements,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "total_evaluated_candidates_observed": len(eval_history),
        "num_distinct_candidate_programs_observed": len({row["candidate_sha256"] for row in eval_history}),
        "num_true_gepa_candidates_observed": sum(1 for row in eval_history if row["candidate_kind"] == "gepa"),
        "num_repeated_candidates_observed": sum(
            1 for row in eval_history if row["candidate_kind"] in {"base_recheck", "repeat"}
        ),
        "total_wall_clock_seconds": time.perf_counter() - run_started_at,
        "run_dir": str(args.run_dir),
    }
    summary.update(milestones)
    write_json(args.run_dir / "summary.json", summary)

    print("[gepa] optimization complete")
    print(f"[gepa] best aggregate score      : {summary['best_aggregate_score']:.10f}")
    print(f"[gepa] best verified mean acc   : {summary['best_verified_mean_accuracy']}")
    print(f"[gepa] best verified mean time  : {summary['best_verified_mean_time_seconds']}")
    print(f"[gepa] best verified valid      : {summary['best_verified_valid']}")
    print(f"[gepa] observed evaluator calls : {summary['total_evaluated_candidates_observed']}")
    if args.max_refinements > 0:
        print(
            f"[gepa] inner refiner           : enabled "
            f"(model={refiner_model}, max_refinements={args.max_refinements})"
        )
    print(f"[gepa] proposal strategy        : {args.proposal_strategy}")
    if args.proposal_strategy == "team":
        print(
            f"[gepa] history-aware proposer  : {args.history_aware_proposer} "
            f"(window={args.history_window})"
        )
    print(f"[gepa] run artifacts            : {args.run_dir}")
    return 0


def main() -> int:
    args = parse_args()
    loaded_env_keys = load_dotenv(DEFAULT_DOTENV_PATH)
    if loaded_env_keys:
        print(f"[setup] loaded env keys from {DEFAULT_DOTENV_PATH}: {', '.join(sorted(loaded_env_keys))}")
    ensure_reflection_auth(
        args.reflection_model,
        dry_run=args.dry_run,
        refiner_model=args.refiner_model or args.reflection_model,
        max_refinements=args.max_refinements,
    )
    eval_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=args.trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
    )
    if args.modal_show_output:
        eval_cfg = replace(eval_cfg, candidate_verbose=True, stream_subprocess_logs=True)

    if args.modal_show_output:
        with modal.enable_output(), app.run():
            return _run_with_modal_context(args, eval_cfg)
    with app.run():
        return _run_with_modal_context(args, eval_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
