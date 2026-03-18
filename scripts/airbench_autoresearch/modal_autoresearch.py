#!/usr/bin/env python3
"""Long-lived Modal-hosted autoresearch loop for AirBench."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from functools import lru_cache
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import modal

THIS_FILE = Path(__file__).resolve()
_parents = THIS_FILE.parents
REPO_ROOT = _parents[2] if len(_parents) >= 3 else THIS_FILE.parent
LOCAL_GEPA_DIR = REPO_ROOT / "scripts" / "airbench_gepa"
LOCAL_AUTORESEARCH_DIR = REPO_ROOT / "scripts" / "airbench_autoresearch"
REMOTE_GEPA_DIR = Path("/root/airbench_gepa")
REMOTE_AUTORESEARCH_DIR = Path("/root/airbench_autoresearch")

if __package__ in (None, ""):
    import sys

    for path in (
        LOCAL_GEPA_DIR,
        LOCAL_AUTORESEARCH_DIR,
        REMOTE_GEPA_DIR,
        REMOTE_AUTORESEARCH_DIR,
    ):
        if path.exists():
            sys.path.append(str(path))

    from airbench_evaluator import (
        AirbenchEvalConfig,
        build_script_args,
        eval_result_from_remote_response,
        evaluate_solver_code,
    )
    from modal_airbench import (
        APP_NAME as AIRBENCH_APP_NAME,
        DEFAULT_GPU,
        REMOTE_DATA_DIR,
        _run_airbench_candidate_impl,
        cifar_volume,
        image as base_image,
    )

    from loop_core import (
        AutoresearchLoopConfig,
        WorkerBrief,
        build_results_writer,
        close_results_writer,
        ensure_auth,
        eval_row,
        is_better,
        is_infra_failure,
        load_dotenv,
        load_incumbent_record,
        load_results_rows,
        normalize_target_accuracy,
        propose_candidate,
        propose_strategy_update,
        propose_worker_briefs,
        serialize_incumbent_record,
        text_sha256,
        update_memory,
        validate_candidate_contract,
        write_json,
        run_meta_autoresearch_loop,
    )
else:
    from ..airbench_gepa.airbench_evaluator import (
        AirbenchEvalConfig,
        build_script_args,
        eval_result_from_remote_response,
        evaluate_solver_code,
    )
    from ..airbench_gepa.modal_airbench import (
        APP_NAME as AIRBENCH_APP_NAME,
        DEFAULT_GPU,
        REMOTE_DATA_DIR,
        _run_airbench_candidate_impl,
        cifar_volume,
        image as base_image,
    )
    from .loop_core import (
        AutoresearchLoopConfig,
        WorkerBrief,
        build_results_writer,
        close_results_writer,
        ensure_auth,
        eval_row,
        is_better,
        is_infra_failure,
        load_dotenv,
        load_incumbent_record,
        load_results_rows,
        normalize_target_accuracy,
        propose_candidate,
        propose_strategy_update,
        propose_worker_briefs,
        serialize_incumbent_record,
        text_sha256,
        update_memory,
        validate_candidate_contract,
        write_json,
        run_meta_autoresearch_loop,
    )

DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "airbench" / "autoresearch_runs"
DEFAULT_CANDIDATE_PATH = Path(__file__).with_name("candidate.py")
DEFAULT_PROGRAM_PATH = Path(__file__).with_name("program.md")
DEFAULT_STRATEGY_PATH = Path(__file__).with_name("strategy.md")
DEFAULT_MEMORY_PATH = Path(__file__).with_name("memory.md")
DEFAULT_RECORD_PATH = Path(__file__).with_name("incumbent_record.json")
REMOTE_RUN_ROOT = "/vol/autoresearch_runs"
APP_NAME = "airbench-autoresearch"


def _build_secret() -> modal.Secret | None:
    load_dotenv(DEFAULT_DOTENV_PATH)
    payload = {
        key: os.environ[key]
        for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")
        if os.environ.get(key)
    }
    if not payload:
        return None
    return modal.Secret.from_dict(payload)


app = modal.App(APP_NAME)
llm_secret = _build_secret()
autoresearch_volume = modal.Volume.from_name("airbench-autoresearch-runs", create_if_missing=True, version=2)
remote_image = (
    base_image
    .pip_install("litellm")
    .add_local_dir(LOCAL_GEPA_DIR, str(REMOTE_GEPA_DIR))
    .add_local_dir(LOCAL_AUTORESEARCH_DIR, str(REMOTE_AUTORESEARCH_DIR))
)


def _secrets_list() -> list[modal.Secret]:
    return [llm_secret] if llm_secret is not None else []


class _InlineRunner:
    def remote(
        self,
        solver_code: str,
        script_args: list[str],
        timeout_seconds: int,
        print_subprocess_logs: bool = False,
    ) -> dict[str, Any]:
        return _run_airbench_candidate_impl(
            solver_code,
            script_args,
            timeout_seconds=timeout_seconds,
            print_subprocess_logs=print_subprocess_logs,
            requested_gpu=DEFAULT_GPU,
        )


def _spawn_remote_eval(
    *,
    solver_code: str,
    config: AirbenchEvalConfig,
) -> modal.functions.FunctionCall:
    return _remote_eval_function().spawn(
        solver_code=solver_code,
        script_args=build_script_args(config),
        timeout_seconds=config.timeout_seconds,
        print_subprocess_logs=False,
    )


@lru_cache(maxsize=1)
def _remote_eval_function() -> modal.Function:
    # The batch coordinator runs in a different Modal app from the GPU runner, so
    # it must resolve the runner by deployed app/function name instead of using
    # the imported local function object.
    return modal.Function.from_name(AIRBENCH_APP_NAME, "run_airbench_candidate")


def _evaluate_candidates_parallel(
    items: list[tuple[int, str, str]],
    config: AirbenchEvalConfig,
) -> dict[int, Any]:
    # Each candidate runs in its own Modal GPU function. This keeps per-candidate
    # timing isolated while still letting the coordinator batch work in parallel.
    results: dict[int, Any] = {}
    pending: dict[int, tuple[str, modal.functions.FunctionCall, int, float]] = {}
    for attempt, _candidate_sha, solver_code in items:
        pending[attempt] = (
            solver_code,
            _spawn_remote_eval(solver_code=solver_code, config=config),
            0,
            time.time(),
        )

    while pending:
        completed_attempts: list[int] = []
        for attempt, (solver_code, function_call, mismatch_attempts, started_at) in list(pending.items()):
            try:
                remote_result = function_call.get(timeout=config.timeout_seconds + 120)
                result = eval_result_from_remote_response(
                    remote_result,
                    config=config,
                    elapsed=time.time() - started_at,
                    mismatch_attempts=mismatch_attempts,
                )
            except Exception as exc:
                result = eval_result_from_remote_response(
                    None,
                    config=config,
                    elapsed=time.time() - started_at,
                    last_remote_exception=exc,
                    mismatch_attempts=mismatch_attempts,
                )

            if result.failure_type == "gpu_mismatch" and mismatch_attempts < config.gpu_mismatch_retries:
                pending[attempt] = (
                    solver_code,
                    _spawn_remote_eval(solver_code=solver_code, config=config),
                    mismatch_attempts + 1,
                    time.time(),
                )
                continue

            results[attempt] = result
            completed_attempts.append(attempt)

        for attempt in completed_attempts:
            pending.pop(attempt, None)

    return results


def _proposal_worker(
    *,
    model: str,
    program_text: str,
    strategy_text: str,
    memory_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    worker_brief: WorkerBrief,
) -> tuple[str, str, str]:
    return propose_candidate(
        model=model,
        program_text=program_text,
        strategy_text=strategy_text,
        memory_text=memory_text,
        candidate_code=candidate_code,
        recent_rows=recent_rows,
        worker_brief=worker_brief,
    )


def _write_round_summary(
    *,
    round_dir: Path,
    all_rows: list[dict[str, Any]],
    incumbent_sha: str,
    incumbent_proxy_result: Any,
    incumbent_strict_result: Any,
    model: str,
    start_time: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "attempts_completed": len([row for row in all_rows if row["phase"] == "proxy"]),
        "kept_attempts": len([row for row in all_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
        "discarded_attempts": len([row for row in all_rows if row["status"] == "discard"]),
        "crashed_attempts": len([row for row in all_rows if row["status"] == "crash"]),
        "infra_failures": len([row for row in all_rows if row["status"] == "infra_fail"]),
        "incumbent_sha256": incumbent_sha,
        "incumbent_mean_accuracy_proxy": incumbent_proxy_result.mean_accuracy,
        "incumbent_mean_time_seconds_proxy": incumbent_proxy_result.mean_time_seconds,
        "incumbent_valid_proxy": incumbent_proxy_result.valid,
        "incumbent_mean_accuracy_strict": incumbent_strict_result.mean_accuracy,
        "incumbent_mean_time_seconds_strict": incumbent_strict_result.mean_time_seconds,
        "incumbent_valid_strict": incumbent_strict_result.valid,
        "elapsed_wall_clock_seconds": time.time() - start_time,
        "run_dir": str(round_dir),
    }
    write_json(round_dir / "summary.json", payload)
    return payload


def _run_parallel_batch_campaign(
    *,
    run_dir: Path,
    candidate_path: Path,
    program_path: Path,
    strategy_path: Path,
    memory_path: Path,
    record_path: Path,
    model: str,
    strategy_model: str | None,
    strategy_rounds: int,
    attempts_per_round: int,
    strict_top_k: int,
    proxy_cfg: AirbenchEvalConfig,
    strict_cfg: AirbenchEvalConfig,
) -> int:
    # The coordinator stays on CPU and fans out GPU-bound evaluations to the
    # fixed AirBench runner. This preserves benchmark integrity while increasing
    # search throughput across independent candidates.
    strategy_history_dir = run_dir / "strategy_history"
    strategy_history_dir.mkdir(parents=True, exist_ok=True)
    (strategy_history_dir / "round_00_start.md").write_text(
        strategy_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    incumbent_code = candidate_path.read_text(encoding="utf-8")
    incumbent_sha = text_sha256(incumbent_code)
    incumbent_proxy_result, incumbent_strict_result = load_incumbent_record(record_path, incumbent_sha)
    incumbent_record_payload = serialize_incumbent_record(incumbent_sha, incumbent_proxy_result, incumbent_strict_result)
    write_json(run_dir / "incumbent_record.json", incumbent_record_payload)
    (run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")

    campaign_rounds: list[dict[str, Any]] = []
    strategist_model_name = strategy_model or model

    for round_idx in range(1, strategy_rounds + 1):
        round_dir = run_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        attempts_dir = round_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        results_writer = build_results_writer(round_dir / "results.tsv")
        start_time = time.time()
        all_rows: list[dict[str, Any]] = []
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []

        try:
            print(f"[meta] round {round_idx}/{strategy_rounds}: planning {attempts_per_round} parallel workers")
            seed_proxy_row = eval_row(
                incumbent_proxy_result,
                attempt=0,
                phase="baseline_proxy_record",
                status="loaded",
                candidate_sha=incumbent_sha,
                parent_sha="",
                change_summary="loaded validated incumbent proxy record",
            )
            results_writer.writerow(seed_proxy_row)
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(seed_proxy_row)
            seed_strict_row = eval_row(
                incumbent_strict_result,
                attempt=0,
                phase="baseline_strict_record",
                status="loaded",
                candidate_sha=incumbent_sha,
                parent_sha="",
                change_summary="loaded validated incumbent strict record",
            )
            results_writer.writerow(seed_strict_row)
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(seed_strict_row)
            accepted_rows.append(seed_strict_row)
            update_memory(memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

            program_text = program_path.read_text(encoding="utf-8")
            strategy_text = strategy_path.read_text(encoding="utf-8")
            memory_text = memory_path.read_text(encoding="utf-8")
            recent_rows = all_rows[-10:]

            worker_briefs, raw_worker_briefs = propose_worker_briefs(
                model=strategist_model_name,
                program_text=program_text,
                strategy_text=strategy_text,
                memory_text=memory_text,
                candidate_code=incumbent_code,
                recent_rows=recent_rows,
                worker_count=attempts_per_round,
            )
            write_json(
                round_dir / "worker_briefs.json",
                [brief.__dict__ for brief in worker_briefs],
            )
            (round_dir / "worker_briefs.raw.txt").write_text(raw_worker_briefs, encoding="utf-8")

            proposal_results: dict[int, tuple[str, str, str] | Exception] = {}
            with ThreadPoolExecutor(max_workers=attempts_per_round) as executor:
                future_to_attempt = {
                    executor.submit(
                        _proposal_worker,
                        model=model,
                        program_text=program_text,
                        strategy_text=strategy_text,
                        memory_text=memory_text,
                        candidate_code=incumbent_code,
                        recent_rows=recent_rows,
                        worker_brief=brief,
                    ): attempt
                    for attempt, brief in enumerate(worker_briefs, start=1)
                }
                for future in as_completed(future_to_attempt):
                    attempt = future_to_attempt[future]
                    try:
                        proposal_results[attempt] = future.result()
                    except Exception as exc:
                        proposal_results[attempt] = exc

            candidates_for_proxy: list[tuple[int, str, str]] = []
            seen_candidate_shas: set[str] = {incumbent_sha}
            proposal_meta: dict[int, tuple[str, str]] = {}
            for attempt, brief in enumerate(worker_briefs, start=1):
                attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                write_json(attempt_dir / "worker_brief.json", brief.__dict__)

                proposal_result = proposal_results.get(attempt)
                if isinstance(proposal_result, Exception):
                    error_row = {
                        "attempt": attempt,
                        "phase": "proposal",
                        "status": "crash",
                        "candidate_sha256": "",
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "meets_target": False,
                        "mean_accuracy": None,
                        "mean_time_seconds": None,
                        "score": 0.0,
                        "failure_type": proposal_result.__class__.__name__,
                        "actual_device_name": None,
                        "runtime_seconds": None,
                        "remote_runtime_seconds": None,
                        "change_summary": str(proposal_result),
                    }
                    results_writer.writerow(error_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(error_row)
                    rejected_rows.append(error_row)
                    continue

                summary, proposed_code, raw_response = proposal_result
                proposed_sha = text_sha256(proposed_code)
                proposal_meta[attempt] = (summary, proposed_sha)
                (attempt_dir / "proposal.raw.txt").write_text(raw_response, encoding="utf-8")
                (attempt_dir / "candidate.py").write_text(proposed_code, encoding="utf-8")
                (attempt_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

                try:
                    compile(proposed_code, "<candidate>", "exec")
                except SyntaxError as exc:
                    syntax_row = {
                        "attempt": attempt,
                        "phase": "proxy",
                        "status": "crash",
                        "candidate_sha256": proposed_sha,
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "meets_target": False,
                        "mean_accuracy": None,
                        "mean_time_seconds": None,
                        "score": 0.0,
                        "failure_type": "syntax_error",
                        "actual_device_name": None,
                        "runtime_seconds": 0.0,
                        "remote_runtime_seconds": None,
                        "change_summary": f"{summary} | syntax error: {exc}",
                    }
                    results_writer.writerow(syntax_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(syntax_row)
                    rejected_rows.append(syntax_row)
                    continue

                contract_error = validate_candidate_contract(proposed_code)
                if contract_error is not None:
                    contract_row = {
                        "attempt": attempt,
                        "phase": "proxy",
                        "status": "crash",
                        "candidate_sha256": proposed_sha,
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "meets_target": False,
                        "mean_accuracy": None,
                        "mean_time_seconds": None,
                        "score": 0.0,
                        "failure_type": "contract_error",
                        "actual_device_name": None,
                        "runtime_seconds": 0.0,
                        "remote_runtime_seconds": None,
                        "change_summary": f"{summary} | contract error: {contract_error}",
                    }
                    results_writer.writerow(contract_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(contract_row)
                    rejected_rows.append(contract_row)
                    continue

                if proposed_sha in seen_candidate_shas:
                    duplicate_row = {
                        "attempt": attempt,
                        "phase": "proxy",
                        "status": "discard",
                        "candidate_sha256": proposed_sha,
                        "parent_sha256": incumbent_sha,
                        "valid": True,
                        "meets_target": None,
                        "mean_accuracy": None,
                        "mean_time_seconds": None,
                        "score": 0.0,
                        "failure_type": "duplicate_candidate",
                        "actual_device_name": None,
                        "runtime_seconds": 0.0,
                        "remote_runtime_seconds": None,
                        "change_summary": f"{summary} | duplicate candidate hash",
                    }
                    results_writer.writerow(duplicate_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(duplicate_row)
                    rejected_rows.append(duplicate_row)
                    continue

                seen_candidate_shas.add(proposed_sha)
                candidates_for_proxy.append((attempt, proposed_sha, proposed_code))

            print(f"[meta] round {round_idx}: running {len(candidates_for_proxy)} proxy evals in parallel")
            proxy_results = _evaluate_candidates_parallel(candidates_for_proxy, proxy_cfg)

            proxy_candidates: list[tuple[int, str, str, Any]] = []
            for attempt, proposed_sha, proposed_code in candidates_for_proxy:
                summary, _ = proposal_meta[attempt]
                result = proxy_results[attempt]
                write_json(attempts_dir / f"attempt_{attempt:03d}" / "proxy_eval.json", result.as_side_info())
                if is_infra_failure(result):
                    row = eval_row(
                        result,
                        attempt=attempt,
                        phase="proxy",
                        status="infra_fail",
                        candidate_sha=proposed_sha,
                        parent_sha=incumbent_sha,
                        change_summary=f"{summary} | infrastructure failure",
                    )
                    results_writer.writerow(row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(row)
                    rejected_rows.append(row)
                    _write_round_summary(
                        round_dir=round_dir,
                        all_rows=all_rows,
                        incumbent_sha=incumbent_sha,
                        incumbent_proxy_result=incumbent_proxy_result,
                        incumbent_strict_result=incumbent_strict_result,
                        model=model,
                        start_time=start_time,
                    )
                    update_memory(memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                    return 1

                if is_better(result, incumbent_proxy_result):
                    status = "candidate"
                    proxy_candidates.append((attempt, proposed_sha, proposed_code, result))
                elif result.valid:
                    status = "discard"
                else:
                    status = "crash"
                row = eval_row(
                    result,
                    attempt=attempt,
                    phase="proxy",
                    status=status,
                    candidate_sha=proposed_sha,
                    parent_sha=incumbent_sha,
                    change_summary=summary,
                )
                results_writer.writerow(row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                all_rows.append(row)
                if status == "candidate":
                    continue
                rejected_rows.append(row)

            proxy_candidates.sort(key=lambda item: item[0])
            ranked_proxy = sorted(
                proxy_candidates,
                key=lambda item: (
                    0 if item[3].as_side_info().get("meets_target") else 1,
                    item[3].mean_time_seconds if item[3].mean_time_seconds is not None else 1e9,
                    -(item[3].mean_accuracy or 0.0),
                ),
            )
            strict_batch = ranked_proxy[: max(0, strict_top_k)]
            if strict_batch:
                print(f"[meta] round {round_idx}: running {len(strict_batch)} strict confirmations in parallel")
                strict_results = _evaluate_candidates_parallel(
                    [(attempt, proposed_sha, proposed_code) for attempt, proposed_sha, proposed_code, _ in strict_batch],
                    strict_cfg,
                )
                best_keep: tuple[int, str, str, Any, Any] | None = None
                strict_candidate_rows: list[tuple[int, str, str, Any]] = []
                for attempt, proposed_sha, proposed_code, proxy_result in strict_batch:
                    summary, _ = proposal_meta[attempt]
                    strict_result = strict_results[attempt]
                    write_json(attempts_dir / f"attempt_{attempt:03d}" / "strict_eval.json", strict_result.as_side_info())
                    if is_infra_failure(strict_result):
                        row = eval_row(
                            strict_result,
                            attempt=attempt,
                            phase="strict_confirm",
                            status="infra_fail",
                            candidate_sha=proposed_sha,
                            parent_sha=incumbent_sha,
                            change_summary=f"{summary} | infrastructure failure during strict confirmation",
                        )
                        results_writer.writerow(row)
                        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                        all_rows.append(row)
                        rejected_rows.append(row)
                        _write_round_summary(
                            round_dir=round_dir,
                            all_rows=all_rows,
                            incumbent_sha=incumbent_sha,
                            incumbent_proxy_result=incumbent_proxy_result,
                            incumbent_strict_result=incumbent_strict_result,
                            model=model,
                            start_time=start_time,
                        )
                        update_memory(memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                        return 1

                    if is_better(strict_result, incumbent_strict_result):
                        if best_keep is None or is_better(strict_result, best_keep[4]):
                            best_keep = (attempt, proposed_sha, proposed_code, proxy_result, strict_result)
                    strict_candidate_rows.append((attempt, proposed_sha, summary, strict_result))

                strict_rows_to_write: list[dict[str, Any]] = []
                for attempt, proposed_sha, summary, strict_result in strict_candidate_rows:
                    status = "discard" if strict_result.valid else "crash"
                    change_summary = f"{summary} | failed strict confirmation"
                    if best_keep is not None and attempt == best_keep[0]:
                        status = "keep"
                        change_summary = summary
                    strict_rows_to_write.append(
                        eval_row(
                            strict_result,
                            attempt=attempt,
                            phase="strict_confirm",
                            status=status,
                            candidate_sha=proposed_sha,
                            parent_sha=incumbent_sha,
                            change_summary=change_summary,
                        )
                    )

                for row in sorted(strict_rows_to_write, key=lambda item: int(item["attempt"])):
                    results_writer.writerow(row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(row)
                    if row["status"] == "keep":
                        accepted_rows.append(row)
                    else:
                        rejected_rows.append(row)

                if best_keep is not None:
                    attempt, proposed_sha, proposed_code, proxy_result, strict_result = best_keep
                    incumbent_code = proposed_code
                    incumbent_sha = proposed_sha
                    incumbent_proxy_result = proxy_result
                    incumbent_strict_result = strict_result
                    candidate_path.write_text(incumbent_code, encoding="utf-8")
                    (run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
                    record_payload = serialize_incumbent_record(incumbent_sha, incumbent_proxy_result, incumbent_strict_result)
                    record_path.write_text(json.dumps(record_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
                    write_json(run_dir / "incumbent_record.json", record_payload)

            update_memory(memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
            round_summary = _write_round_summary(
                round_dir=round_dir,
                all_rows=all_rows,
                incumbent_sha=incumbent_sha,
                incumbent_proxy_result=incumbent_proxy_result,
                incumbent_strict_result=incumbent_strict_result,
                model=model,
                start_time=start_time,
            )
            campaign_rounds.append(
                {
                    "round": round_idx,
                    "exit_code": 0,
                    "run_dir": str(round_dir),
                    "summary": round_summary,
                }
            )
        finally:
            close_results_writer(results_writer)

        if round_idx >= strategy_rounds:
            continue

        print(f"[meta] round {round_idx}: revising strategy")
        current_strategy_text = strategy_path.read_text(encoding="utf-8")
        memory_text = memory_path.read_text(encoding="utf-8")
        round_rows = load_results_rows(round_dir / "results.tsv")
        new_strategy_text, raw_strategy_response = propose_strategy_update(
            model=strategist_model_name,
            program_text=program_path.read_text(encoding="utf-8"),
            current_strategy_text=current_strategy_text,
            memory_text=memory_text,
            round_summary=campaign_rounds[-1]["summary"],
            round_rows=round_rows,
        )
        (strategy_history_dir / f"round_{round_idx:02d}_strategy.raw.txt").write_text(raw_strategy_response, encoding="utf-8")
        strategy_path.write_text(new_strategy_text.rstrip() + "\n", encoding="utf-8")
        (strategy_history_dir / f"round_{round_idx:02d}_strategy.md").write_text(new_strategy_text.rstrip() + "\n", encoding="utf-8")

    final_record = json.loads(record_path.read_text(encoding="utf-8"))
    write_json(
        run_dir / "summary.json",
        {
            "model": model,
            "strategy_model": strategist_model_name,
            "strategy_rounds": strategy_rounds,
            "attempts_per_round": attempts_per_round,
            "parallel_workers": attempts_per_round,
            "strict_top_k": strict_top_k,
            "exit_code": 0,
            "rounds": campaign_rounds,
            "final_incumbent_sha256": final_record.get("candidate_sha256"),
            "final_incumbent_strict": final_record.get("strict"),
            "run_dir": str(run_dir),
        },
    )
    return 0


@app.function(
    image=remote_image,
    gpu=DEFAULT_GPU,
    volumes={REMOTE_DATA_DIR: cifar_volume, REMOTE_RUN_ROOT: autoresearch_volume},
    secrets=_secrets_list(),
    timeout=60 * 60 * 8,
    cpu=8,
    memory=32768,
)
def run_autoresearch_loop_remote(
    *,
    run_name: str,
    initial_candidate_code: str,
    initial_program_text: str,
    initial_strategy_text: str,
    initial_memory_text: str,
    initial_record_text: str,
    model: str,
    strategy_model: str | None,
    strategy_rounds: int,
    max_attempts: int,
    target_accuracy: float,
    proxy_trials: int,
    strict_trials: int,
    warmup_trials: int,
    timeout_seconds: int,
    final_strict_eval: bool,
    stream_logs: bool,
) -> dict[str, Any]:
    ensure_auth(model)
    run_dir = Path(REMOTE_RUN_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = run_dir / "candidate.py"
    program_path = run_dir / "program.md"
    strategy_path = run_dir / "strategy.md"
    memory_path = run_dir / "memory.md"
    record_path = run_dir / "incumbent_record.json"
    candidate_path.write_text(initial_candidate_code, encoding="utf-8")
    program_path.write_text(initial_program_text, encoding="utf-8")
    strategy_path.write_text(initial_strategy_text, encoding="utf-8")
    memory_path.write_text(initial_memory_text, encoding="utf-8")
    record_path.write_text(initial_record_text, encoding="utf-8")

    proxy_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(target_accuracy),
        trials=proxy_trials,
        warmup_trials=warmup_trials,
        timeout_seconds=timeout_seconds,
        preflight=True,
        candidate_verbose=stream_logs,
        stream_subprocess_logs=stream_logs,
    )
    strict_cfg = replace(proxy_cfg, trials=strict_trials)
    inline_runner = _InlineRunner()

    def evaluate_proxy(code: str):
        return evaluate_solver_code(code, proxy_cfg, inline_runner)

    def evaluate_strict(code: str):
        return evaluate_solver_code(code, strict_cfg, inline_runner)

    loop_cfg = AutoresearchLoopConfig(
        candidate_path=candidate_path,
        program_path=program_path,
        strategy_path=strategy_path,
        memory_path=memory_path,
        incumbent_record_path=record_path,
        run_dir=run_dir,
        model=model,
        max_attempts=max_attempts,
        final_strict_eval=final_strict_eval,
        strategy_rounds=strategy_rounds,
        strategy_model=strategy_model,
    )
    exit_code = run_meta_autoresearch_loop(loop_cfg, evaluate_proxy, evaluate_strict, logger=print)
    autoresearch_volume.commit()

    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "exit_code": exit_code,
            "run_name": run_name,
            "remote_run_dir": str(run_dir),
            "requested_gpu": DEFAULT_GPU,
        }
    )
    return summary


@app.function(
    image=remote_image,
    volumes={REMOTE_RUN_ROOT: autoresearch_volume},
    secrets=_secrets_list(),
    timeout=60 * 60 * 8,
    cpu=8,
    memory=32768,
)
def run_parallel_autoresearch_batches_remote(
    *,
    run_name: str,
    initial_candidate_code: str,
    initial_program_text: str,
    initial_strategy_text: str,
    initial_memory_text: str,
    initial_record_text: str,
    model: str,
    strategy_model: str | None,
    strategy_rounds: int,
    attempts_per_round: int,
    parallel_workers: int,
    strict_top_k: int,
    target_accuracy: float,
    proxy_trials: int,
    strict_trials: int,
    warmup_trials: int,
    timeout_seconds: int,
    stream_logs: bool,
) -> dict[str, Any]:
    ensure_auth(model)
    if strategy_model:
        ensure_auth(strategy_model)
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least 1")
    if attempts_per_round != parallel_workers:
        raise ValueError("parallel batch mode currently requires attempts_per_round == parallel_workers")
    if strict_top_k < 0:
        raise ValueError("strict_top_k must be non-negative")
    strict_top_k = min(strict_top_k, attempts_per_round)

    run_dir = Path(REMOTE_RUN_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = run_dir / "candidate.py"
    program_path = run_dir / "program.md"
    strategy_path = run_dir / "strategy.md"
    memory_path = run_dir / "memory.md"
    record_path = run_dir / "incumbent_record.json"
    candidate_path.write_text(initial_candidate_code, encoding="utf-8")
    program_path.write_text(initial_program_text, encoding="utf-8")
    strategy_path.write_text(initial_strategy_text, encoding="utf-8")
    memory_path.write_text(initial_memory_text, encoding="utf-8")
    record_path.write_text(initial_record_text, encoding="utf-8")

    proxy_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(target_accuracy),
        trials=proxy_trials,
        warmup_trials=warmup_trials,
        timeout_seconds=timeout_seconds,
        preflight=True,
        candidate_verbose=stream_logs,
        stream_subprocess_logs=False,
    )
    strict_cfg = replace(proxy_cfg, trials=strict_trials)
    exit_code = _run_parallel_batch_campaign(
        run_dir=run_dir,
        candidate_path=candidate_path,
        program_path=program_path,
        strategy_path=strategy_path,
        memory_path=memory_path,
        record_path=record_path,
        model=model,
        strategy_model=strategy_model,
        strategy_rounds=strategy_rounds,
        attempts_per_round=attempts_per_round,
        strict_top_k=strict_top_k,
        proxy_cfg=proxy_cfg,
        strict_cfg=strict_cfg,
    )
    autoresearch_volume.commit()

    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "exit_code": exit_code,
            "run_name": run_name,
            "remote_run_dir": str(run_dir),
            "requested_gpu": DEFAULT_GPU,
            "parallel_workers": parallel_workers,
            "strict_top_k": strict_top_k,
        }
    )
    return summary


def _volume_run_dir(run_name: str) -> str:
    return f"/{run_name}"


def _read_volume_file(path: str) -> bytes:
    stream = autoresearch_volume.read_file(path)
    if hasattr(stream, "__aiter__"):
        async def _collect_async() -> bytes:
            chunks: list[bytes] = []
            async for chunk in stream:  # type: ignore[union-attr]
                chunks.append(chunk)
            return b"".join(chunks)

        import asyncio

        return asyncio.run(_collect_async())
    return b"".join(stream)


def _sync_run_from_volume(run_name: str, local_root: Path) -> Path:
    remote_root = _volume_run_dir(run_name)
    entries = autoresearch_volume.listdir(remote_root, recursive=True)
    if not entries:
        raise FileNotFoundError(f"No remote run artifacts found for {run_name!r}")

    local_run_dir = local_root / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        relative_path = entry.path.lstrip("/")
        if relative_path.startswith(run_name + "/"):
            relative_path = relative_path[len(run_name) + 1 :]
        if not relative_path:
            continue
        destination = local_run_dir / relative_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _read_volume_file(entry.path)
        destination.write_bytes(payload)
    return local_run_dir


@app.local_entrypoint(name="launch")
def launch(
    max_attempts: int = 20,
    model: str = "gemini/gemini-3.1-flash-lite-preview",
    strategy_model: str = "",
    strategy_rounds: int = 1,
    parallel_workers: int = 1,
    strict_top_k: int = 2,
    target_accuracy: float = 94.0,
    proxy_trials: int = 1,
    strict_trials: int = 5,
    warmup_trials: int = 1,
    timeout_seconds: int = 60 * 15,
    final_strict_eval: bool = False,
    stream_logs: bool = True,
    wait_for_result: bool = False,
) -> None:
    load_dotenv(DEFAULT_DOTENV_PATH)
    ensure_auth(model)
    if strategy_model:
        ensure_auth(strategy_model)
    if parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")
    if strict_top_k < 0:
        raise ValueError("--strict-top-k must be non-negative")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_kwargs = dict(
        run_name=run_name,
        initial_candidate_code=DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"),
        initial_program_text=DEFAULT_PROGRAM_PATH.read_text(encoding="utf-8"),
        initial_strategy_text=DEFAULT_STRATEGY_PATH.read_text(encoding="utf-8"),
        initial_memory_text=DEFAULT_MEMORY_PATH.read_text(encoding="utf-8"),
        initial_record_text=DEFAULT_RECORD_PATH.read_text(encoding="utf-8"),
        model=model,
        strategy_model=strategy_model or None,
        strategy_rounds=strategy_rounds,
        target_accuracy=target_accuracy,
        proxy_trials=proxy_trials,
        strict_trials=strict_trials,
        warmup_trials=warmup_trials,
        timeout_seconds=timeout_seconds,
        stream_logs=stream_logs,
    )

    if parallel_workers > 1:
        if max_attempts != parallel_workers:
            raise ValueError(
                "parallel batch mode currently requires --max-attempts to equal --parallel-workers"
            )
        runner = run_parallel_autoresearch_batches_remote
        kwargs = dict(
            **base_kwargs,
            attempts_per_round=max_attempts,
            parallel_workers=parallel_workers,
            strict_top_k=strict_top_k,
        )
    else:
        runner = run_autoresearch_loop_remote
        kwargs = dict(
            **base_kwargs,
            max_attempts=max_attempts,
            final_strict_eval=final_strict_eval,
        )

    if wait_for_result:
        result = runner.remote(**kwargs)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return

    function_call = runner.spawn(**kwargs)
    print(
        json.dumps(
            {
                "run_name": run_name,
                "function_call_id": function_call.object_id,
                "remote_run_dir": f"{REMOTE_RUN_ROOT}/{run_name}",
                "parallel_workers": parallel_workers,
                "strict_top_k": strict_top_k if parallel_workers > 1 else None,
                "pull_command": (
                    "modal run scripts/airbench_autoresearch/modal_autoresearch.py::pull "
                    f"--run-name {run_name} --apply-incumbent"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="pull")
def pull(
    run_name: str,
    local_root: str = str(DEFAULT_RUN_ROOT),
    apply_incumbent: bool = False,
) -> None:
    local_run_dir = _sync_run_from_volume(run_name, Path(local_root))
    if apply_incumbent:
        incumbent_path = local_run_dir / "incumbent.py"
        memory_path = local_run_dir / "memory.md"
        if incumbent_path.exists():
            DEFAULT_CANDIDATE_PATH.write_text(incumbent_path.read_text(encoding="utf-8"), encoding="utf-8")
        if memory_path.exists():
            DEFAULT_MEMORY_PATH.write_text(memory_path.read_text(encoding="utf-8"), encoding="utf-8")
        strategy_path = local_run_dir / "strategy.md"
        if strategy_path.exists():
            DEFAULT_STRATEGY_PATH.write_text(strategy_path.read_text(encoding="utf-8"), encoding="utf-8")
        record_path = local_run_dir / "incumbent_record.json"
        if record_path.exists():
            DEFAULT_RECORD_PATH.write_text(record_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        json.dumps(
            {
                "local_run_dir": str(local_run_dir),
                "applied_incumbent": apply_incumbent,
            },
            indent=2,
            sort_keys=True,
        )
    )
