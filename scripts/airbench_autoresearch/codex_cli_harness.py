#!/usr/bin/env python3
"""Local multi-agent AirBench harness orchestrated via Codex CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1] / "airbench_gepa"))
    from airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from modal_airbench import app, run_airbench_candidate

    sys.path.append(str(Path(__file__).resolve().parent))
    from loop_core import (
        build_results_writer,
        close_results_writer,
        eval_row,
        is_better,
        is_infra_failure,
        load_dotenv,
        normalize_target_accuracy,
        text_sha256,
        update_memory,
        write_json,
    )
else:
    raise RuntimeError("codex_cli_harness.py is intended to be executed as a script")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "airbench" / "codex_cli_runs"
DEFAULT_CANDIDATE_PATH = Path(__file__).with_name("candidate.py")
DEFAULT_PROGRAM_PATH = Path(__file__).with_name("program.md")
DEFAULT_MEMORY_PATH = Path(__file__).with_name("memory.md")


@dataclass(frozen=True)
class WorkerBrief:
    title: str
    family: str
    hypothesis: str
    instructions: str


@dataclass(frozen=True)
class CodexExecResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    last_message: str
    runtime_seconds: float


def extract_summary_and_code(raw_text: str) -> tuple[str, str]:
    # The Codex CLI harness still uses a full-file proposal protocol:
    # one SUMMARY line plus one fenced Python block containing the entire
    # candidate.py file. Keep this parser local so it does not drift with
    # the section-locked loop_core proposal format.
    summary = "model proposal"
    summary_match = re.search(r"^SUMMARY:\s*(.+)$", raw_text, flags=re.MULTILINE)
    if summary_match:
        summary = summary_match.group(1).strip()

    fence_match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        code = fence_match.group(1).strip()
    else:
        code = raw_text.strip()
        if summary_match:
            code = raw_text[summary_match.end() :].strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else ""
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0].rstrip()
    if not code:
        raise ValueError("model response did not include candidate code")
    return summary, code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--program-path", type=Path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers-per-round", type=int, default=3)
    parser.add_argument("--strict-top-k", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float, default=94.0)
    parser.add_argument("--proxy-trials", type=int, default=1)
    parser.add_argument("--strict-trials", type=int, default=3)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=60 * 15)
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=60 * 15,
        help="Per-Codex coordinator/worker/reviewer timeout in seconds.",
    )
    parser.add_argument(
        "--final-strict-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one final strict evaluation on the final incumbent.",
    )
    parser.add_argument(
        "--apply-incumbent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite candidate.py and memory.md with the final incumbent and memory after the run.",
    )
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream Modal and candidate logs while the app is active.",
    )
    parser.add_argument("--codex-executable", type=str, default="codex")
    parser.add_argument("--codex-model", type=str, default="")
    parser.add_argument(
        "--codex-oss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Codex CLI with a local OSS provider.",
    )
    parser.add_argument("--codex-local-provider", choices=("ollama", "lmstudio"), default="")
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    return parser.parse_args()


def _require_codex(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Could not find Codex CLI executable {executable!r} on PATH")
    return resolved


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _json_schema_for_briefs(worker_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["briefs"],
        "properties": {
            "briefs": {
                "type": "array",
                "minItems": worker_count,
                "maxItems": worker_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "family", "hypothesis", "instructions"],
                    "properties": {
                        "title": {"type": "string"},
                        "family": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "instructions": {"type": "string"},
                    },
                },
            },
        },
    }


def _json_schema_for_review() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "lessons", "promising_attempts", "recommended_attempt"],
        "properties": {
            "summary": {"type": "string"},
            "lessons": {"type": "array", "items": {"type": "string"}},
            "promising_attempts": {"type": "array", "items": {"type": "integer"}},
            "recommended_attempt": {"type": ["integer", "null"]},
        },
    }


def _run_codex_exec(
    *,
    executable: str,
    prompt: str,
    cwd: Path,
    output_path: Path,
    timeout_seconds: int,
    sandbox: str,
    model: str,
    use_oss: bool,
    local_provider: str,
    schema_path: Path | None = None,
) -> CodexExecResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        executable,
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "--output-last-message",
        str(output_path),
    ]
    if schema_path is not None:
        argv.extend(["--output-schema", str(schema_path)])
    if model:
        argv.extend(["-m", model])
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
    last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    return CodexExecResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        last_message=last_message,
        runtime_seconds=time.time() - started_at,
    )


def _parse_worker_briefs(raw_text: str, expected_count: int) -> list[WorkerBrief]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object containing worker briefs")
    items = payload.get("briefs")
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError(f"expected JSON object with briefs array of length {expected_count}")
    briefs: list[WorkerBrief] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("worker brief entries must be objects")
        brief = WorkerBrief(
            title=str(item.get("title", "")).strip(),
            family=str(item.get("family", "")).strip(),
            hypothesis=str(item.get("hypothesis", "")).strip(),
            instructions=str(item.get("instructions", "")).strip(),
        )
        if not all((brief.title, brief.family, brief.hypothesis, brief.instructions)):
            raise ValueError("worker brief fields must all be non-empty")
        briefs.append(brief)
    return briefs


def _review_notes_text(payload: dict[str, Any]) -> str:
    lines = ["# Reviewer Notes", ""]
    lines.append(f"- summary: {payload.get('summary', 'none')}")
    recommended = payload.get("recommended_attempt")
    lines.append(f"- recommended_attempt: {recommended}")
    promising = payload.get("promising_attempts", [])
    lines.append(f"- promising_attempts: {', '.join(str(x) for x in promising) if promising else 'none'}")
    lines.append("")
    lines.append("## Lessons")
    lessons = payload.get("lessons") or []
    if lessons:
        for lesson in lessons:
            lines.append(f"- {lesson}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _coordinator_prompt(
    *,
    program_text: str,
    memory_text: str,
    reviewer_notes_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    workers_per_round: int,
) -> str:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: acc={row.get('mean_accuracy')} "
        f"time={row.get('mean_time_seconds')} failure={row.get('failure_type')} "
        f"summary={row.get('change_summary')}"
        for row in recent_rows[-10:]
    ) or "- no recent attempts"
    return (
        "You are coordinating a small local multi-agent coding batch for AirBench.\n\n"
        "Return a JSON object with one key, 'briefs', whose value is an array of exactly "
        f"{workers_per_round} distinct worker briefs.\n"
        "Each brief must contain: title, family, hypothesis, instructions.\n"
        "Make the briefs meaningfully different. Avoid issuing multiple briefs that are just tiny variants of the same scalar tweak.\n"
        "The workers will edit the full candidate.py file, so instructions should be concrete enough to implement without rewriting the wrapper.\n"
        "Preserve CLI flags, JSON output, and benchmark semantics.\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Reviewer notes:\n{reviewer_notes_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Current candidate.py:\n```python\n{candidate_code}\n```\n\n"
        "Return only JSON."
    )


def _worker_prompt(
    *,
    program_text: str,
    memory_text: str,
    reviewer_notes_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    brief: WorkerBrief,
) -> str:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: acc={row.get('mean_accuracy')} "
        f"time={row.get('mean_time_seconds')} failure={row.get('failure_type')} "
        f"summary={row.get('change_summary')}"
        for row in recent_rows[-8:]
    ) or "- no recent attempts"
    return (
        "You are one worker in a local Codex CLI AirBench optimization harness.\n"
        "Return a revised full candidate.py file implementing one coherent experiment.\n"
        "Do not rewrite the CLI wrapper, JSON output block, or the overall benchmark contract unless absolutely necessary.\n"
        "Keep the change technically coherent and readable.\n"
        "Respond with exactly two parts: a first line 'SUMMARY: ...' and one fenced Python code block containing the full updated file.\n\n"
        f"Assigned brief:\n"
        f"- title: {brief.title}\n"
        f"- family: {brief.family}\n"
        f"- hypothesis: {brief.hypothesis}\n"
        f"- instructions: {brief.instructions}\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Reviewer notes:\n{reviewer_notes_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Current candidate.py:\n```python\n{candidate_code}\n```\n\n"
        "Produce one full updated file."
    )


def _reviewer_prompt(
    *,
    program_text: str,
    memory_text: str,
    incumbent_proxy: Any,
    incumbent_strict: Any,
    round_rows: list[dict[str, Any]],
) -> str:
    attempt_lines = []
    for row in round_rows:
        if int(row["attempt"]) == 0:
            continue
        attempt_lines.append(
            f"- attempt {row['attempt']} phase={row['phase']} status={row['status']} "
            f"acc={row.get('mean_accuracy')} time={row.get('mean_time_seconds')} "
            f"failure={row.get('failure_type')} summary={row.get('change_summary')}"
        )
    attempts_text = "\n".join(attempt_lines) or "- no attempts"
    return (
        "You are reviewing one AirBench optimization round.\n"
        "Return concise JSON with summary, lessons, promising_attempts, and recommended_attempt.\n"
        "Lessons should help the next coordinator choose better experiments.\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Incumbent proxy: acc={incumbent_proxy.mean_accuracy} time={incumbent_proxy.mean_time_seconds}\n"
        f"Incumbent strict: acc={incumbent_strict.mean_accuracy} time={incumbent_strict.mean_time_seconds}\n\n"
        f"Round attempts:\n{attempts_text}\n\n"
        "Return only JSON."
    )


def _ranking_key(result: Any) -> tuple[int, float, float]:
    meets_target = bool(result.as_side_info().get("meets_target"))
    return (
        0 if meets_target else 1,
        result.mean_time_seconds if result.mean_time_seconds is not None else 1e9,
        -(result.mean_accuracy or 0.0),
    )


def _worker_task(
    *,
    codex_executable: str,
    cwd: Path,
    round_dir: Path,
    attempt: int,
    prompt: str,
    codex_timeout_seconds: int,
    codex_sandbox: str,
    codex_model: str,
    codex_oss: bool,
    codex_local_provider: str,
) -> tuple[int, CodexExecResult]:
    output_path = round_dir / f"worker_{attempt:02d}.output.txt"
    result = _run_codex_exec(
        executable=codex_executable,
        prompt=prompt,
        cwd=cwd,
        output_path=output_path,
        timeout_seconds=codex_timeout_seconds,
        sandbox=codex_sandbox,
        model=codex_model,
        use_oss=codex_oss,
        local_provider=codex_local_provider,
    )
    return attempt, result


def main() -> int:
    args = parse_args()
    load_dotenv(DEFAULT_DOTENV_PATH)
    codex_executable = _require_codex(args.codex_executable)

    if not args.candidate_path.exists():
        raise FileNotFoundError(f"Candidate not found: {args.candidate_path}")
    if not args.program_path.exists():
        raise FileNotFoundError(f"Program instructions not found: {args.program_path}")
    if not args.memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {args.memory_path}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    reviewer_notes_path = args.run_dir / "reviewer_notes.md"
    reviewer_notes_path.write_text("# Reviewer Notes\n\n- none yet\n", encoding="utf-8")

    program_text = args.program_path.read_text(encoding="utf-8")
    incumbent_code = args.candidate_path.read_text(encoding="utf-8")
    incumbent_sha = text_sha256(incumbent_code)
    incumbent_memory_path = args.run_dir / "memory.md"
    incumbent_memory_path.write_text(args.memory_path.read_text(encoding="utf-8"), encoding="utf-8")
    (args.run_dir / "program.md").write_text(program_text, encoding="utf-8")
    (args.run_dir / "initial_candidate.py").write_text(incumbent_code, encoding="utf-8")

    proxy_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=args.proxy_trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
        preflight=True,
    )
    strict_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=args.strict_trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
        preflight=True,
    )
    if args.modal_show_output:
        proxy_cfg = replace(proxy_cfg, candidate_verbose=True, stream_subprocess_logs=True)
        strict_cfg = replace(strict_cfg, candidate_verbose=True, stream_subprocess_logs=True)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    recent_rows: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    start_time = time.time()

    def evaluate_proxy(code: str):
        return evaluate_solver_code(code, proxy_cfg, run_airbench_candidate)

    def evaluate_strict(code: str):
        return evaluate_solver_code(code, strict_cfg, run_airbench_candidate)

    def _run_harness() -> int:
        nonlocal incumbent_code, incumbent_sha, recent_rows

        print("[codex] evaluating incumbent seed candidate")
        incumbent_proxy_result = evaluate_proxy(incumbent_code)
        write_json(args.run_dir / "seed_proxy_eval.json", incumbent_proxy_result.as_side_info())
        if is_infra_failure(incumbent_proxy_result):
            write_json(
                args.run_dir / "summary.json",
                {
                    "exit_code": 1,
                    "termination_reason": "gpu_mismatch",
                    "run_dir": str(args.run_dir),
                },
            )
            return 1

        print("[codex] verifying incumbent seed candidate with strict eval")
        incumbent_strict_result = evaluate_strict(incumbent_code)
        write_json(args.run_dir / "seed_strict_eval.json", incumbent_strict_result.as_side_info())
        if is_infra_failure(incumbent_strict_result):
            write_json(
                args.run_dir / "summary.json",
                {
                    "exit_code": 1,
                    "termination_reason": "gpu_mismatch",
                    "run_dir": str(args.run_dir),
                },
            )
            return 1

        (args.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
        update_memory(incumbent_memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

        for round_idx in range(1, args.rounds + 1):
            round_dir = args.run_dir / f"round_{round_idx:02d}"
            attempts_dir = round_dir / "attempts"
            round_dir.mkdir(parents=True, exist_ok=True)
            attempts_dir.mkdir(parents=True, exist_ok=True)
            results_writer = build_results_writer(round_dir / "results.tsv")
            round_rows: list[dict[str, Any]] = []

            try:
                baseline_proxy_row = eval_row(
                    incumbent_proxy_result,
                    attempt=0,
                    phase="baseline_proxy_record",
                    status="loaded",
                    candidate_sha=incumbent_sha,
                    parent_sha="",
                    change_summary="loaded incumbent proxy",
                )
                results_writer.writerow(baseline_proxy_row)
                baseline_strict_row = eval_row(
                    incumbent_strict_result,
                    attempt=0,
                    phase="baseline_strict_record",
                    status="loaded",
                    candidate_sha=incumbent_sha,
                    parent_sha="",
                    change_summary="loaded incumbent strict",
                )
                results_writer.writerow(baseline_strict_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                round_rows.extend([baseline_proxy_row, baseline_strict_row])

                coordinator_prompt = _coordinator_prompt(
                    program_text=program_text,
                    memory_text=incumbent_memory_path.read_text(encoding="utf-8"),
                    reviewer_notes_text=reviewer_notes_path.read_text(encoding="utf-8"),
                    candidate_code=incumbent_code,
                    recent_rows=recent_rows,
                    workers_per_round=args.workers_per_round,
                )
                (round_dir / "coordinator_prompt.txt").write_text(coordinator_prompt, encoding="utf-8")
                coordinator_schema_path = round_dir / "coordinator.schema.json"
                write_json(coordinator_schema_path, _json_schema_for_briefs(args.workers_per_round))
                coordinator_exec = _run_codex_exec(
                    executable=codex_executable,
                    prompt=coordinator_prompt,
                    cwd=REPO_ROOT,
                    output_path=round_dir / "coordinator_output.txt",
                    timeout_seconds=args.codex_timeout_seconds,
                    sandbox=args.codex_sandbox,
                    model=args.codex_model,
                    use_oss=args.codex_oss,
                    local_provider=args.codex_local_provider,
                    schema_path=coordinator_schema_path,
                )
                (round_dir / "coordinator.stdout.log").write_text(coordinator_exec.stdout, encoding="utf-8")
                (round_dir / "coordinator.stderr.log").write_text(coordinator_exec.stderr, encoding="utf-8")
                if coordinator_exec.returncode != 0:
                    raise RuntimeError(f"coordinator failed: {_tail(coordinator_exec.stderr or coordinator_exec.stdout)}")
                worker_briefs = _parse_worker_briefs(coordinator_exec.last_message, args.workers_per_round)
                write_json(round_dir / "worker_briefs.json", [brief.__dict__ for brief in worker_briefs])

                proposal_results: dict[int, tuple[WorkerBrief, CodexExecResult | Exception]] = {}
                with ThreadPoolExecutor(max_workers=args.workers_per_round) as executor:
                    future_to_attempt = {}
                    for attempt, brief in enumerate(worker_briefs, start=1):
                        prompt = _worker_prompt(
                            program_text=program_text,
                            memory_text=incumbent_memory_path.read_text(encoding="utf-8"),
                            reviewer_notes_text=reviewer_notes_path.read_text(encoding="utf-8"),
                            candidate_code=incumbent_code,
                            recent_rows=recent_rows,
                            brief=brief,
                        )
                        (round_dir / f"worker_{attempt:02d}.prompt.txt").write_text(prompt, encoding="utf-8")
                        future = executor.submit(
                            _worker_task,
                            codex_executable=codex_executable,
                            cwd=REPO_ROOT,
                            round_dir=round_dir,
                            attempt=attempt,
                            prompt=prompt,
                            codex_timeout_seconds=args.codex_timeout_seconds,
                            codex_sandbox=args.codex_sandbox,
                            codex_model=args.codex_model,
                            codex_oss=args.codex_oss,
                            codex_local_provider=args.codex_local_provider,
                        )
                        future_to_attempt[future] = (attempt, brief)

                    for future in as_completed(future_to_attempt):
                        attempt, brief = future_to_attempt[future]
                        try:
                            _, worker_exec = future.result()
                            proposal_results[attempt] = (brief, worker_exec)
                        except Exception as exc:
                            proposal_results[attempt] = (brief, exc)

                proxy_candidates: list[tuple[int, str, str, str]] = []
                seen_candidate_shas: set[str] = {incumbent_sha}
                for attempt in range(1, args.workers_per_round + 1):
                    brief, result_or_exc = proposal_results[attempt]
                    attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    write_json(attempt_dir / "worker_brief.json", brief.__dict__)

                    if isinstance(result_or_exc, Exception):
                        row = {
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
                            "failure_type": result_or_exc.__class__.__name__,
                            "actual_device_name": None,
                            "runtime_seconds": None,
                            "remote_runtime_seconds": None,
                            "change_summary": str(result_or_exc),
                        }
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        continue

                    worker_exec = result_or_exc
                    (attempt_dir / "proposal.raw.txt").write_text(worker_exec.last_message, encoding="utf-8")
                    (attempt_dir / "worker.stdout.log").write_text(worker_exec.stdout, encoding="utf-8")
                    (attempt_dir / "worker.stderr.log").write_text(worker_exec.stderr, encoding="utf-8")
                    if worker_exec.returncode != 0:
                        row = {
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
                            "failure_type": "codex_exec_error",
                            "actual_device_name": None,
                            "runtime_seconds": worker_exec.runtime_seconds,
                            "remote_runtime_seconds": None,
                            "change_summary": _tail(worker_exec.stderr or worker_exec.stdout),
                        }
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        continue

                    try:
                        summary, proposed_code = extract_summary_and_code(worker_exec.last_message)
                    except Exception as exc:
                        row = {
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
                            "failure_type": exc.__class__.__name__,
                            "actual_device_name": None,
                            "runtime_seconds": worker_exec.runtime_seconds,
                            "remote_runtime_seconds": None,
                            "change_summary": str(exc),
                        }
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        continue

                    proposed_sha = text_sha256(proposed_code)
                    (attempt_dir / "candidate.py").write_text(proposed_code, encoding="utf-8")
                    (attempt_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
                    try:
                        compile(proposed_code, "<candidate>", "exec")
                    except SyntaxError as exc:
                        row = {
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
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        continue

                    if proposed_sha in seen_candidate_shas:
                        row = {
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
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        continue

                    seen_candidate_shas.add(proposed_sha)
                    proxy_candidates.append((attempt, proposed_sha, proposed_code, summary))

                strict_pool: list[tuple[int, str, str, str, Any]] = []
                for attempt, proposed_sha, proposed_code, summary in proxy_candidates:
                    proposed_result = evaluate_proxy(proposed_code)
                    write_json(attempts_dir / f"attempt_{attempt:03d}" / "proxy_eval.json", proposed_result.as_side_info())
                    if is_infra_failure(proposed_result):
                        row = eval_row(
                            proposed_result,
                            attempt=attempt,
                            phase="proxy",
                            status="infra_fail",
                            candidate_sha=proposed_sha,
                            parent_sha=incumbent_sha,
                            change_summary=f"{summary} | infrastructure failure",
                        )
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                        write_json(
                            args.run_dir / "summary.json",
                            {"exit_code": 1, "termination_reason": "gpu_mismatch", "run_dir": str(args.run_dir)},
                        )
                        return 1

                    if is_better(proposed_result, incumbent_proxy_result):
                        status = "candidate"
                        strict_pool.append((attempt, proposed_sha, proposed_code, summary, proposed_result))
                    elif proposed_result.valid:
                        status = "discard"
                    else:
                        status = "crash"

                    row = eval_row(
                        proposed_result,
                        attempt=attempt,
                        phase="proxy",
                        status=status,
                        candidate_sha=proposed_sha,
                        parent_sha=incumbent_sha,
                        change_summary=summary,
                    )
                    results_writer.writerow(row)
                    round_rows.append(row)
                    if status != "candidate":
                        rejected_rows.append(row)

                strict_pool.sort(key=lambda item: _ranking_key(item[4]))
                best_keep: tuple[int, str, str, str, Any, Any] | None = None
                for attempt, proposed_sha, proposed_code, summary, proposed_proxy_result in strict_pool[: max(0, args.strict_top_k)]:
                    proposed_strict_result = evaluate_strict(proposed_code)
                    write_json(
                        attempts_dir / f"attempt_{attempt:03d}" / "strict_eval.json",
                        proposed_strict_result.as_side_info(),
                    )
                    if is_infra_failure(proposed_strict_result):
                        row = eval_row(
                            proposed_strict_result,
                            attempt=attempt,
                            phase="strict_confirm",
                            status="infra_fail",
                            candidate_sha=proposed_sha,
                            parent_sha=incumbent_sha,
                            change_summary=f"{summary} | infrastructure failure during strict confirmation",
                        )
                        results_writer.writerow(row)
                        round_rows.append(row)
                        rejected_rows.append(row)
                        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                        write_json(
                            args.run_dir / "summary.json",
                            {"exit_code": 1, "termination_reason": "gpu_mismatch", "run_dir": str(args.run_dir)},
                        )
                        return 1

                    if is_better(proposed_strict_result, incumbent_strict_result):
                        if best_keep is None or is_better(proposed_strict_result, best_keep[5]):
                            best_keep = (
                                attempt,
                                proposed_sha,
                                proposed_code,
                                summary,
                                proposed_proxy_result,
                                proposed_strict_result,
                            )

                    strict_status = "discard" if proposed_strict_result.valid else "crash"
                    if best_keep is not None and attempt == best_keep[0]:
                        strict_status = "keep"
                    row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status=strict_status,
                        candidate_sha=proposed_sha,
                        parent_sha=incumbent_sha,
                        change_summary=summary if strict_status == "keep" else f"{summary} | failed strict confirmation",
                    )
                    results_writer.writerow(row)
                    round_rows.append(row)
                    if strict_status == "keep":
                        accepted_rows.append(row)
                    else:
                        rejected_rows.append(row)

                if best_keep is not None:
                    _, incumbent_sha, incumbent_code, _, incumbent_proxy_result, incumbent_strict_result = best_keep
                    (args.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")

                reviewer_prompt = _reviewer_prompt(
                    program_text=program_text,
                    memory_text=incumbent_memory_path.read_text(encoding="utf-8"),
                    incumbent_proxy=incumbent_proxy_result,
                    incumbent_strict=incumbent_strict_result,
                    round_rows=round_rows,
                )
                (round_dir / "reviewer_prompt.txt").write_text(reviewer_prompt, encoding="utf-8")
                reviewer_schema_path = round_dir / "reviewer.schema.json"
                write_json(reviewer_schema_path, _json_schema_for_review())
                reviewer_exec = _run_codex_exec(
                    executable=codex_executable,
                    prompt=reviewer_prompt,
                    cwd=REPO_ROOT,
                    output_path=round_dir / "reviewer_output.txt",
                    timeout_seconds=args.codex_timeout_seconds,
                    sandbox=args.codex_sandbox,
                    model=args.codex_model,
                    use_oss=args.codex_oss,
                    local_provider=args.codex_local_provider,
                    schema_path=reviewer_schema_path,
                )
                (round_dir / "reviewer.stdout.log").write_text(reviewer_exec.stdout, encoding="utf-8")
                (round_dir / "reviewer.stderr.log").write_text(reviewer_exec.stderr, encoding="utf-8")
                reviewer_payload: dict[str, Any] | None = None
                if reviewer_exec.returncode == 0 and reviewer_exec.last_message.strip():
                    try:
                        reviewer_payload = json.loads(reviewer_exec.last_message)
                        write_json(round_dir / "reviewer.json", reviewer_payload)
                        reviewer_notes_path.write_text(_review_notes_text(reviewer_payload), encoding="utf-8")
                    except json.JSONDecodeError:
                        pass

                update_memory(incumbent_memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                round_summary = {
                    "round": round_idx,
                    "attempts_completed": len([row for row in round_rows if row["phase"] == "proxy"]),
                    "kept_attempts": len([row for row in round_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
                    "discarded_attempts": len([row for row in round_rows if row["status"] == "discard"]),
                    "crashed_attempts": len([row for row in round_rows if row["status"] == "crash"]),
                    "incumbent_sha256": incumbent_sha,
                    "incumbent_mean_accuracy_proxy": incumbent_proxy_result.mean_accuracy,
                    "incumbent_mean_time_seconds_proxy": incumbent_proxy_result.mean_time_seconds,
                    "incumbent_mean_accuracy_strict": incumbent_strict_result.mean_accuracy,
                    "incumbent_mean_time_seconds_strict": incumbent_strict_result.mean_time_seconds,
                    "reviewer_summary": reviewer_payload.get("summary") if reviewer_payload else None,
                    "elapsed_wall_clock_seconds": time.time() - start_time,
                    "run_dir": str(round_dir),
                }
                write_json(round_dir / "summary.json", round_summary)
                round_summaries.append(round_summary)
                recent_rows = round_rows[-10:]
            finally:
                close_results_writer(results_writer)

        final_payload = {
            "exit_code": 0,
            "rounds": round_summaries,
            "final_incumbent_sha256": incumbent_sha,
            "final_incumbent_mean_accuracy_proxy": incumbent_proxy_result.mean_accuracy,
            "final_incumbent_mean_time_seconds_proxy": incumbent_proxy_result.mean_time_seconds,
            "final_incumbent_mean_accuracy_strict": incumbent_strict_result.mean_accuracy,
            "final_incumbent_mean_time_seconds_strict": incumbent_strict_result.mean_time_seconds,
            "elapsed_wall_clock_seconds": time.time() - start_time,
            "run_dir": str(args.run_dir),
        }

        if args.final_strict_eval:
            final_strict_result = evaluate_strict(incumbent_code)
            write_json(args.run_dir / "final_strict_eval.json", final_strict_result.as_side_info())
            final_payload.update(
                {
                    "final_strict_valid": final_strict_result.valid,
                    "final_strict_mean_accuracy": final_strict_result.mean_accuracy,
                    "final_strict_mean_time_seconds": final_strict_result.mean_time_seconds,
                }
            )

        write_json(args.run_dir / "summary.json", final_payload)
        (args.run_dir / "candidate.py").write_text(incumbent_code, encoding="utf-8")
        return 0

    if args.modal_show_output:
        with modal.enable_output(), app.run():
            exit_code = _run_harness()
    else:
        with app.run():
            exit_code = _run_harness()

    if exit_code == 0 and args.apply_incumbent:
        args.candidate_path.write_text((args.run_dir / "candidate.py").read_text(encoding="utf-8"), encoding="utf-8")
        args.memory_path.write_text(incumbent_memory_path.read_text(encoding="utf-8"), encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
