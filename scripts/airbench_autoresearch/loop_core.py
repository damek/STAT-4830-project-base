#!/usr/bin/env python3
"""Shared autoresearch loop logic for local and Modal-hosted runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def ensure_auth(model: str) -> None:
    if model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is required for model {model!r}")
    if model.startswith("gemini/") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(f"GEMINI_API_KEY or GOOGLE_API_KEY is required for model {model!r}")
    if model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(f"OPENROUTER_API_KEY is required for model {model!r}")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_summary_and_code(raw_text: str) -> tuple[str, str]:
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


def build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "attempt",
        "phase",
        "status",
        "candidate_sha256",
        "parent_sha256",
        "valid",
        "meets_target",
        "mean_accuracy",
        "mean_time_seconds",
        "score",
        "failure_type",
        "actual_device_name",
        "runtime_seconds",
        "remote_runtime_seconds",
        "change_summary",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer._handle = handle  # type: ignore[attr-defined]
    return writer


def close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def eval_row(result: Any, *, attempt: int, phase: str, status: str, candidate_sha: str, parent_sha: str, change_summary: str) -> dict[str, Any]:
    side = result.as_side_info()
    return {
        "attempt": attempt,
        "phase": phase,
        "status": status,
        "candidate_sha256": candidate_sha,
        "parent_sha256": parent_sha,
        "valid": result.valid,
        "meets_target": side.get("meets_target"),
        "mean_accuracy": result.mean_accuracy,
        "mean_time_seconds": result.mean_time_seconds,
        "score": result.score,
        "failure_type": result.failure_type,
        "actual_device_name": side.get("actual_device_name"),
        "runtime_seconds": result.runtime_seconds,
        "remote_runtime_seconds": side.get("remote_runtime_seconds"),
        "change_summary": change_summary,
    }


def is_better(new: Any, old: Any) -> bool:
    if not new.valid:
        return False
    if not old.valid:
        return True

    new_meets = bool(new.as_side_info().get("meets_target"))
    old_meets = bool(old.as_side_info().get("meets_target"))
    if new_meets != old_meets:
        return new_meets

    if new_meets and old_meets:
        if new.mean_time_seconds is None or old.mean_time_seconds is None:
            return False
        if abs(new.mean_time_seconds - old.mean_time_seconds) > 1e-9:
            return new.mean_time_seconds < old.mean_time_seconds
        if new.mean_accuracy is not None and old.mean_accuracy is not None:
            return new.mean_accuracy > old.mean_accuracy + 1e-9
        return False

    if new.mean_accuracy is not None and old.mean_accuracy is not None:
        if abs(new.mean_accuracy - old.mean_accuracy) > 1e-9:
            return new.mean_accuracy > old.mean_accuracy
    if new.mean_time_seconds is not None and old.mean_time_seconds is not None:
        if abs(new.mean_time_seconds - old.mean_time_seconds) > 1e-9:
            return new.mean_time_seconds < old.mean_time_seconds
    return False


def update_memory(memory_path: Path, incumbent_result: Any, accepted_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> None:
    side = incumbent_result.as_side_info()
    lines = ["# Memory", "", "## Current Best"]
    lines.append(f"- valid: {incumbent_result.valid}")
    lines.append(f"- meets_target: {side.get('meets_target')}")
    lines.append(f"- mean_accuracy: {incumbent_result.mean_accuracy}")
    lines.append(f"- mean_time_seconds: {incumbent_result.mean_time_seconds}")
    lines.append(f"- score: {incumbent_result.score}")
    lines.append("")
    lines.append("## Recent Accepted Changes")
    accepted_changes = [row for row in accepted_rows if row.get("attempt", 0) > 0]
    if accepted_changes:
        for row in accepted_changes[-3:]:
            lines.append(f"- attempt {row['attempt']}: {row['change_summary']}")
    else:
        lines.append("- none yet")
    lines.append("")
    lines.append("## Recent Rejections / Failures")
    if rejected_rows:
        for row in rejected_rows[-5:]:
            failure = row.get("failure_type") or row.get("status") or "discard"
            lines.append(f"- attempt {row['attempt']}: {failure} | {row['change_summary']}")
    else:
        lines.append("- none yet")
    lines.append("")
    lines.append("## Repeated Failure Modes")
    lines.append("- Preserve dtype consistency between normalized inputs and half-precision conv weights/biases.")
    lines.append("- Preserve CLI flags, especially --verbose.")
    lines.append("- Avoid CUDAGraph-unsafe repeated compiled inference patterns in TTA.")
    memory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prompt(*, program_text: str, memory_text: str, candidate_code: str, recent_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: acc={row.get('mean_accuracy')} time={row.get('mean_time_seconds')} failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in recent_rows
    ) or "- no recent attempts"
    system = (
        "You are improving one Python training script in a keep/discard experiment loop. "
        "Return one small, defensible edit to the current file. Prefer simplicity and local edits. "
        "Do not return explanations after the code. Respond with exactly two parts: "
        "a first line 'SUMMARY: ...' followed by one fenced Python code block containing the full updated file."
    )
    user = (
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Current candidate.py:\n```python\n{candidate_code}\n```\n\n"
        "Produce a revised full file. Prefer a small, local edit unless a broader rewrite is truly necessary."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def propose_candidate(*, model: str, program_text: str, memory_text: str, candidate_code: str, recent_rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    import litellm

    messages = build_prompt(
        program_text=program_text,
        memory_text=memory_text,
        candidate_code=candidate_code,
        recent_rows=recent_rows,
    )
    response = litellm.completion(model=model, messages=messages)
    raw_text = response.choices[0].message.content or ""
    summary, code = extract_summary_and_code(raw_text)
    return summary, code, raw_text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def is_infra_failure(result: Any) -> bool:
    return result.failure_type == "gpu_mismatch"


@dataclass(frozen=True)
class AutoresearchLoopConfig:
    candidate_path: Path
    program_path: Path
    memory_path: Path
    run_dir: Path
    model: str
    max_attempts: int
    final_strict_eval: bool = True


def run_autoresearch_loop(
    config: AutoresearchLoopConfig,
    evaluate_proxy: Callable[[str], Any],
    evaluate_strict: Callable[[str], Any],
    *,
    logger: Callable[[str], None] = print,
) -> int:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = config.run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    results_writer = build_results_writer(config.run_dir / "results.tsv")
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    current_code = config.candidate_path.read_text(encoding="utf-8")
    incumbent_code = current_code
    incumbent_sha = text_sha256(incumbent_code)
    start_time = time.time()

    def write_partial_summary(*, incumbent_proxy: Any, incumbent_strict: Any | None, infra_failures: int) -> None:
        payload = {
            "model": config.model,
            "attempts_completed": len([row for row in all_rows if row["phase"] == "proxy"]),
            "kept_attempts": len([row for row in all_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
            "discarded_attempts": len([row for row in all_rows if row["status"] == "discard"]),
            "crashed_attempts": len([row for row in all_rows if row["status"] == "crash"]),
            "infra_failures": infra_failures,
            "incumbent_sha256": incumbent_sha,
            "incumbent_mean_accuracy_proxy": incumbent_proxy.mean_accuracy,
            "incumbent_mean_time_seconds_proxy": incumbent_proxy.mean_time_seconds,
            "incumbent_valid_proxy": incumbent_proxy.valid,
            "incumbent_mean_accuracy_strict": incumbent_strict.mean_accuracy if incumbent_strict else None,
            "incumbent_mean_time_seconds_strict": incumbent_strict.mean_time_seconds if incumbent_strict else None,
            "incumbent_valid_strict": incumbent_strict.valid if incumbent_strict else None,
            "elapsed_wall_clock_seconds": time.time() - start_time,
            "run_dir": str(config.run_dir),
            "terminated_early": True,
            "termination_reason": "gpu_mismatch",
        }
        write_json(config.run_dir / "summary.json", payload)

    try:
        logger("[loop] evaluating incumbent seed candidate")
        incumbent_proxy_result = evaluate_proxy(incumbent_code)
        seed_proxy_row = eval_row(
            incumbent_proxy_result,
            attempt=0,
            phase="baseline_proxy",
            status="keep",
            candidate_sha=incumbent_sha,
            parent_sha="",
            change_summary="initial seed candidate",
        )
        results_writer.writerow(seed_proxy_row)
        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
        all_rows.append(seed_proxy_row)
        write_json(config.run_dir / "seed_proxy_eval.json", incumbent_proxy_result.as_side_info())
        if is_infra_failure(incumbent_proxy_result):
            rejected_rows.append(seed_proxy_row)
            logger("[loop] baseline proxy failed due to gpu_mismatch; aborting run")
            update_memory(config.memory_path, incumbent_proxy_result, accepted_rows, rejected_rows)
            write_partial_summary(incumbent_proxy=incumbent_proxy_result, incumbent_strict=None, infra_failures=1)
            return 1

        logger("[loop] verifying incumbent seed candidate with strict eval")
        incumbent_strict_result = evaluate_strict(incumbent_code)
        seed_strict_row = eval_row(
            incumbent_strict_result,
            attempt=0,
            phase="baseline_strict",
            status="verified",
            candidate_sha=incumbent_sha,
            parent_sha="",
            change_summary="initial seed candidate strict verification",
        )
        results_writer.writerow(seed_strict_row)
        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
        all_rows.append(seed_strict_row)
        accepted_rows.append(seed_strict_row)
        write_json(config.run_dir / "seed_strict_eval.json", incumbent_strict_result.as_side_info())
        if is_infra_failure(incumbent_strict_result):
            logger("[loop] baseline strict failed due to gpu_mismatch; aborting run")
            update_memory(config.memory_path, incumbent_proxy_result, accepted_rows, rejected_rows)
            write_partial_summary(incumbent_proxy=incumbent_proxy_result, incumbent_strict=incumbent_strict_result, infra_failures=1)
            return 1
        (config.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
        update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

        infra_failures = 0

        for attempt in range(1, config.max_attempts + 1):
            logger(f"[loop] attempt {attempt}/{config.max_attempts}: proposing edit")
            program_text = config.program_path.read_text(encoding="utf-8")
            memory_text = config.memory_path.read_text(encoding="utf-8")
            recent_rows = all_rows[-5:]
            try:
                summary, proposed_code, raw_response = propose_candidate(
                    model=config.model,
                    program_text=program_text,
                    memory_text=memory_text,
                    candidate_code=incumbent_code,
                    recent_rows=recent_rows,
                )
            except Exception as exc:
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
                    "failure_type": exc.__class__.__name__,
                    "actual_device_name": None,
                    "runtime_seconds": None,
                    "remote_runtime_seconds": None,
                    "change_summary": str(exc),
                }
                results_writer.writerow(error_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(error_row)
                all_rows.append(error_row)
                logger(f"[loop] attempt {attempt}: proposal failed: {exc}")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            proposed_sha = text_sha256(proposed_code)
            if proposed_sha == incumbent_sha:
                no_change_row = {
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
                    "failure_type": "no_change",
                    "actual_device_name": None,
                    "runtime_seconds": 0.0,
                    "remote_runtime_seconds": None,
                    "change_summary": f"{summary} | no code change produced",
                }
                results_writer.writerow(no_change_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(no_change_row)
                all_rows.append(no_change_row)
                logger(f"[loop] attempt {attempt}: discard before eval (no code change)")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
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
                rejected_rows.append(syntax_row)
                all_rows.append(syntax_row)
                logger(f"[loop] attempt {attempt}: crash before eval (syntax error)")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            config.candidate_path.write_text(proposed_code, encoding="utf-8")
            logger(f"[loop] attempt {attempt}: running proxy eval")
            proposed_result = evaluate_proxy(proposed_code)
            write_json(attempt_dir / "proxy_eval.json", proposed_result.as_side_info())
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
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                all_rows.append(row)
                rejected_rows.append(row)
                config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                infra_failures += 1
                logger(f"[loop] attempt {attempt}: gpu_mismatch after internal retries; aborting run")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                write_partial_summary(
                    incumbent_proxy=incumbent_proxy_result,
                    incumbent_strict=incumbent_strict_result,
                    infra_failures=infra_failures,
                )
                return 1

            if is_better(proposed_result, incumbent_proxy_result):
                status = "candidate"
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
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(row)

            if status == "candidate":
                logger(f"[loop] attempt {attempt}: proxy candidate found, running strict confirmation")
                proposed_strict_result = evaluate_strict(proposed_code)
                write_json(attempt_dir / "strict_eval.json", proposed_strict_result.as_side_info())
                if is_infra_failure(proposed_strict_result):
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status="infra_fail",
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=f"{summary} | infrastructure failure during strict confirmation",
                    )
                    results_writer.writerow(strict_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(strict_row)
                    rejected_rows.append(strict_row)
                    config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                    infra_failures += 1
                    logger(f"[loop] attempt {attempt}: gpu_mismatch during strict confirmation; aborting run")
                    update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                    write_partial_summary(
                        incumbent_proxy=incumbent_proxy_result,
                        incumbent_strict=incumbent_strict_result,
                        infra_failures=infra_failures,
                    )
                    return 1
                if is_better(proposed_strict_result, incumbent_strict_result):
                    strict_status = "keep"
                    incumbent_code = proposed_code
                    incumbent_sha = proposed_sha
                    incumbent_proxy_result = proposed_result
                    incumbent_strict_result = proposed_strict_result
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status=strict_status,
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=summary,
                    )
                    accepted_rows.append(strict_row)
                    (config.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
                    logger(
                        f"[loop] attempt {attempt}: keep after strict confirm "
                        f"acc={proposed_strict_result.mean_accuracy} time={proposed_strict_result.mean_time_seconds}"
                    )
                else:
                    strict_status = "discard" if proposed_strict_result.valid else "crash"
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status=strict_status,
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=f"{summary} | failed strict confirmation",
                    )
                    rejected_rows.append(strict_row)
                    config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                    logger(
                        f"[loop] attempt {attempt}: {strict_status} after strict confirm "
                        f"acc={proposed_strict_result.mean_accuracy} time={proposed_strict_result.mean_time_seconds} "
                        f"failure={proposed_strict_result.failure_type}"
                    )
                results_writer.writerow(strict_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                all_rows.append(strict_row)
            else:
                rejected_rows.append(row)
                config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                logger(
                    f"[loop] attempt {attempt}: {status} acc={proposed_result.mean_accuracy} "
                    f"time={proposed_result.mean_time_seconds} failure={proposed_result.failure_type}"
                )

            update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

        final_payload = {
            "model": config.model,
            "attempts_completed": len([row for row in all_rows if row["phase"] == "proxy"]),
            "kept_attempts": len([row for row in all_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
            "discarded_attempts": len([row for row in all_rows if row["status"] == "discard"]),
            "crashed_attempts": len([row for row in all_rows if row["status"] == "crash"]),
            "infra_failures": infra_failures,
            "incumbent_sha256": incumbent_sha,
            "incumbent_mean_accuracy_proxy": incumbent_proxy_result.mean_accuracy,
            "incumbent_mean_time_seconds_proxy": incumbent_proxy_result.mean_time_seconds,
            "incumbent_valid_proxy": incumbent_proxy_result.valid,
            "incumbent_mean_accuracy_strict": incumbent_strict_result.mean_accuracy,
            "incumbent_mean_time_seconds_strict": incumbent_strict_result.mean_time_seconds,
            "incumbent_valid_strict": incumbent_strict_result.valid,
            "elapsed_wall_clock_seconds": time.time() - start_time,
            "run_dir": str(config.run_dir),
        }

        if config.final_strict_eval:
            logger("[loop] running final strict evaluation on incumbent")
            strict_result = evaluate_strict(incumbent_code)
            write_json(config.run_dir / "final_strict_eval.json", strict_result.as_side_info())
            strict_row = eval_row(
                strict_result,
                attempt=config.max_attempts + 1,
                phase="final_strict",
                status="final",
                candidate_sha=incumbent_sha,
                parent_sha=incumbent_sha,
                change_summary="final incumbent verification",
            )
            results_writer.writerow(strict_row)
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(strict_row)
            final_payload.update(
                {
                    "final_strict_valid": strict_result.valid,
                    "final_strict_mean_accuracy": strict_result.mean_accuracy,
                    "final_strict_mean_time_seconds": strict_result.mean_time_seconds,
                    "final_strict_score": strict_result.score,
                }
            )

        write_json(config.run_dir / "summary.json", final_payload)
        return 0
    finally:
        close_results_writer(results_writer)
