#!/usr/bin/env python3
"""Shared helpers for Meta-Harness vector-db-bench evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_REVISIONS_ROOT = Path(__file__).with_name("meta_harness") / "revisions"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "data"
    / "vector_db_bench"
    / "qwen3_meta"
    / "meta_harness_runs"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)

RESULT_COLUMNS = [
    "attempt",
    "revision_id",
    "status",
    "process_returncode",
    "valid",
    "qps",
    "recall",
    "recall_passed",
    "result_source",
    "tool_calls_used",
    "tool_calls_total",
    "best_qps",
    "best_recall",
    "last_qps",
    "last_recall",
    "elapsed_secs",
    "work_dir",
    "notes",
]

WORKER_READABLE_ROOTS = ("src", "benchmarks", "profiling")
WORKER_READABLE_FILES = ("Cargo.toml",)


@dataclass(frozen=True)
class RevisionConfig:
    revision_id: str
    description: str
    attempts_per_eval: int
    official_tools_only: bool
    zero_completion_retry_limit: int
    extra_user_messages: tuple[str, ...]
    added_helper_tools: tuple[str, ...]
    helper_tools_module: Path | None
    seed_files_dir: Path | None
    seed_files_mount_dir: str | None
    notes: str
    revision_dir: Path


@dataclass(frozen=True)
class AttemptProcessResult:
    returncode: int
    elapsed_secs: float


@dataclass(frozen=True)
class AttemptOutcome:
    attempt_index: int
    revision_id: str
    status: str
    process_returncode: int
    valid: bool
    qps: float
    recall: float
    recall_passed: bool
    result_source: str
    tool_calls_used: int
    tool_calls_total: int
    best_qps: float
    best_recall: float
    last_qps: float
    last_recall: float
    elapsed_secs: float
    work_dir: str
    notes: str


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
    writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def flush_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.flush()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ensure_blank_seed(blank_seed_dir: Path, source: Path) -> None:
    if blank_seed_dir.exists():
        return
    if not source.is_dir():
        raise FileNotFoundError(f"blank seed source not found: {source}")
    copy_tree(source, blank_seed_dir)
    db_stub = (blank_seed_dir / "src" / "db.rs").read_text(encoding="utf-8")
    dist_stub = (blank_seed_dir / "src" / "distance.rs").read_text(encoding="utf-8")
    if "todo!(" not in db_stub or "todo!(" not in dist_stub:
        raise RuntimeError(
            "blank seed source does not look like the official empty scaffold; "
            "pass --blank-seed-source explicitly."
        )


def load_revision_config(revisions_root: Path, revision_id: str) -> RevisionConfig:
    revision_dir = (revisions_root / revision_id).resolve()
    revision_path = revision_dir / "revision.toml"
    if not revision_path.exists():
        raise FileNotFoundError(f"revision config not found: {revision_path}")
    payload = tomllib.loads(revision_path.read_text(encoding="utf-8"))
    seed_files_raw = str(payload.get("seed_files_dir", "") or "").strip()
    seed_files_dir = (revision_dir / seed_files_raw).resolve() if seed_files_raw else None
    seed_files_mount_raw = str(payload.get("seed_files_mount_dir", "") or "").strip()
    seed_files_mount_dir = seed_files_mount_raw or ("src" if seed_files_dir is not None else None)
    helper_tools_raw = str(payload.get("helper_tools_module", "") or "").strip()
    helper_tools_module = (revision_dir / helper_tools_raw).resolve() if helper_tools_raw else None
    default_helper_tools_module = revision_dir / "helper_tools.py"
    if helper_tools_module is None and default_helper_tools_module.exists():
        helper_tools_module = default_helper_tools_module.resolve()
    if seed_files_dir is not None and not seed_files_dir.exists():
        raise FileNotFoundError(f"seed_files_dir not found for revision {revision_id}: {seed_files_dir}")
    if helper_tools_module is not None and not helper_tools_module.exists():
        raise FileNotFoundError(
            f"helper_tools_module not found for revision {revision_id}: {helper_tools_module}"
        )
    added_helper_tools = tuple(str(item) for item in payload.get("added_helper_tools", []))
    if added_helper_tools and helper_tools_module is None:
        raise ValueError(
            f"revision {revision_id} declares added_helper_tools but no helper_tools_module "
            "was provided. Add helper_tools_module = 'helper_tools.py' or remove added_helper_tools."
        )
    return RevisionConfig(
        revision_id=str(payload.get("id", revision_id)),
        description=str(payload.get("description", "")),
        attempts_per_eval=int(payload.get("attempts_per_eval", 3)),
        official_tools_only=bool(payload.get("official_tools_only", True)),
        zero_completion_retry_limit=int(payload.get("zero_completion_retry_limit", 0)),
        extra_user_messages=tuple(str(item) for item in payload.get("extra_user_messages", [])),
        added_helper_tools=added_helper_tools,
        helper_tools_module=helper_tools_module,
        seed_files_dir=seed_files_dir,
        seed_files_mount_dir=seed_files_mount_dir,
        notes=str(payload.get("notes", "")),
        revision_dir=revision_dir,
    )


def revision_uses_custom_runtime(revision: RevisionConfig) -> bool:
    return (
        revision.helper_tools_module is not None
        or bool(revision.added_helper_tools)
        or revision.zero_completion_retry_limit > 0
    )


def _chat_message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


def _is_worker_readable_relpath(path: PurePosixPath) -> bool:
    normalized = path.as_posix().lstrip("./")
    if normalized in WORKER_READABLE_FILES:
        return True
    return any(
        normalized == root or normalized.startswith(f"{root}/")
        for root in WORKER_READABLE_ROOTS
    )


def _iter_relative_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _looks_like_worker_path_ref(token: str) -> bool:
    stripped = token.strip()
    if not stripped or " " in stripped or "\n" in stripped:
        return False
    if stripped.startswith(("{", "[", "(")) or "*" in stripped:
        return False
    return "/" in stripped or stripped.endswith((".md", ".rs", ".toml", ".json", ".txt", ".svg"))


def _extract_worker_path_refs(messages: tuple[str, ...]) -> set[str]:
    refs: set[str] = set()
    for message in messages:
        for token in re.findall(r"`([^`\n]+)`", message):
            if _looks_like_worker_path_ref(token):
                refs.add(token.strip())
    return refs


def validate_revision_worker_contract(*, blank_seed_dir: Path, revision: RevisionConfig) -> None:
    available_paths = _iter_relative_files(blank_seed_dir)

    if revision.seed_files_dir is not None:
        if not revision.seed_files_mount_dir:
            raise ValueError(
                f"revision {revision.revision_id} declares seed_files_dir but no seed_files_mount_dir"
            )
        mount_root = PurePosixPath(revision.seed_files_mount_dir)
        if not _is_worker_readable_relpath(mount_root):
            raise ValueError(
                f"revision {revision.revision_id} mounts seed files under unreadable path "
                f"{revision.seed_files_mount_dir!r}; use a worker-readable scope such as 'src'"
            )
        for path in sorted(revision.seed_files_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = PurePosixPath(path.relative_to(revision.seed_files_dir).as_posix())
            dest = mount_root / rel
            if not _is_worker_readable_relpath(dest):
                raise ValueError(
                    f"revision {revision.revision_id} seeds {rel.as_posix()!r} to unreadable path "
                    f"{dest.as_posix()!r}"
                )
            available_paths.add(dest.as_posix())

    for ref in sorted(_extract_worker_path_refs(revision.extra_user_messages)):
        ref_path = PurePosixPath(ref)
        if not _is_worker_readable_relpath(ref_path):
            raise ValueError(
                f"revision {revision.revision_id} references unreadable worker path {ref!r} "
                f"in extra_user_messages"
            )
        if ref not in available_paths:
            raise ValueError(
                f"revision {revision.revision_id} references {ref!r} in extra_user_messages, "
                "but that file will not exist in the fresh attempt workspace"
            )


def _copy_seed_files(seed_files_dir: Path, work_dir: Path, mount_dir: str) -> None:
    for path in sorted(seed_files_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(seed_files_dir)
        dest = work_dir / mount_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def prepare_workdir(
    *,
    bench_repo: Path,
    blank_seed_dir: Path,
    work_dir: Path,
    revision: RevisionConfig,
    max_tool_calls: int,
) -> None:
    """Prepare a workdir for a fresh attempt.

    For `h0`, this leaves workdir absent so the official `run_eval.sh` can
    initialize it exactly as upstream. For later revisions, if extra files or
    extra initial messages are requested, we preseed the workdir and
    `session_context.json` so the upstream agent resumes from that initial state.
    """

    if (
        revision.seed_files_dir is None
        and not revision.extra_user_messages
        and not revision_uses_custom_runtime(revision)
    ):
        if work_dir.exists():
            shutil.rmtree(work_dir)
        return

    copy_tree(blank_seed_dir, work_dir)
    if revision.seed_files_dir is not None:
        if not revision.seed_files_mount_dir:
            raise ValueError(
                f"revision {revision.revision_id} declares seed_files_dir but no seed_files_mount_dir"
            )
        _copy_seed_files(revision.seed_files_dir, work_dir, revision.seed_files_mount_dir)

    system_prompt = (bench_repo / "agent" / "system_prompt.txt").read_text(encoding="utf-8")
    messages = [
        _chat_message("system", system_prompt),
        _chat_message("user", "Begin. Read the project files and start implementing."),
    ]
    messages.extend(_chat_message("user", content) for content in revision.extra_user_messages)
    session_context = {
        "tool_calls_used": 0,
        "tool_calls_total": max_tool_calls,
        "messages": messages,
        "last_benchmark": None,
        "best_benchmark": None,
        "call_log": [],
    }
    (work_dir / "session_context.json").write_text(json.dumps(session_context), encoding="utf-8")


def run_revision_attempt(
    *,
    revision: RevisionConfig,
    bench_repo: Path,
    work_dir: Path,
    results_dir: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    thinking_mode: str,
    reasoning_effort: str,
    api_interval_ms: int,
    cpu_cores: str,
    data_dir: Path,
    max_tool_calls: int,
) -> AttemptProcessResult:
    if revision_uses_custom_runtime(revision):
        try:
            from meta_harness_runtime import run_custom_attempt
        except ModuleNotFoundError:  # pragma: no cover - package import path
            from .meta_harness_runtime import run_custom_attempt

        return run_custom_attempt(
            revision=revision,
            bench_repo=bench_repo,
            work_dir=work_dir,
            results_dir=results_dir,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            api_interval_ms=api_interval_ms,
            cpu_cores=cpu_cores,
            data_dir=data_dir,
            max_tool_calls=max_tool_calls,
        )
    return run_official_attempt(
        bench_repo=bench_repo,
        work_dir=work_dir,
        results_dir=results_dir,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        api_interval_ms=api_interval_ms,
        cpu_cores=cpu_cores,
        data_dir=data_dir,
        max_tool_calls=max_tool_calls,
    )


def run_official_attempt(
    *,
    bench_repo: Path,
    work_dir: Path,
    results_dir: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    thinking_mode: str,
    reasoning_effort: str,
    api_interval_ms: int,
    cpu_cores: str,
    data_dir: Path,
    max_tool_calls: int,
) -> AttemptProcessResult:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": model_name,
            "API_URL": base_url,
            "API_KEY": api_key,
            "MODEL_ID": model_id,
            "THINKING_MODE": thinking_mode,
            "REASONING_EFFORT": reasoning_effort,
            "API_INTERVAL_MS": str(api_interval_ms),
            "CPU_CORES": cpu_cores,
            "WORK_DIR": str(work_dir),
            "DATA_DIR": str(data_dir),
            "RESULTS_DIR": str(results_dir),
            "MAX_TOOL_CALLS": str(max_tool_calls),
        }
    )

    stdout_path = results_dir.parent / "run_eval.stdout.log"
    stderr_path = results_dir.parent / "run_eval.stderr.log"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        proc = subprocess.Popen(
            ["bash", str(bench_repo / "scripts" / "run_eval.sh")],
            cwd=bench_repo,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        returncode = proc.wait()
    elapsed = time.time() - started
    with stderr_path.open("a", encoding="utf-8") as stderr_handle:
        stderr_handle.write(f"\n[wall_clock_seconds]={elapsed:.2f}\n")
    return AttemptProcessResult(returncode=returncode, elapsed_secs=elapsed)


def load_eval_log(work_dir: Path) -> dict[str, Any] | None:
    path = work_dir / "eval_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_official_result(results_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(
        p for p in results_dir.glob("*.json") if p.name != "leaderboard.json"
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _benchmark_fields(payload: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(payload, dict):
        return 0.0, 0.0
    return (
        float(payload.get("qps", 0.0) or 0.0),
        float(payload.get("recall", 0.0) or 0.0),
    )


def attempt_outcome_from_logs(
    *,
    attempt_index: int,
    revision: RevisionConfig,
    process: AttemptProcessResult,
    work_dir: Path,
    results_dir: Path,
) -> AttemptOutcome:
    eval_log = load_eval_log(work_dir)
    result_payload = load_official_result(results_dir)
    best_qps, best_recall = _benchmark_fields((eval_log or {}).get("best_benchmark"))
    last_qps, last_recall = _benchmark_fields((eval_log or {}).get("last_benchmark"))
    qps = float((result_payload or {}).get("qps", 0.0) or 0.0)
    recall = float((result_payload or {}).get("recall", 0.0) or 0.0)
    recall_passed = bool((result_payload or {}).get("recall_passed", False))
    result_source = str((result_payload or {}).get("result_source", "none"))
    tool_calls_used = int((eval_log or {}).get("tool_calls_used", 0) or 0)
    tool_calls_total = int((eval_log or {}).get("tool_calls_total", 0) or 0)
    valid = process.returncode == 0 and recall_passed and qps > 0.0

    notes_parts: list[str] = []
    if revision.notes:
        notes_parts.append(revision.notes)
    if revision.official_tools_only:
        notes_parts.append("official_tools_only")
    if revision.added_helper_tools:
        notes_parts.append(f"added_helper_tools={','.join(revision.added_helper_tools)}")
    if revision.zero_completion_retry_limit > 0:
        notes_parts.append(f"zero_completion_retry_limit={revision.zero_completion_retry_limit}")
    if result_source and result_source != "none":
        notes_parts.append(f"result_source={result_source}")

    if process.returncode != 0:
        status = "error"
    elif valid:
        status = "valid"
    else:
        status = "invalid"

    return AttemptOutcome(
        attempt_index=attempt_index,
        revision_id=revision.revision_id,
        status=status,
        process_returncode=process.returncode,
        valid=valid,
        qps=qps,
        recall=recall,
        recall_passed=recall_passed,
        result_source=result_source,
        tool_calls_used=tool_calls_used,
        tool_calls_total=tool_calls_total,
        best_qps=best_qps,
        best_recall=best_recall,
        last_qps=last_qps,
        last_recall=last_recall,
        elapsed_secs=process.elapsed_secs,
        work_dir=str(work_dir),
        notes="; ".join(notes_parts),
    )


def attempt_row(outcome: AttemptOutcome) -> dict[str, Any]:
    return {
        "attempt": outcome.attempt_index,
        "revision_id": outcome.revision_id,
        "status": outcome.status,
        "process_returncode": outcome.process_returncode,
        "valid": str(outcome.valid).lower(),
        "qps": f"{outcome.qps:.2f}",
        "recall": f"{outcome.recall:.4f}",
        "recall_passed": str(outcome.recall_passed).lower(),
        "result_source": outcome.result_source,
        "tool_calls_used": outcome.tool_calls_used,
        "tool_calls_total": outcome.tool_calls_total,
        "best_qps": f"{outcome.best_qps:.2f}",
        "best_recall": f"{outcome.best_recall:.4f}",
        "last_qps": f"{outcome.last_qps:.2f}",
        "last_recall": f"{outcome.last_recall:.4f}",
        "elapsed_secs": f"{outcome.elapsed_secs:.2f}",
        "work_dir": outcome.work_dir,
        "notes": outcome.notes,
    }
