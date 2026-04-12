#!/usr/bin/env python3
"""Local multi-agent vector-db-bench harness orchestrated via Codex CLI.

Editable skeleton files are discovered at runtime (Rust sources + Cargo.toml/build.rs),
excluding protected API/server paths; see CONTRACT.md next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "codex_cli_runs"
DEFAULT_PROGRAM_PATH = Path(__file__).with_name("program.md")
DEFAULT_MEMORY_TEMPLATE = "# Memory\n\n- no prior rounds\n"
DEFAULT_REVIEWER_NOTES = "# Reviewer Notes\n\n- no reviewer notes yet\n"

# Protected skeleton paths: HTTP/API + server shell (see CONTRACT.md).
PROTECTED_SKELETON_PATHS: frozenset[str] = frozenset({"src/api.rs", "src/main.rs"})
SKIP_SKELETON_DIR_NAMES: frozenset[str] = frozenset({".git", "target", ".idea", ".vscode"})
# Shown to the coordinator/workers as read-only context.
READ_ONLY_CONTEXT_FILES: tuple[str, ...] = tuple(sorted(PROTECTED_SKELETON_PATHS))

SEEDED_BASELINE_FILES = {
    "Cargo.toml": """[package]
name = "vector-db-skeleton"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
lto = true
codegen-units = 1
debug = true
""",
    "src/db.rs": """use crate::api::*;
use crate::distance::l2_distance;
use std::cmp::Ordering;
use std::sync::RwLock;

const VECTOR_DIM: usize = 128;

struct Storage {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

pub struct VectorDB {
    storage: RwLock<Storage>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(Storage {
                ids: Vec::new(),
                vectors: Vec::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != VECTOR_DIM {
            return;
        }
        let mut storage = self.storage.write().unwrap();
        storage.ids.push(id);
        storage.vectors.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut storage = self.storage.write().unwrap();
        storage.ids.reserve(vectors.len());
        storage.vectors.reserve(vectors.len() * VECTOR_DIM);
        let mut inserted = 0usize;
        for (id, vector) in vectors {
            if vector.len() != VECTOR_DIM {
                continue;
            }
            storage.ids.push(id);
            storage.vectors.extend_from_slice(&vector);
            inserted += 1;
        }
        inserted
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        if query.len() != VECTOR_DIM || top_k == 0 {
            return Vec::new();
        }

        let storage = self.storage.read().unwrap();
        let count = storage.ids.len();
        if count == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(u64, f64)> = Vec::with_capacity(count);
        for (idx, id) in storage.ids.iter().copied().enumerate() {
            let start = idx * VECTOR_DIM;
            let end = start + VECTOR_DIM;
            let dist = l2_distance(query, &storage.vectors[start..end]);
            scored.push((id, dist));
        }

        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(top_k as usize)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}
""",
    "src/distance.rs": """pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (*x as f64) - (*y as f64);
            diff * diff
        })
        .sum()
}
""",
}


def _normalize_skeleton_rel_path(raw: str) -> str:
    s = raw.replace("\\", "/").strip()
    if not s or s.startswith("/"):
        raise ValueError(f"invalid skeleton-relative path: {raw!r}")
    parts: list[str] = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise ValueError(f"invalid skeleton-relative path (..): {raw!r}")
        parts.append(seg)
    return "/".join(parts)


def _is_allowed_mutable_rel_path(rel: str) -> bool:
    if not rel or rel in PROTECTED_SKELETON_PATHS:
        return False
    if any(part in SKIP_SKELETON_DIR_NAMES for part in rel.split("/")):
        return False
    if rel.startswith("target/"):
        return False
    if rel == "Cargo.toml" or rel == "build.rs":
        return True
    if rel.startswith("src/") and rel.endswith(".rs"):
        return True
    return False


def _discover_editable_skeleton_files(skeleton_dir: Path) -> dict[str, str]:
    skeleton_dir = skeleton_dir.resolve()
    if not skeleton_dir.is_dir():
        raise FileNotFoundError(f"missing skeleton directory: {skeleton_dir}")
    found: dict[str, str] = {}
    for path in sorted(skeleton_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skeleton_dir).as_posix()
        if rel in PROTECTED_SKELETON_PATHS:
            continue
        if any(part in SKIP_SKELETON_DIR_NAMES for part in path.relative_to(skeleton_dir).parts):
            continue
        if rel.endswith(".rs") or rel in ("Cargo.toml", "build.rs"):
            if _is_allowed_mutable_rel_path(rel):
                found[rel] = _read_text(path)
    if not found:
        raise FileNotFoundError(f"no editable files discovered under {skeleton_dir}")
    return found


def _needs_seeded_baseline(files: dict[str, str]) -> bool:
    return ("todo!(" in files.get("src/db.rs", "")) or ("todo!(" in files.get("src/distance.rs", ""))


RESULT_COLUMNS = [
    "attempt",
    "phase",
    "status",
    "candidate_sha256",
    "parent_sha256",
    "valid",
    "recall_passed",
    "anti_cheat_passed",
    "build_ok",
    "runtime_ok",
    "qps",
    "recall",
    "avg_latency_ms",
    "p95_latency_ms",
    "score",
    "failure_type",
    "runtime_seconds",
    "benchmark_duration_secs",
    "change_summary",
]


@dataclass(frozen=True)
class WorkerBrief:
    title: str
    family: str
    hypothesis: str
    instructions: str
    target_files: tuple[str, ...]


@dataclass(frozen=True)
class CodexExecResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    last_message: str
    runtime_seconds: float


@dataclass(frozen=True)
class BenchEvalResult:
    valid: bool
    build_ok: bool
    runtime_ok: bool
    recall_passed: bool
    anti_cheat_passed: bool
    qps: float
    recall: float
    avg_latency_ms: float
    p95_latency_ms: float
    benchmark_duration_secs: float
    failure_type: str | None
    message: str
    payload: dict[str, Any] | None
    runtime_seconds: float


@dataclass(frozen=True)
class EvalInputs:
    base_vectors: Path
    query_vectors: Path
    ground_truth: Path


@dataclass(frozen=True)
class RunConfig:
    bench_repo: Path
    skeleton_dir: Path
    benchmark_dir: Path
    benchmark_bin: Path
    server_bin_name: str
    server_url: str
    server_port: int
    cpu_cores: str
    build_timeout_seconds: int
    benchmark_timeout_seconds: int
    startup_timeout_seconds: int
    proxy_inputs: EvalInputs
    strict_inputs: EvalInputs
    concurrency: int
    warmup: int
    recall_threshold: float
    seed: int
    codex_executable: str
    codex_timeout_seconds: int
    codex_sandbox: str
    codex_model: str
    codex_oss: bool
    codex_local_provider: str
    modal_show_output: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True, help="Path to a local clone of KCORES/vector-db-bench")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--program-path", type=Path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers-per-round", type=int, default=3)
    parser.add_argument("--strict-top-k", type=int, default=1)
    parser.add_argument("--recall-threshold", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proxy-max-queries", type=int, default=2000)
    parser.add_argument("--strict-max-queries", type=int, default=0, help="0 means use all queries")
    parser.add_argument("--build-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--server-bin-name", type=str, default="vector-db-skeleton")
    parser.add_argument("--benchmark-bin-name", type=str, default="vector-db-benchmark")
    parser.add_argument("--cpu-cores", type=str, default="", help="Optional taskset CPU core list, e.g. 0-3")
    parser.add_argument("--base-vectors", type=Path, default=None)
    parser.add_argument("--query-vectors", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument(
        "--apply-incumbent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write final incumbent editable files back into bench_repo/skeleton (protected paths untouched).",
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
        default="workspace-write",
    )
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=60 * 15,
        help="Per-Codex coordinator/worker/reviewer timeout in seconds.",
    )
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved compatibility flag; local harness does not use Modal.",
    )
    return parser.parse_args()


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
    writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def _close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def _require_codex(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Could not find Codex CLI executable {executable!r} on PATH")
    return resolved


def _parse_server_port(server_url: str) -> int:
    parsed = urlparse(server_url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"server URL must include host and port: {server_url!r}")
    return int(parsed.port)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _surface_sha(files: dict[str, str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for relpath in sorted(files):
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relpath].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _bootstrap_seed_surface(skeleton_dir: Path) -> dict[str, str]:
    files = _discover_editable_skeleton_files(skeleton_dir)
    if _needs_seeded_baseline(files):
        merged = dict(files)
        for relpath, content in SEEDED_BASELINE_FILES.items():
            merged[relpath] = content.rstrip() + "\n"
        return merged
    return {relpath: content.rstrip() + "\n" for relpath, content in files.items()}


def _read_readonly_context(skeleton_dir: Path) -> str:
    parts: list[str] = []
    for relpath in READ_ONLY_CONTEXT_FILES:
        path = skeleton_dir / relpath
        if path.exists():
            parts.append(f"## {relpath}\n```rust\n{_read_text(path).rstrip()}\n```")
    return "\n\n".join(parts) or "(no read-only context files found)"


def _materialize_workspace(*, skeleton_dir: Path, workspace_dir: Path, files: dict[str, str]) -> None:
    workspace_dir = workspace_dir.resolve()
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(skeleton_dir, workspace_dir)
    for relpath, content in files.items():
        dest = workspace_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def _maybe_make_subset(source: Path, destination: Path, max_items: int) -> Path:
    if max_items <= 0:
        return source
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload[:max_items]) + "\n", encoding="utf-8")
    return destination


def _resolve_base_vectors_file(bench_repo: Path, run_dir: Path) -> Path:
    data_dir = bench_repo / "data"
    single_file = data_dir / "base_vectors.json"
    if single_file.exists():
        return single_file

    shard_paths = sorted(data_dir.glob("base_vectors_*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            f"Could not find base_vectors.json or base_vectors_*.json under {data_dir}"
        )

    merged_path = run_dir / "benchmark_inputs" / "base_vectors_merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        return merged_path

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
    return merged_path


def _json_schema_for_briefs(worker_count: int, *, incumbent_paths: Sequence[str]) -> dict[str, Any]:
    path_enum = sorted(incumbent_paths)
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
                    "required": ["title", "family", "hypothesis", "instructions", "target_files"],
                    "properties": {
                        "title": {"type": "string"},
                        "family": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "instructions": {"type": "string"},
                        "target_files": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": path_enum},
                        },
                    },
                },
            }
        },
    }


def _json_schema_for_worker_output(*, incumbent_paths: Sequence[str]) -> dict[str, Any]:
    sorted_paths = sorted(incumbent_paths)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "files"],
        "properties": {
            "summary": {"type": "string"},
            "files": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "required": sorted_paths,
                "properties": {relpath: {"type": "string"} for relpath in sorted_paths},
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


def _parse_worker_briefs(
    raw_text: str, expected_count: int, *, allowed_target_paths: frozenset[str]
) -> list[WorkerBrief]:
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
        target_files = item.get("target_files")
        if not isinstance(target_files, list) or not target_files:
            raise ValueError("target_files must be a non-empty list")
        normalized: list[str] = []
        for raw in target_files:
            rel = _normalize_skeleton_rel_path(str(raw))
            if rel not in allowed_target_paths:
                raise ValueError(f"worker brief used unknown target file(s): {raw!r}")
            normalized.append(rel)
        target_tuple = tuple(normalized)
        brief = WorkerBrief(
            title=str(item.get("title", "")).strip(),
            family=str(item.get("family", "")).strip(),
            hypothesis=str(item.get("hypothesis", "")).strip(),
            instructions=str(item.get("instructions", "")).strip(),
            target_files=target_tuple,
        )
        if not all((brief.title, brief.family, brief.hypothesis, brief.instructions)):
            raise ValueError("worker brief fields must all be non-empty")
        briefs.append(brief)
    return briefs


def _parse_worker_output(raw_text: str, *, incumbent_files: dict[str, str]) -> tuple[str, dict[str, str]]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("worker output must be a JSON object")
    summary = str(payload.get("summary", "")).strip()
    files_payload = payload.get("files")
    if not summary:
        raise ValueError("worker output missing summary")
    if not isinstance(files_payload, dict):
        raise ValueError("worker output missing files object")
    parsed: dict[str, str] = {}
    for raw_key, content in files_payload.items():
        rel = _normalize_skeleton_rel_path(str(raw_key))
        if not _is_allowed_mutable_rel_path(rel):
            raise ValueError(f"worker output used disallowed path: {raw_key!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"worker output missing non-empty file content for {rel}")
        parsed[rel] = content.rstrip() + "\n"
    merged: dict[str, str] = dict(incumbent_files)
    for rel in incumbent_files:
        if rel not in parsed:
            raise ValueError(f"worker output must include every incumbent path; missing {rel}")
    merged.update(parsed)
    return summary, merged


def _review_notes_text(payload: dict[str, Any]) -> str:
    lines = ["# Reviewer Notes", ""]
    lines.append(f"- summary: {payload.get('summary', 'none')}")
    lines.append(f"- recommended_attempt: {payload.get('recommended_attempt')}")
    promising = payload.get("promising_attempts", []) or []
    lines.append(f"- promising_attempts: {', '.join(str(x) for x in promising) if promising else 'none'}")
    lines.append("")
    lines.append("## Lessons")
    lessons = payload.get("lessons") or []
    if lessons:
        for lesson in lessons:
            lines.append(f"- {lesson}")
    else:
        lines.append("- no lessons yet")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _score_result(result: BenchEvalResult) -> float:
    if result.valid:
        return 1_000_000.0 + result.qps
    return result.recall * 1000.0 + max(result.qps, 0.0) * 1e-3


def _is_better(new: BenchEvalResult, old: BenchEvalResult) -> bool:
    if new.valid != old.valid:
        return new.valid
    if new.valid and old.valid:
        if abs(new.qps - old.qps) > 1e-9:
            return new.qps > old.qps
        return new.recall > old.recall
    if abs(new.recall - old.recall) > 1e-9:
        return new.recall > old.recall
    return new.qps > old.qps


def _eval_row(
    result: BenchEvalResult,
    *,
    attempt: int,
    phase: str,
    status: str,
    candidate_sha: str,
    parent_sha: str,
    change_summary: str,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "phase": phase,
        "status": status,
        "candidate_sha256": candidate_sha,
        "parent_sha256": parent_sha,
        "valid": result.valid,
        "recall_passed": result.recall_passed,
        "anti_cheat_passed": result.anti_cheat_passed,
        "build_ok": result.build_ok,
        "runtime_ok": result.runtime_ok,
        "qps": result.qps,
        "recall": result.recall,
        "avg_latency_ms": result.avg_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "score": _score_result(result),
        "failure_type": result.failure_type,
        "runtime_seconds": result.runtime_seconds,
        "benchmark_duration_secs": result.benchmark_duration_secs,
        "change_summary": change_summary,
    }


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    elapsed = time.time() - started
    stderr_path.write_text(proc.stderr + f"\n[wall_clock_seconds]={elapsed:.2f}\n", encoding="utf-8")
    return proc


def _wait_for_server(host: str, port: int, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("benchmark produced empty stdout")
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("benchmark stdout did not contain JSON")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("benchmark JSON root must be an object")
    return payload


def _evaluate_candidate(
    *,
    candidate_files: dict[str, str],
    workspace_dir: Path,
    eval_dir: Path,
    config: RunConfig,
    inputs: EvalInputs,
) -> BenchEvalResult:
    workspace_dir = workspace_dir.resolve()
    eval_dir = eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    _materialize_workspace(skeleton_dir=config.skeleton_dir, workspace_dir=workspace_dir, files=candidate_files)

    build_cmd = ["cargo", "build", "--release"]
    build_stdout_path = eval_dir / "build.stdout.log"
    build_stderr_path = eval_dir / "build.stderr.log"
    try:
        build_proc = _run_command(
            build_cmd,
            cwd=workspace_dir,
            timeout_seconds=config.build_timeout_seconds,
            stdout_path=build_stdout_path,
            stderr_path=build_stderr_path,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        if isinstance(stdout_text, bytes):
            stdout_text = stdout_text.decode("utf-8", errors="replace")
        if isinstance(stderr_text, bytes):
            stderr_text = stderr_text.decode("utf-8", errors="replace")
        build_stdout_path.write_text(stdout_text, encoding="utf-8")
        build_stderr_path.write_text(
            stderr_text + f"\n[timeout_seconds]={config.build_timeout_seconds}\n",
            encoding="utf-8",
        )
        return BenchEvalResult(
            valid=False,
            build_ok=False,
            runtime_ok=False,
            recall_passed=False,
            anti_cheat_passed=False,
            qps=0.0,
            recall=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            benchmark_duration_secs=0.0,
            failure_type="build_timeout",
            message=f"cargo build timed out after {config.build_timeout_seconds}s",
            payload=None,
            runtime_seconds=time.time() - started_at,
        )
    if build_proc.returncode != 0:
        return BenchEvalResult(
            valid=False,
            build_ok=False,
            runtime_ok=False,
            recall_passed=False,
            anti_cheat_passed=False,
            qps=0.0,
            recall=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            benchmark_duration_secs=0.0,
            failure_type="build_error",
            message=_tail(build_proc.stderr or build_proc.stdout),
            payload=None,
            runtime_seconds=time.time() - started_at,
        )

    server_cmd: list[str] = []
    if config.cpu_cores:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise FileNotFoundError("taskset not found but --cpu-cores was provided")
        server_cmd.extend([taskset, "-c", config.cpu_cores])
    server_cmd.append(str(workspace_dir / "target" / "release" / config.server_bin_name))

    server_stdout = (eval_dir / "server.stdout.log").open("w", encoding="utf-8")
    server_stderr = (eval_dir / "server.stderr.log").open("w", encoding="utf-8")
    server_proc: subprocess.Popen[str] | None = None
    try:
        server_proc = subprocess.Popen(server_cmd, cwd=workspace_dir, stdout=server_stdout, stderr=server_stderr, text=True)
        parsed = urlparse(config.server_url)
        if not _wait_for_server(parsed.hostname or "127.0.0.1", config.server_port, config.startup_timeout_seconds):
            return BenchEvalResult(
                valid=False,
                build_ok=True,
                runtime_ok=False,
                recall_passed=False,
                anti_cheat_passed=False,
                qps=0.0,
                recall=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                benchmark_duration_secs=0.0,
                failure_type="server_startup_timeout",
                message=f"server did not accept connections on {config.server_url}",
                payload=None,
                runtime_seconds=time.time() - started_at,
            )

        benchmark_cmd = [
            str(config.benchmark_bin),
            "--server-url",
            config.server_url,
            "--concurrency",
            str(config.concurrency),
            "--warmup",
            str(config.warmup),
            "--base-vectors",
            str(inputs.base_vectors),
            "--query-vectors",
            str(inputs.query_vectors),
            "--ground-truth",
            str(inputs.ground_truth),
            "--recall-threshold",
            str(config.recall_threshold),
            "--seed",
            str(config.seed),
        ]
        benchmark_stdout_path = eval_dir / "benchmark.stdout.log"
        benchmark_stderr_path = eval_dir / "benchmark.stderr.log"
        try:
            bench_proc = _run_command(
                benchmark_cmd,
                cwd=config.benchmark_dir,
                timeout_seconds=config.benchmark_timeout_seconds,
                stdout_path=benchmark_stdout_path,
                stderr_path=benchmark_stderr_path,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_text = exc.stdout or ""
            stderr_text = exc.stderr or ""
            if isinstance(stdout_text, bytes):
                stdout_text = stdout_text.decode("utf-8", errors="replace")
            if isinstance(stderr_text, bytes):
                stderr_text = stderr_text.decode("utf-8", errors="replace")
            benchmark_stdout_path.write_text(stdout_text, encoding="utf-8")
            benchmark_stderr_path.write_text(
                stderr_text + f"\n[timeout_seconds]={config.benchmark_timeout_seconds}\n",
                encoding="utf-8",
            )
            return BenchEvalResult(
                valid=False,
                build_ok=True,
                runtime_ok=False,
                recall_passed=False,
                anti_cheat_passed=False,
                qps=0.0,
                recall=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                benchmark_duration_secs=0.0,
                failure_type="benchmark_timeout",
                message=f"benchmark timed out after {config.benchmark_timeout_seconds}s",
                payload=None,
                runtime_seconds=time.time() - started_at,
            )
        if bench_proc.returncode != 0:
            return BenchEvalResult(
                valid=False,
                build_ok=True,
                runtime_ok=False,
                recall_passed=False,
                anti_cheat_passed=False,
                qps=0.0,
                recall=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                benchmark_duration_secs=0.0,
                failure_type="benchmark_error",
                message=_tail(bench_proc.stderr or bench_proc.stdout),
                payload=None,
                runtime_seconds=time.time() - started_at,
            )
        payload = _extract_json_payload(bench_proc.stdout)
        benchmark = payload.get("benchmark") or {}
        anti_cheat = payload.get("anti_cheat") or {}
        recall = float(benchmark.get("recall", 0.0) or 0.0)
        qps = float(benchmark.get("qps", 0.0) or 0.0)
        recall_passed = bool(benchmark.get("recall_passed", False))
        anti_cheat_passed = bool(anti_cheat.get("passed", False))
        valid = bool(recall_passed and anti_cheat_passed)
        message = (
            f"qps={qps:.2f} recall={recall:.4f} "
            f"recall_passed={recall_passed} anti_cheat_passed={anti_cheat_passed}"
        )
        return BenchEvalResult(
            valid=valid,
            build_ok=True,
            runtime_ok=True,
            recall_passed=recall_passed,
            anti_cheat_passed=anti_cheat_passed,
            qps=qps,
            recall=recall,
            avg_latency_ms=float(benchmark.get("avg_latency_ms", 0.0) or 0.0),
            p95_latency_ms=float(benchmark.get("p95_latency_ms", 0.0) or 0.0),
            benchmark_duration_secs=float(benchmark.get("duration_secs", 0.0) or 0.0),
            failure_type=None if valid else "constraint_failed",
            message=message,
            payload=payload,
            runtime_seconds=time.time() - started_at,
        )
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5)
        server_stdout.close()
        server_stderr.close()


def _prepare_inputs(args: argparse.Namespace, run_dir: Path) -> tuple[EvalInputs, EvalInputs]:
    base_vectors = (args.base_vectors or _resolve_base_vectors_file(args.bench_repo, run_dir)).resolve()
    query_vectors = (args.query_vectors or (args.bench_repo / "data" / "query_vectors.json")).resolve()
    ground_truth = (args.ground_truth or (args.bench_repo / "data" / "ground_truth.json")).resolve()
    for path in (base_vectors, query_vectors, ground_truth):
        if not path.exists():
            raise FileNotFoundError(f"required benchmark data file not found: {path}")
    input_dir = run_dir / "benchmark_inputs"
    proxy_inputs = EvalInputs(
        base_vectors=base_vectors,
        query_vectors=_maybe_make_subset(query_vectors, input_dir / f"query_vectors_proxy_{args.proxy_max_queries}.json", args.proxy_max_queries).resolve(),
        ground_truth=_maybe_make_subset(ground_truth, input_dir / f"ground_truth_proxy_{args.proxy_max_queries}.json", args.proxy_max_queries).resolve(),
    )
    strict_inputs = EvalInputs(
        base_vectors=base_vectors,
        query_vectors=_maybe_make_subset(query_vectors, input_dir / f"query_vectors_strict_{args.strict_max_queries}.json", args.strict_max_queries).resolve(),
        ground_truth=_maybe_make_subset(ground_truth, input_dir / f"ground_truth_strict_{args.strict_max_queries}.json", args.strict_max_queries).resolve(),
    )
    return proxy_inputs, strict_inputs


def _ensure_benchmark_binary(benchmark_dir: Path, benchmark_bin_name: str, timeout_seconds: int) -> Path:
    proc = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to build benchmark crate: {_tail(proc.stderr or proc.stdout)}")
    benchmark_bin = benchmark_dir / "target" / "release" / benchmark_bin_name
    if not benchmark_bin.exists():
        raise FileNotFoundError(f"benchmark binary not found after build: {benchmark_bin}")
    return benchmark_bin


def _coordinator_prompt(
    *,
    program_text: str,
    memory_text: str,
    reviewer_notes_text: str,
    mutable_files: dict[str, str],
    readonly_context: str,
    recent_rows: list[dict[str, Any]],
    workers_per_round: int,
) -> str:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['phase']} {row['status']}]: qps={row.get('qps')} recall={row.get('recall')} "
        f"valid={row.get('valid')} failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in recent_rows[-10:]
    ) or "- no recent attempts"
    editable_list = ", ".join(sorted(mutable_files.keys()))
    mutable_text = "\n\n".join(
        f"## {relpath}\n```{ 'toml' if relpath.endswith('.toml') else 'rust' }\n{content.rstrip()}\n```"
        for relpath, content in sorted(mutable_files.items())
    )
    return (
        "You are coordinating a small local multi-agent coding batch for vector-db-bench.\n\n"
        "Return a JSON object with one key, 'briefs', whose value is an array of exactly "
        f"{workers_per_round} distinct worker briefs.\n"
        "Each brief must contain: title, family, hypothesis, instructions, target_files.\n"
        f"target_files entries must be drawn from the current editable paths: {editable_list}.\n"
        "You may assign workers to different subsets (e.g. distance kernel vs db layout vs new src modules).\n"
        "Do not propose edits to protected files (see read-only context): they are not in the list above.\n"
        "Bias toward recall-safe throughput: memory layout, distance kernel, top-k, indexing (IVF/HNSW/etc.), "
        "parallelism, and release/profile tuning.\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Reviewer notes:\n{reviewer_notes_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Read-only interface context:\n{readonly_context}\n\n"
        f"Current mutable files:\n{mutable_text}\n\n"
        "Return only JSON."
    )


def _worker_prompt(
    *,
    program_text: str,
    memory_text: str,
    reviewer_notes_text: str,
    mutable_files: dict[str, str],
    readonly_context: str,
    recent_rows: list[dict[str, Any]],
    brief: WorkerBrief,
) -> str:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: qps={row.get('qps')} recall={row.get('recall')} "
        f"failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in recent_rows[-8:]
    ) or "- no recent attempts"
    editable_sorted = sorted(mutable_files.keys())
    mutable_text = "\n\n".join(
        f"## {relpath}\n```{ 'toml' if relpath.endswith('.toml') else 'rust' }\n{content.rstrip()}\n```"
        for relpath, content in sorted(mutable_files.items())
    )
    return (
        "You are one worker in a local Codex CLI optimization harness for vector-db-bench.\n"
        "Return a JSON object with keys 'summary' and 'files'.\n"
        "The 'files' object must include full updated contents for EVERY current editable path "
        f"(listed below: {', '.join(editable_sorted)}).\n"
        "You may ADD new editable Rust modules under src/ (e.g. src/ivf.rs) by including extra keys in 'files'; "
        "new files must be wired from db.rs/Cargo.toml as needed and must stay on the allowed paths "
        "(src/**/*.rs, Cargo.toml, build.rs only).\n"
        "Do not change protected API/server files (see read-only context). Preserve recall >= threshold and anti-cheat.\n"
        "Your code must compile in Rust release mode.\n\n"
        f"Assigned brief:\n"
        f"- title: {brief.title}\n"
        f"- family: {brief.family}\n"
        f"- hypothesis: {brief.hypothesis}\n"
        f"- instructions: {brief.instructions}\n"
        f"- target_files: {', '.join(brief.target_files)}\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Reviewer notes:\n{reviewer_notes_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Read-only interface context:\n{readonly_context}\n\n"
        f"Current mutable files:\n{mutable_text}\n\n"
        "Return only JSON."
    )


def _reviewer_prompt(
    *,
    program_text: str,
    memory_text: str,
    incumbent_proxy: BenchEvalResult,
    incumbent_strict: BenchEvalResult,
    round_rows: list[dict[str, Any]],
) -> str:
    attempts_text = "\n".join(
        f"- attempt {row['attempt']} phase={row['phase']} status={row['status']} qps={row.get('qps')} recall={row.get('recall')} valid={row.get('valid')} failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in round_rows
        if int(row["attempt"]) != 0
    ) or "- no attempts"
    return (
        "You are reviewing one vector-db-bench optimization round.\n"
        "Return concise JSON with summary, lessons, promising_attempts, and recommended_attempt.\n"
        "Lessons should improve the next coordinator batch.\n\n"
        f"Program instructions:\n{program_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Incumbent proxy: qps={incumbent_proxy.qps} recall={incumbent_proxy.recall} valid={incumbent_proxy.valid}\n"
        f"Incumbent strict: qps={incumbent_strict.qps} recall={incumbent_strict.recall} valid={incumbent_strict.valid}\n\n"
        f"Round attempts:\n{attempts_text}\n\n"
        "Return only JSON."
    )


def _update_memory(memory_path: Path, round_idx: int, kept_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> None:
    lines = [memory_path.read_text(encoding="utf-8").rstrip(), "", f"## Round {round_idx}"]
    if kept_rows:
        for row in kept_rows:
            lines.append(
                f"- kept attempt {row['attempt']}: qps={row.get('qps')} recall={row.get('recall')} summary={row.get('change_summary')}"
            )
    else:
        lines.append("- no kept attempts")
    rejected_slice = rejected_rows[-5:]
    for row in rejected_slice:
        lines.append(
            f"- rejected attempt {row['attempt']}: phase={row.get('phase')} failure={row.get('failure_type')} qps={row.get('qps')} recall={row.get('recall')} summary={row.get('change_summary')}"
        )
    memory_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    incumbent_files: dict[str, str],
) -> tuple[int, CodexExecResult]:
    schema_path = round_dir / f"worker_{attempt:02d}.schema.json"
    _write_json(
        schema_path,
        _json_schema_for_worker_output(incumbent_paths=sorted(incumbent_files.keys())),
    )
    result = _run_codex_exec(
        executable=codex_executable,
        prompt=prompt,
        cwd=cwd,
        output_path=round_dir / f"worker_{attempt:02d}.output.json",
        timeout_seconds=codex_timeout_seconds,
        sandbox=codex_sandbox,
        model=codex_model,
        use_oss=codex_oss,
        local_provider=codex_local_provider,
        schema_path=schema_path,
    )
    return attempt, result


def _prepare_config(args: argparse.Namespace) -> RunConfig:
    args.run_dir = args.run_dir.resolve()
    bench_repo = args.bench_repo.resolve()
    skeleton_dir = bench_repo / "skeleton"
    benchmark_dir = bench_repo / "benchmark"
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"missing skeleton directory: {skeleton_dir}")
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"missing benchmark directory: {benchmark_dir}")
    benchmark_bin = _ensure_benchmark_binary(benchmark_dir, args.benchmark_bin_name, args.build_timeout_seconds)
    proxy_inputs, strict_inputs = _prepare_inputs(args, args.run_dir)
    server_port = _parse_server_port(args.server_url)
    return RunConfig(
        bench_repo=bench_repo,
        skeleton_dir=skeleton_dir,
        benchmark_dir=benchmark_dir,
        benchmark_bin=benchmark_bin,
        server_bin_name=args.server_bin_name,
        server_url=args.server_url,
        server_port=server_port,
        cpu_cores=args.cpu_cores,
        build_timeout_seconds=args.build_timeout_seconds,
        benchmark_timeout_seconds=args.benchmark_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        proxy_inputs=proxy_inputs,
        strict_inputs=strict_inputs,
        concurrency=args.concurrency,
        warmup=args.warmup,
        recall_threshold=args.recall_threshold,
        seed=args.seed,
        codex_executable=_require_codex(args.codex_executable),
        codex_timeout_seconds=args.codex_timeout_seconds,
        codex_sandbox=args.codex_sandbox,
        codex_model=args.codex_model,
        codex_oss=args.codex_oss,
        codex_local_provider=args.codex_local_provider,
        modal_show_output=args.modal_show_output,
    )


def _run_harness(args: argparse.Namespace) -> int:
    config = _prepare_config(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    incumbent_files = _bootstrap_seed_surface(config.skeleton_dir)
    incumbent_sha = _surface_sha(incumbent_files)
    program_text = _read_text(args.program_path)
    memory_path = args.run_dir / "memory.md"
    reviewer_notes_path = args.run_dir / "reviewer_notes.md"
    if not memory_path.exists():
        memory_path.write_text(DEFAULT_MEMORY_TEMPLATE, encoding="utf-8")
    if not reviewer_notes_path.exists():
        reviewer_notes_path.write_text(DEFAULT_REVIEWER_NOTES, encoding="utf-8")
    (args.run_dir / "initial_candidate").mkdir(exist_ok=True)
    for relpath, content in incumbent_files.items():
        dest = args.run_dir / "initial_candidate" / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    args.run_dir.joinpath("program.md").write_text(program_text, encoding="utf-8")

    recent_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    readonly_context = _read_readonly_context(config.skeleton_dir)

    print("[vector-db] evaluating incumbent seed candidate", flush=True)
    incumbent_proxy = _evaluate_candidate(
        candidate_files=incumbent_files,
        workspace_dir=args.run_dir / "seed_proxy_workspace",
        eval_dir=args.run_dir / "seed_proxy_eval",
        config=config,
        inputs=config.proxy_inputs,
    )
    _write_json(args.run_dir / "seed_proxy_eval.json", {
        **({} if incumbent_proxy.payload is None else incumbent_proxy.payload),
        "message": incumbent_proxy.message,
        "valid": incumbent_proxy.valid,
        "qps": incumbent_proxy.qps,
        "recall": incumbent_proxy.recall,
        "runtime_seconds": incumbent_proxy.runtime_seconds,
        "failure_type": incumbent_proxy.failure_type,
        "build_ok": incumbent_proxy.build_ok,
        "runtime_ok": incumbent_proxy.runtime_ok,
        "anti_cheat_passed": incumbent_proxy.anti_cheat_passed,
    })
    print("[vector-db] verifying incumbent seed candidate with strict eval", flush=True)
    incumbent_strict = _evaluate_candidate(
        candidate_files=incumbent_files,
        workspace_dir=args.run_dir / "seed_strict_workspace",
        eval_dir=args.run_dir / "seed_strict_eval",
        config=config,
        inputs=config.strict_inputs,
    )
    _write_json(args.run_dir / "seed_strict_eval.json", {
        **({} if incumbent_strict.payload is None else incumbent_strict.payload),
        "message": incumbent_strict.message,
        "valid": incumbent_strict.valid,
        "qps": incumbent_strict.qps,
        "recall": incumbent_strict.recall,
        "runtime_seconds": incumbent_strict.runtime_seconds,
        "failure_type": incumbent_strict.failure_type,
        "build_ok": incumbent_strict.build_ok,
        "runtime_ok": incumbent_strict.runtime_ok,
        "anti_cheat_passed": incumbent_strict.anti_cheat_passed,
    })

    round_summaries: list[dict[str, Any]] = []
    for round_idx in range(1, args.rounds + 1):
        round_started_at = time.time()
        round_dir = args.run_dir / f"round_{round_idx:02d}"
        attempts_dir = round_dir / "attempts"
        round_dir.mkdir(parents=True, exist_ok=True)
        attempts_dir.mkdir(parents=True, exist_ok=True)
        results_writer = _build_results_writer(round_dir / "results.tsv")
        round_rows: list[dict[str, Any]] = []
        try:
            for row in (
                _eval_row(incumbent_proxy, attempt=0, phase="baseline_proxy_record", status="loaded", candidate_sha=incumbent_sha, parent_sha="", change_summary="loaded incumbent proxy"),
                _eval_row(incumbent_strict, attempt=0, phase="baseline_strict_record", status="loaded", candidate_sha=incumbent_sha, parent_sha="", change_summary="loaded incumbent strict"),
            ):
                results_writer.writerow(row)
                round_rows.append(row)
            getattr(results_writer, "_handle").flush()

            coordinator_prompt = _coordinator_prompt(
                program_text=program_text,
                memory_text=memory_path.read_text(encoding="utf-8"),
                reviewer_notes_text=reviewer_notes_path.read_text(encoding="utf-8"),
                mutable_files=incumbent_files,
                readonly_context=readonly_context,
                recent_rows=recent_rows,
                workers_per_round=args.workers_per_round,
            )
            (round_dir / "coordinator_prompt.txt").write_text(coordinator_prompt, encoding="utf-8")
            coordinator_schema = round_dir / "coordinator.schema.json"
            _write_json(
                coordinator_schema,
                _json_schema_for_briefs(
                    args.workers_per_round,
                    incumbent_paths=sorted(incumbent_files.keys()),
                ),
            )
            coordinator_exec = _run_codex_exec(
                executable=config.codex_executable,
                prompt=coordinator_prompt,
                cwd=REPO_ROOT,
                output_path=round_dir / "coordinator_output.json",
                timeout_seconds=config.codex_timeout_seconds,
                sandbox=config.codex_sandbox,
                model=config.codex_model,
                use_oss=config.codex_oss,
                local_provider=config.codex_local_provider,
                schema_path=coordinator_schema,
            )
            (round_dir / "coordinator.stdout.log").write_text(coordinator_exec.stdout, encoding="utf-8")
            (round_dir / "coordinator.stderr.log").write_text(coordinator_exec.stderr, encoding="utf-8")
            if coordinator_exec.returncode != 0:
                raise RuntimeError(f"coordinator failed: {_tail(coordinator_exec.stderr or coordinator_exec.stdout)}")
            worker_briefs = _parse_worker_briefs(
                coordinator_exec.last_message,
                args.workers_per_round,
                allowed_target_paths=frozenset(incumbent_files.keys()),
            )
            _write_json(round_dir / "worker_briefs.json", {"briefs": [brief.__dict__ for brief in worker_briefs]})

            proposal_results: dict[int, tuple[WorkerBrief, CodexExecResult | Exception]] = {}
            with ThreadPoolExecutor(max_workers=args.workers_per_round) as executor:
                future_map = {}
                for attempt, brief in enumerate(worker_briefs, start=1):
                    prompt = _worker_prompt(
                        program_text=program_text,
                        memory_text=memory_path.read_text(encoding="utf-8"),
                        reviewer_notes_text=reviewer_notes_path.read_text(encoding="utf-8"),
                        mutable_files=incumbent_files,
                        readonly_context=readonly_context,
                        recent_rows=recent_rows,
                        brief=brief,
                    )
                    (round_dir / f"worker_{attempt:02d}.prompt.txt").write_text(prompt, encoding="utf-8")
                    future = executor.submit(
                        _worker_task,
                        codex_executable=config.codex_executable,
                        cwd=REPO_ROOT,
                        round_dir=round_dir,
                        attempt=attempt,
                        prompt=prompt,
                        codex_timeout_seconds=config.codex_timeout_seconds,
                        codex_sandbox=config.codex_sandbox,
                        codex_model=config.codex_model,
                        codex_oss=config.codex_oss,
                        codex_local_provider=config.codex_local_provider,
                        incumbent_files=incumbent_files,
                    )
                    future_map[future] = (attempt, brief)
                for future in as_completed(future_map):
                    attempt, brief = future_map[future]
                    try:
                        _, exec_result = future.result()
                        proposal_results[attempt] = (brief, exec_result)
                    except Exception as exc:
                        proposal_results[attempt] = (brief, exc)

            proxy_candidates: list[tuple[int, str, dict[str, str], str]] = []
            seen_shas: set[str] = {incumbent_sha}
            for attempt in range(1, args.workers_per_round + 1):
                brief, proposal = proposal_results[attempt]
                attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                _write_json(attempt_dir / "worker_brief.json", {**brief.__dict__, "target_files": list(brief.target_files)})
                if isinstance(proposal, Exception):
                    row = {
                        "attempt": attempt,
                        "phase": "proposal",
                        "status": "crash",
                        "candidate_sha256": "",
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "recall_passed": False,
                        "anti_cheat_passed": False,
                        "build_ok": False,
                        "runtime_ok": False,
                        "qps": None,
                        "recall": None,
                        "avg_latency_ms": None,
                        "p95_latency_ms": None,
                        "score": 0.0,
                        "failure_type": proposal.__class__.__name__,
                        "runtime_seconds": None,
                        "benchmark_duration_secs": None,
                        "change_summary": str(proposal),
                    }
                    results_writer.writerow(row)
                    round_rows.append(row)
                    rejected_rows.append(row)
                    continue
                (attempt_dir / "worker.stdout.log").write_text(proposal.stdout, encoding="utf-8")
                (attempt_dir / "worker.stderr.log").write_text(proposal.stderr, encoding="utf-8")
                if proposal.returncode != 0:
                    row = {
                        "attempt": attempt,
                        "phase": "proposal",
                        "status": "crash",
                        "candidate_sha256": "",
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "recall_passed": False,
                        "anti_cheat_passed": False,
                        "build_ok": False,
                        "runtime_ok": False,
                        "qps": None,
                        "recall": None,
                        "avg_latency_ms": None,
                        "p95_latency_ms": None,
                        "score": 0.0,
                        "failure_type": "codex_exec_error",
                        "runtime_seconds": proposal.runtime_seconds,
                        "benchmark_duration_secs": None,
                        "change_summary": _tail(proposal.stderr or proposal.stdout),
                    }
                    results_writer.writerow(row)
                    round_rows.append(row)
                    rejected_rows.append(row)
                    continue
                try:
                    summary, candidate_files = _parse_worker_output(
                        proposal.last_message,
                        incumbent_files=incumbent_files,
                    )
                except Exception as exc:
                    row = {
                        "attempt": attempt,
                        "phase": "proposal",
                        "status": "crash",
                        "candidate_sha256": "",
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "recall_passed": False,
                        "anti_cheat_passed": False,
                        "build_ok": False,
                        "runtime_ok": False,
                        "qps": None,
                        "recall": None,
                        "avg_latency_ms": None,
                        "p95_latency_ms": None,
                        "score": 0.0,
                        "failure_type": exc.__class__.__name__,
                        "runtime_seconds": proposal.runtime_seconds,
                        "benchmark_duration_secs": None,
                        "change_summary": str(exc),
                    }
                    results_writer.writerow(row)
                    round_rows.append(row)
                    rejected_rows.append(row)
                    continue
                candidate_sha = _surface_sha(candidate_files)
                candidate_dir = attempt_dir / "candidate"
                for relpath, content in candidate_files.items():
                    dest = candidate_dir / relpath
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                (attempt_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
                if candidate_sha in seen_shas:
                    row = {
                        "attempt": attempt,
                        "phase": "proposal",
                        "status": "discard",
                        "candidate_sha256": candidate_sha,
                        "parent_sha256": incumbent_sha,
                        "valid": False,
                        "recall_passed": False,
                        "anti_cheat_passed": False,
                        "build_ok": False,
                        "runtime_ok": False,
                        "qps": None,
                        "recall": None,
                        "avg_latency_ms": None,
                        "p95_latency_ms": None,
                        "score": 0.0,
                        "failure_type": "duplicate_candidate",
                        "runtime_seconds": proposal.runtime_seconds,
                        "benchmark_duration_secs": None,
                        "change_summary": summary,
                    }
                    results_writer.writerow(row)
                    round_rows.append(row)
                    rejected_rows.append(row)
                    continue
                seen_shas.add(candidate_sha)
                proxy_candidates.append((attempt, summary, candidate_files, candidate_sha))

            proxy_scored: list[tuple[int, str, dict[str, str], str, BenchEvalResult]] = []
            for attempt, summary, candidate_files, candidate_sha in proxy_candidates:
                attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
                proxy_result = _evaluate_candidate(
                    candidate_files=candidate_files,
                    workspace_dir=attempt_dir / "proxy_workspace",
                    eval_dir=attempt_dir / "proxy_eval",
                    config=config,
                    inputs=config.proxy_inputs,
                )
                _write_json(attempt_dir / "proxy_eval.json", {
                    **({} if proxy_result.payload is None else proxy_result.payload),
                    "message": proxy_result.message,
                    "valid": proxy_result.valid,
                    "qps": proxy_result.qps,
                    "recall": proxy_result.recall,
                    "runtime_seconds": proxy_result.runtime_seconds,
                    "failure_type": proxy_result.failure_type,
                    "build_ok": proxy_result.build_ok,
                    "runtime_ok": proxy_result.runtime_ok,
                    "anti_cheat_passed": proxy_result.anti_cheat_passed,
                })
                row = _eval_row(
                    proxy_result,
                    attempt=attempt,
                    phase="proxy",
                    status="ok" if proxy_result.build_ok and proxy_result.runtime_ok else "crash",
                    candidate_sha=candidate_sha,
                    parent_sha=incumbent_sha,
                    change_summary=summary,
                )
                results_writer.writerow(row)
                round_rows.append(row)
                if proxy_result.build_ok and proxy_result.runtime_ok:
                    proxy_scored.append((attempt, summary, candidate_files, candidate_sha, proxy_result))
                else:
                    rejected_rows.append(row)

            proxy_scored.sort(key=lambda item: (_score_result(item[4]), item[4].qps, item[4].recall), reverse=True)
            strict_candidates = [item for item in proxy_scored if _is_better(item[4], incumbent_proxy)][: args.strict_top_k]

            best_strict_item: tuple[int, str, dict[str, str], str, BenchEvalResult] | None = None
            for attempt, summary, candidate_files, candidate_sha, _proxy_result in strict_candidates:
                attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
                strict_result = _evaluate_candidate(
                    candidate_files=candidate_files,
                    workspace_dir=attempt_dir / "strict_workspace",
                    eval_dir=attempt_dir / "strict_eval",
                    config=config,
                    inputs=config.strict_inputs,
                )
                _write_json(attempt_dir / "strict_eval.json", {
                    **({} if strict_result.payload is None else strict_result.payload),
                    "message": strict_result.message,
                    "valid": strict_result.valid,
                    "qps": strict_result.qps,
                    "recall": strict_result.recall,
                    "runtime_seconds": strict_result.runtime_seconds,
                    "failure_type": strict_result.failure_type,
                    "build_ok": strict_result.build_ok,
                    "runtime_ok": strict_result.runtime_ok,
                    "anti_cheat_passed": strict_result.anti_cheat_passed,
                })
                strict_row = _eval_row(
                    strict_result,
                    attempt=attempt,
                    phase="strict",
                    status="keep" if _is_better(strict_result, incumbent_strict) else "discard",
                    candidate_sha=candidate_sha,
                    parent_sha=incumbent_sha,
                    change_summary=summary,
                )
                results_writer.writerow(strict_row)
                round_rows.append(strict_row)
                if _is_better(strict_result, incumbent_strict):
                    if best_strict_item is None or _is_better(strict_result, best_strict_item[4]):
                        best_strict_item = (attempt, summary, candidate_files, candidate_sha, strict_result)
                else:
                    rejected_rows.append(strict_row)

            if best_strict_item is not None:
                attempt, summary, candidate_files, candidate_sha, strict_result = best_strict_item
                incumbent_files = candidate_files
                incumbent_sha = candidate_sha
                incumbent_strict = strict_result
                # Refresh proxy score for the new incumbent on the current proxy split.
                incumbent_proxy = next((item[4] for item in proxy_scored if item[0] == attempt), incumbent_proxy)
                accepted_row = _eval_row(
                    strict_result,
                    attempt=attempt,
                    phase="promotion",
                    status="keep",
                    candidate_sha=candidate_sha,
                    parent_sha="",
                    change_summary=summary,
                )
                accepted_rows.append(accepted_row)
                (args.run_dir / "incumbent").mkdir(exist_ok=True)
                for relpath, content in incumbent_files.items():
                    dest = args.run_dir / "incumbent" / relpath
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
            
            reviewer_prompt = _reviewer_prompt(
                program_text=program_text,
                memory_text=memory_path.read_text(encoding="utf-8"),
                incumbent_proxy=incumbent_proxy,
                incumbent_strict=incumbent_strict,
                round_rows=round_rows,
            )
            (round_dir / "reviewer_prompt.txt").write_text(reviewer_prompt, encoding="utf-8")
            reviewer_schema = round_dir / "reviewer.schema.json"
            _write_json(reviewer_schema, _json_schema_for_review())
            reviewer_exec = _run_codex_exec(
                executable=config.codex_executable,
                prompt=reviewer_prompt,
                cwd=REPO_ROOT,
                output_path=round_dir / "reviewer_output.json",
                timeout_seconds=config.codex_timeout_seconds,
                sandbox=config.codex_sandbox,
                model=config.codex_model,
                use_oss=config.codex_oss,
                local_provider=config.codex_local_provider,
                schema_path=reviewer_schema,
            )
            (round_dir / "reviewer.stdout.log").write_text(reviewer_exec.stdout, encoding="utf-8")
            (round_dir / "reviewer.stderr.log").write_text(reviewer_exec.stderr, encoding="utf-8")
            reviewer_payload: dict[str, Any]
            if reviewer_exec.returncode == 0:
                reviewer_payload = json.loads(reviewer_exec.last_message)
            else:
                reviewer_payload = {
                    "summary": f"reviewer failed: {_tail(reviewer_exec.stderr or reviewer_exec.stdout)}",
                    "lessons": ["reviewer failed; inspect stderr log"],
                    "promising_attempts": [],
                    "recommended_attempt": None,
                }
            reviewer_notes_path.write_text(_review_notes_text(reviewer_payload), encoding="utf-8")
            _update_memory(memory_path, round_idx, accepted_rows, rejected_rows)

            round_summary = {
                "round": round_idx,
                "run_dir": str(round_dir),
                "attempts_completed": sum(1 for row in round_rows if int(row["attempt"]) != 0 and row["phase"] == "proxy"),
                "crashed_attempts": sum(1 for row in round_rows if row["status"] == "crash"),
                "discarded_attempts": sum(1 for row in round_rows if row["status"] == "discard"),
                "kept_attempts": sum(1 for row in round_rows if row["status"] == "keep"),
                "incumbent_qps_proxy": incumbent_proxy.qps,
                "incumbent_qps_strict": incumbent_strict.qps,
                "incumbent_recall_proxy": incumbent_proxy.recall,
                "incumbent_recall_strict": incumbent_strict.recall,
                "incumbent_valid_strict": incumbent_strict.valid,
                "incumbent_sha256": incumbent_sha,
                "reviewer_summary": reviewer_payload.get("summary", ""),
                "elapsed_wall_clock_seconds": time.time() - round_started_at,
            }
            _write_json(round_dir / "summary.json", round_summary)
            round_summaries.append(round_summary)
            recent_rows.extend(round_rows)
            recent_rows = recent_rows[-30:]
        finally:
            _close_results_writer(results_writer)

    if args.apply_incumbent:
        for relpath, content in incumbent_files.items():
            dest = config.skeleton_dir / relpath
            dest.write_text(content, encoding="utf-8")

    final_summary = {
        "exit_code": 0,
        "run_dir": str(args.run_dir),
        "final_incumbent_sha256": incumbent_sha,
        "final_incumbent_qps_proxy": incumbent_proxy.qps,
        "final_incumbent_qps_strict": incumbent_strict.qps,
        "final_incumbent_recall_proxy": incumbent_proxy.recall,
        "final_incumbent_recall_strict": incumbent_strict.recall,
        "final_incumbent_valid_strict": incumbent_strict.valid,
        "rounds": round_summaries,
    }
    _write_json(args.run_dir / "summary.json", final_summary)
    return 0


def main() -> int:
    args = parse_args()
    return _run_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
