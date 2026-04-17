#!/usr/bin/env python3
"""Custom Meta-Harness runtime with additive helper-tool and retry support.

This runtime is used for revisions that declare helper tools or custom
zero-completion retry behavior. It keeps the official toolset intact and
appends revision-local helper tools on top when present.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import signal
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from meta_harness_common import AttemptProcessResult, RevisionConfig
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .meta_harness_common import AttemptProcessResult, RevisionConfig

MAX_RETRIES = 5
BUILD_TIMEOUT_SECS = 300
BENCHMARK_TIMEOUT_SECS = 600
SERVER_READY_TIMEOUT_SECS = 30
SERVER_POLL_INTERVAL_MS = 0.2
RECALL_THRESHOLD = 0.95
DEFAULT_BENCHMARK_BIN_NAME = "vector-db-benchmark"
DEFAULT_SYSTEM_PROMPT_PATH = Path("agent/system_prompt.txt")
DEFAULT_OPENROUTER_PROVIDER_IGNORE = ("ionstream",)
META_HARNESS_STATE_DIRNAME = ".meta_harness"
BEST_CANDIDATE_DIRNAME = "best_candidate"
BEST_CANDIDATE_MANIFEST = "best_candidate_manifest.json"
MAINLINE_DIRNAME = "mainline"
MAINLINE_MANIFEST = "mainline_manifest.json"
EXPERIMENTS_DIRNAME = "experiments"
CAMPAIGN_STATE_FILENAME = "campaign_state.json"
PROGRESS_STATE_FILENAME = "progress_state.json"
MAX_IDENTICAL_BENCHMARK_FAILURES = 3
ZERO_COMPLETION_RETRY_NUDGE = (
    "Your previous response contained zero completion tokens. "
    "Continue the run by either calling an appropriate tool or giving a concise next step. "
    "Do not repeat prior file reads unless needed."
)


JsonDict = dict[str, Any]
ToolHandler = Callable[["HelperToolContext", dict[str, Any]], Any]

MILESTONE_LADDER: tuple[tuple[float, str], ...] = (
    (0.0, "no_valid_result"),
    (1.0, "valid_exact_baseline"),
    (50.0, "qps_50"),
    (100.0, "qps_100"),
    (500.0, "qps_500"),
    (1000.0, "qps_1000"),
    (2000.0, "qps_2000"),
    (3000.0, "qps_3000"),
    (4000.0, "qps_4000"),
)


@dataclass(frozen=True)
class HelperToolSpec:
    name: str
    description: str
    parameters: JsonDict
    handler: ToolHandler


@dataclass
class RuntimeState:
    tool_calls_used: int
    tool_calls_total: int
    start_time: float
    server_running: bool
    last_benchmark: JsonDict | None
    best_benchmark: JsonDict | None
    call_log: list[JsonDict]

    @classmethod
    def new(cls, tool_calls_total: int) -> "RuntimeState":
        return cls(
            tool_calls_used=0,
            tool_calls_total=tool_calls_total,
            start_time=time.time(),
            server_running=False,
            last_benchmark=None,
            best_benchmark=None,
            call_log=[],
        )

    def elapsed_secs(self) -> float:
        return time.time() - self.start_time

    def tool_calls_remaining(self) -> int:
        return max(0, self.tool_calls_total - self.tool_calls_used)

    def get_status(self) -> JsonDict:
        return {
            "tool_calls_used": self.tool_calls_used,
            "tool_calls_remaining": self.tool_calls_remaining(),
            "tool_calls_total": self.tool_calls_total,
            "elapsed_time_secs": self.elapsed_secs(),
            "server_running": self.server_running,
            "last_benchmark": self.last_benchmark,
            "best_benchmark": self.best_benchmark,
        }

    def record_call(self, tool: str, input_json: JsonDict, output_json: JsonDict, duration_ms: int) -> None:
        self.tool_calls_used += 1
        self.call_log.append(
            {
                "index": self.tool_calls_used,
                "tool": tool,
                "input": input_json,
                "output": output_json,
                "duration_ms": duration_ms,
                "timestamp": _now_rfc3339(),
            }
        )

    def consider_benchmark(self, benchmark: JsonDict | None) -> None:
        if not benchmark:
            return
        self.last_benchmark = benchmark
        if benchmark.get("recall_passed") and benchmark.get("qps", 0.0) > 0.0:
            if self.best_benchmark is None or float(benchmark.get("qps", 0.0)) > float(self.best_benchmark.get("qps", 0.0)):
                self.best_benchmark = benchmark


def _chat_message(role: str, content: str) -> JsonDict:
    return {
        "role": role,
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


class AgentLogger:
    def __init__(self, work_dir: Path):
        self.path = work_dir / "agent_log.jsonl"
        self.handle = self.path.open("a", encoding="utf-8")
        self.iteration = 0

    def close(self) -> None:
        self.handle.close()

    def _log(self, event: JsonDict) -> None:
        self.handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.handle.flush()

    def log_session_start(self, model: str, work_dir: str, thinking_mode: str) -> None:
        self._log(
            {
                "event": "session_start",
                "model": model,
                "work_dir": work_dir,
                "thinking_mode": thinking_mode,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_llm_request(self, message_count: int) -> None:
        self.iteration += 1
        self._log(
            {
                "event": "llm_request",
                "iteration": self.iteration,
                "message_count": message_count,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_llm_response(
        self,
        *,
        has_tool_calls: bool,
        tool_call_count: int,
        content: str | None,
        thinking_content: str | None,
        duration_ms: int,
    ) -> None:
        self._log(
            {
                "event": "llm_response",
                "iteration": self.iteration,
                "has_tool_calls": has_tool_calls,
                "tool_call_count": tool_call_count,
                "content": content,
                "thinking_content": thinking_content,
                "duration_ms": duration_ms,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_llm_response_meta(
        self,
        *,
        response_id: str | None,
        response_model: str | None,
        finish_reason: str | None,
        native_finish_reason: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        reasoning_tokens: int | None,
    ) -> None:
        self._log(
            {
                "event": "llm_response_meta",
                "iteration": self.iteration,
                "response_id": response_id,
                "response_model": response_model,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_tool_call(self, index: int, tool: str, arguments: str, tool_call_id: str) -> None:
        self._log(
            {
                "event": "tool_call",
                "index": index,
                "tool": tool,
                "arguments": arguments,
                "tool_call_id": tool_call_id,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_tool_result(self, index: int, tool: str, tool_call_id: str, result: JsonDict, duration_ms: int) -> None:
        self._log(
            {
                "event": "tool_result",
                "index": index,
                "tool": tool,
                "tool_call_id": tool_call_id,
                "result": result,
                "duration_ms": duration_ms,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_session_end(self, tool_calls_used: int, tool_calls_total: int, elapsed_secs: float, reason: str) -> None:
        self._log(
            {
                "event": "session_end",
                "tool_calls_used": tool_calls_used,
                "tool_calls_total": tool_calls_total,
                "elapsed_secs": elapsed_secs,
                "reason": reason,
                "timestamp": _now_rfc3339(),
            }
        )

    def log_error(self, message: str) -> None:
        self._log(
            {
                "event": "error",
                "message": message,
                "timestamp": _now_rfc3339(),
            }
        )


class HelperToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, HelperToolSpec] = {}

    def add_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: JsonDict,
        handler: ToolHandler,
    ) -> None:
        if name in self._specs:
            raise ValueError(f"duplicate helper tool name: {name}")
        self._specs[name] = HelperToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def select(self, allowed_names: tuple[str, ...]) -> dict[str, HelperToolSpec]:
        if not allowed_names:
            return dict(self._specs)
        selected: dict[str, HelperToolSpec] = {}
        for name in allowed_names:
            if name not in self._specs:
                available = ", ".join(sorted(self._specs)) or "<none>"
                raise ValueError(
                    f"helper tool {name!r} declared in revision.toml but not registered by helper_tools_module. "
                    f"Available helper tools: {available}"
                )
            selected[name] = self._specs[name]
        return selected


class HelperToolContext:
    def __init__(self, runtime: "MetaHarnessRuntime") -> None:
        self.runtime = runtime
        self.work_dir = runtime.work_dir
        self.results_dir = runtime.results_dir
        self.bench_repo = runtime.bench_repo
        self.data_dir = runtime.data_dir
        self.cpu_cores = runtime.cpu_cores
        self.revision = runtime.revision

    def read_file(self, path: str) -> JsonDict:
        return self.runtime._tool_read_file(path)

    def write_file(self, path: str, content: str) -> JsonDict:
        return self.runtime._tool_write_file(path, content)

    def list_files(self, path: str) -> JsonDict:
        return self.runtime._tool_list_files(path)

    def run_benchmark(
        self,
        *,
        concurrency: int | None = None,
        warmup: int | None = None,
        max_queries: int | None = None,
    ) -> JsonDict:
        return self.runtime._tool_run_benchmark(concurrency, warmup, max_queries)

    def run_profiling(self, *, duration: int | None = None) -> JsonDict:
        return self.runtime._tool_run_profiling(duration)

    def run_correctness_test(self) -> JsonDict:
        return self.runtime._tool_run_correctness_test()

    def build_project(self) -> JsonDict:
        return self.runtime._tool_build_project()

    def get_status(self) -> JsonDict:
        return self.runtime._tool_get_status()

    def restore_best_candidate(self) -> JsonDict:
        return self.runtime._restore_best_candidate()

    def checkpoint_best_candidate(self, *, note: str = "") -> JsonDict:
        return self.runtime._checkpoint_best_candidate(note)

    def checkpoint_mainline(self, *, note: str = "") -> JsonDict:
        return self.runtime._checkpoint_mainline(note)

    def restore_mainline(self) -> JsonDict:
        return self.runtime._restore_mainline()

    def fork_experiment(self, *, label: str = "") -> JsonDict:
        return self.runtime._fork_experiment(label)

    def promote_experiment(self, *, label: str = "") -> JsonDict:
        return self.runtime._promote_experiment(label)

    def discard_experiment(self, *, label: str = "") -> JsonDict:
        return self.runtime._discard_experiment(label)

    def get_campaign_state(self) -> JsonDict:
        return self.runtime._campaign_state_payload()

    def get_call_log(self) -> list[JsonDict]:
        return list(self.runtime.state.call_log)

    def get_progress_state(self) -> JsonDict:
        return self.runtime._progress_state_payload()


class MetaHarnessRuntime:
    def __init__(
        self,
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
    ) -> None:
        self.revision = revision
        self.bench_repo = bench_repo.resolve()
        self.work_dir = work_dir.resolve()
        self.results_dir = results_dir.resolve()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.api_interval_ms = api_interval_ms
        self.cpu_cores = cpu_cores
        self.data_dir = data_dir.resolve()
        self.max_tool_calls = max_tool_calls
        self.stderr_log_path = self.results_dir.parent / "run_eval.stderr.log"
        self.stdout_log_path = self.results_dir.parent / "run_eval.stdout.log"
        self.stderr_handle = None
        self.stdout_handle = None
        self.state = RuntimeState.new(max_tool_calls)
        self.benchmark_failure_guard: JsonDict = {}
        self.last_llm_call_at: float | None = None
        self.benchmark_bin = self._ensure_benchmark_binary()
        self.helper_tools = self._load_helper_tools()
        self.messages = self._load_initial_messages()
        self.logger = AgentLogger(self.work_dir)
        self.helper_context = HelperToolContext(self)
        self.finish_summary = ""
        self.zero_completion_retry_limit = max(0, self.revision.zero_completion_retry_limit)

    def run(self) -> AttemptProcessResult:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.stderr_handle = self.stderr_log_path.open("w", encoding="utf-8")
        self.stdout_handle = self.stdout_log_path.open("w", encoding="utf-8")
        returncode = 0
        try:
            self._log_stderr("[meta-runtime] Starting custom Meta-Harness runtime")
            self.logger.log_session_start(self.model_name, str(self.work_dir), self.thinking_mode)
            reason = self._run_loop()
            self.logger.log_session_end(
                self.state.tool_calls_used,
                self.state.tool_calls_total,
                self.state.elapsed_secs(),
                reason,
            )
            self._save_eval_log()
            self._save_final_result(reason)
        except Exception as exc:  # noqa: BLE001
            returncode = 1
            self.logger.log_error(str(exc))
            self._log_stderr(f"[meta-runtime] ERROR: {exc}")
            self._save_eval_log()
            self._save_final_result("runtime_error")
        finally:
            self.logger.close()
            if self.stderr_handle is not None:
                self.stderr_handle.write(f"\n[wall_clock_seconds]={time.time() - started:.2f}\n")
                self.stderr_handle.close()
            if self.stdout_handle is not None:
                self.stdout_handle.close()
        return AttemptProcessResult(returncode=returncode, elapsed_secs=time.time() - started)

    def _run_loop(self) -> str:
        zero_completion_retries = 0
        while True:
            if self.state.tool_calls_used >= self.state.tool_calls_total:
                self._log_stderr(
                    f"[meta-runtime] Tool call limit reached ({self.state.tool_calls_used}/{self.state.tool_calls_total}). Auto-triggering finish."
                )
                result = self._tool_finish("Tool call limit reached - auto finish")
                self._persist_after_finish(result)
                return "tool_call_limit"

            self.logger.log_llm_request(len(self.messages))
            response, response_meta, duration_ms = self._call_llm()
            self.logger.log_llm_response_meta(
                response_id=response_meta.get("response_id"),
                response_model=response_meta.get("response_model"),
                finish_reason=response_meta.get("finish_reason"),
                native_finish_reason=response_meta.get("native_finish_reason"),
                prompt_tokens=_maybe_int(response_meta.get("prompt_tokens")),
                completion_tokens=_maybe_int(response_meta.get("completion_tokens")),
                total_tokens=_maybe_int(response_meta.get("total_tokens")),
                reasoning_tokens=_maybe_int(response_meta.get("reasoning_tokens")),
            )
            reasoning_content = response.get("reasoning_content") or response.get("reasoning")
            tool_calls = response.get("tool_calls")
            content = response.get("content")
            completion_tokens = _maybe_int(response_meta.get("completion_tokens"))
            is_zero_completion_empty = (
                completion_tokens == 0
                and not content
                and not reasoning_content
                and (tool_calls is None or len(tool_calls) == 0)
            )

            if tool_calls is not None:
                if not tool_calls:
                    self.logger.log_llm_response(
                        has_tool_calls=False,
                        tool_call_count=0,
                        content=content,
                        thinking_content=reasoning_content,
                        duration_ms=duration_ms,
                    )
                    if content:
                        zero_completion_retries = 0
                        self.messages.append(self._assistant_content_message(content, reasoning_content))
                        self._save_session_context()
                        continue
                    if (
                        is_zero_completion_empty
                        and zero_completion_retries < self.zero_completion_retry_limit
                    ):
                        zero_completion_retries += 1
                        self._log_stderr(
                            "[meta-runtime] Zero-completion response "
                            f"({zero_completion_retries}/{self.zero_completion_retry_limit}); retrying."
                        )
                        self.messages.append(_chat_message("user", ZERO_COMPLETION_RETRY_NUDGE))
                        self._save_session_context()
                        continue
                    return "empty_response"

                zero_completion_retries = 0
                self.logger.log_llm_response(
                    has_tool_calls=True,
                    tool_call_count=len(tool_calls),
                    content=content,
                    thinking_content=reasoning_content,
                    duration_ms=duration_ms,
                )
                self.messages.append(self._assistant_tool_calls_message(tool_calls, reasoning_content))

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"].get("arguments", "{}")
                    tool_call_id = tc["id"]
                    self.logger.log_tool_call(self.state.tool_calls_used + 1, tool_name, tool_args, tool_call_id)
                    started = time.time()
                    result = self._dispatch_tool_call(tool_name, tool_args)
                    duration_tool_ms = int((time.time() - started) * 1000)
                    input_json = {"tool": tool_name, "params": _json_loads(tool_args)}
                    self.state.record_call(tool_name, input_json, result, duration_tool_ms)
                    self.logger.log_tool_result(self.state.tool_calls_used, tool_name, tool_call_id, result, duration_tool_ms)
                    self.messages.append(self._tool_result_message(tool_call_id, result))
                    self._save_session_context()

                    if tool_name == "finish":
                        self._persist_after_finish(result)
                        return "finish_called"

                    if self.state.tool_calls_used >= self.state.tool_calls_total:
                        finish_result = self._tool_finish("Tool call limit reached - auto finish")
                        self._persist_after_finish(finish_result)
                        return "tool_call_limit"
                continue

            if content:
                zero_completion_retries = 0
                self.logger.log_llm_response(
                    has_tool_calls=False,
                    tool_call_count=0,
                    content=content,
                    thinking_content=reasoning_content,
                    duration_ms=duration_ms,
                )
                self.messages.append(self._assistant_content_message(content, reasoning_content))
                self._save_session_context()
                continue

            self.logger.log_llm_response(
                has_tool_calls=False,
                tool_call_count=0,
                content=None,
                thinking_content=reasoning_content,
                duration_ms=duration_ms,
            )
            if (
                is_zero_completion_empty
                and zero_completion_retries < self.zero_completion_retry_limit
            ):
                zero_completion_retries += 1
                self._log_stderr(
                    "[meta-runtime] Zero-completion response "
                    f"({zero_completion_retries}/{self.zero_completion_retry_limit}); retrying."
                )
                self.messages.append(_chat_message("user", ZERO_COMPLETION_RETRY_NUDGE))
                self._save_session_context()
                continue
            return "empty_response"

    def _persist_after_finish(self, result: JsonDict) -> None:
        final_benchmark = result.get("final_benchmark")
        if isinstance(final_benchmark, dict):
            self.state.consider_benchmark(final_benchmark)
            if final_benchmark.get("recall_passed") and float(final_benchmark.get("qps", 0.0) or 0.0) > 0.0:
                self._snapshot_best_candidate(final_benchmark, note="finish")
        self._save_session_context()
        self._write_progress_state()

    def _meta_harness_state_dir(self) -> Path:
        state_dir = self.work_dir / META_HARNESS_STATE_DIRNAME
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def _best_candidate_dir(self) -> Path:
        return self._meta_harness_state_dir() / BEST_CANDIDATE_DIRNAME

    def _best_candidate_manifest_path(self) -> Path:
        return self._meta_harness_state_dir() / BEST_CANDIDATE_MANIFEST

    def _mainline_dir(self) -> Path:
        return self._meta_harness_state_dir() / MAINLINE_DIRNAME

    def _mainline_manifest_path(self) -> Path:
        return self._meta_harness_state_dir() / MAINLINE_MANIFEST

    def _experiments_dir(self) -> Path:
        path = self._meta_harness_state_dir() / EXPERIMENTS_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _campaign_state_path(self) -> Path:
        return self._meta_harness_state_dir() / CAMPAIGN_STATE_FILENAME

    def _progress_state_path(self) -> Path:
        return self._meta_harness_state_dir() / PROGRESS_STATE_FILENAME

    def _candidate_source_relpaths(self, base_dir: Path | None = None) -> tuple[str, ...]:
        root = (base_dir or self.work_dir).resolve()
        relpaths: set[str] = set()
        cargo_path = root / "Cargo.toml"
        if cargo_path.exists() and _is_write_allowed("Cargo.toml") and not _is_readonly("Cargo.toml"):
            relpaths.add("Cargo.toml")
        src_root = root / "src"
        if src_root.exists():
            for path in src_root.rglob("*.rs"):
                rel = path.relative_to(root).as_posix()
                if _is_write_allowed(rel) and not _is_readonly(rel):
                    relpaths.add(rel)
        return tuple(sorted(relpaths))

    def _load_best_candidate_manifest(self) -> JsonDict | None:
        path = self._best_candidate_manifest_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return payload if isinstance(payload, dict) else None

    def _load_json_dict(self, path: Path) -> JsonDict | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return payload if isinstance(payload, dict) else None

    def _load_mainline_manifest(self) -> JsonDict | None:
        return self._load_json_dict(self._mainline_manifest_path())

    def _load_campaign_state(self) -> JsonDict:
        payload = self._load_json_dict(self._campaign_state_path()) or {}
        active_branch = str(payload.get("active_branch", "mainline") or "mainline")
        experiments = payload.get("experiments")
        if not isinstance(experiments, dict):
            experiments = {}
        return {
            "active_branch": active_branch,
            "experiments": experiments,
            "last_promotion_at": payload.get("last_promotion_at"),
        }

    def _write_campaign_state(self, payload: JsonDict) -> None:
        self._campaign_state_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _campaign_state_payload(self) -> JsonDict:
        payload = self._load_campaign_state()
        payload["mainline_manifest"] = self._load_mainline_manifest()
        payload["best_candidate_manifest"] = self._load_best_candidate_manifest()
        payload["benchmark_failure_guard"] = self.benchmark_failure_guard
        return payload

    def _experiment_dir(self, label: str) -> Path:
        return self._experiments_dir() / self._slugify_label(label)

    def _experiment_manifest_path(self, label: str) -> Path:
        return self._experiment_dir(label) / "manifest.json"

    def _slugify_label(self, label: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9._-]+", "-", (label or "").strip()).strip("-")
        return base or "experiment"

    def _copy_candidate_files(self, destination_root: Path) -> tuple[str, ...]:
        if destination_root.exists():
            shutil.rmtree(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        relpaths = self._candidate_source_relpaths()
        for rel in relpaths:
            src = self.work_dir / rel
            dst = destination_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return relpaths

    def _restore_candidate_files(self, snapshot_dir: Path, *, files: list[str]) -> int:
        snapshot_files = {
            rel for rel in files if _is_write_allowed(rel) and not _is_readonly(rel)
        }
        current_files = set(self._candidate_source_relpaths())
        for rel in sorted(current_files - snapshot_files):
            path = self.work_dir / rel
            if path.exists():
                path.unlink()
        restored = 0
        for rel in sorted(snapshot_files):
            src = snapshot_dir / rel
            if not src.exists():
                continue
            dst = self.work_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        return restored

    def _source_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for rel in self._candidate_source_relpaths():
            path = self.work_dir / rel
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\x00")
            if path.exists():
                hasher.update(path.read_bytes())
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def _normalize_failure_signature(self, message: str) -> str:
        normalized = message.strip()
        normalized = re.sub(r"127\.0\.0\.1:\d+", "127.0.0.1:PORT", normalized)
        normalized = re.sub(r"/tmp/[^ \n]+", "/tmp/PATH", normalized)
        normalized = re.sub(r"/home/[^ \n]+", "/home/PATH", normalized)
        return normalized[:600]

    def _record_benchmark_failure(
        self,
        *,
        concurrency: int,
        warmup: int,
        max_queries: int,
        message: str,
    ) -> None:
        signature = self._normalize_failure_signature(message)
        payload = {
            "args": {
                "concurrency": concurrency,
                "warmup": warmup,
                "max_queries": max_queries,
            },
            "source_fingerprint": self._source_fingerprint(),
            "signature": signature,
            "count": 1,
            "timestamp": _now_rfc3339(),
        }
        if (
            self.benchmark_failure_guard.get("args") == payload["args"]
            and self.benchmark_failure_guard.get("source_fingerprint") == payload["source_fingerprint"]
            and self.benchmark_failure_guard.get("signature") == payload["signature"]
        ):
            payload["count"] = int(self.benchmark_failure_guard.get("count", 0) or 0) + 1
        self.benchmark_failure_guard = payload

    def _clear_benchmark_failure_guard(self) -> None:
        self.benchmark_failure_guard = {}

    def _benchmark_failure_block_reason(
        self,
        *,
        concurrency: int,
        warmup: int,
        max_queries: int,
    ) -> str | None:
        if not self.benchmark_failure_guard:
            return None
        args_payload = {
            "concurrency": concurrency,
            "warmup": warmup,
            "max_queries": max_queries,
        }
        if (
            self.benchmark_failure_guard.get("args") == args_payload
            and self.benchmark_failure_guard.get("source_fingerprint") == self._source_fingerprint()
            and int(self.benchmark_failure_guard.get("count", 0) or 0) >= MAX_IDENTICAL_BENCHMARK_FAILURES
        ):
            return (
                "Blocked repeated benchmark attempt after "
                f"{int(self.benchmark_failure_guard['count'])} identical failures with no code or parameter change. "
                "Change the implementation or benchmark settings before retrying."
            )
        return None

    def _milestone_name(self, qps: float) -> str:
        chosen = MILESTONE_LADDER[0][1]
        for threshold, label in MILESTONE_LADDER:
            if qps >= threshold:
                chosen = label
        return chosen

    def _progress_state_payload(self) -> JsonDict:
        best = self.state.best_benchmark or {}
        best_qps = float(best.get("qps", 0.0) or 0.0)
        milestone_name = self._milestone_name(best_qps)
        next_milestone = None
        for threshold, label in MILESTONE_LADDER:
            if threshold > best_qps:
                next_milestone = {
                    "label": label,
                    "target_qps": threshold,
                    "gap_qps": max(0.0, threshold - best_qps),
                }
                break
        return {
            "tool_calls_used": self.state.tool_calls_used,
            "tool_calls_total": self.state.tool_calls_total,
            "elapsed_secs": self.state.elapsed_secs(),
            "last_benchmark": self.state.last_benchmark,
            "best_benchmark": self.state.best_benchmark,
            "milestone": {
                "label": milestone_name,
                "best_qps": best_qps,
            },
            "next_milestone": next_milestone,
            "best_candidate_manifest": self._load_best_candidate_manifest(),
            "campaign_state": self._campaign_state_payload(),
            "benchmark_failure_guard": self.benchmark_failure_guard,
        }

    def _write_progress_state(self) -> None:
        self._progress_state_path().write_text(
            json.dumps(self._progress_state_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _snapshot_best_candidate(self, benchmark: JsonDict | None, *, note: str) -> JsonDict:
        snapshot_dir = self._best_candidate_dir()
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        relpaths = self._candidate_source_relpaths()
        for rel in relpaths:
            src = self.work_dir / rel
            dst = snapshot_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        payload = {
            "timestamp": _now_rfc3339(),
            "note": note,
            "tool_calls_used": self.state.tool_calls_used,
            "best_benchmark": benchmark or self.state.best_benchmark,
            "milestone": self._milestone_name(float((benchmark or self.state.best_benchmark or {}).get("qps", 0.0) or 0.0)),
            "files": list(relpaths),
        }
        self._best_candidate_manifest_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._write_progress_state()
        return payload

    def _checkpoint_best_candidate(self, note: str) -> JsonDict:
        benchmark = self.state.best_benchmark or self.state.last_benchmark
        payload = self._snapshot_best_candidate(benchmark, note=note or "manual_checkpoint")
        return {
            "type": "CheckpointBestCandidate",
            "status": "ok",
            "manifest": payload,
        }

    def _checkpoint_mainline(self, note: str) -> JsonDict:
        relpaths = self._copy_candidate_files(self._mainline_dir())
        benchmark = self.state.best_benchmark or self.state.last_benchmark
        payload = {
            "timestamp": _now_rfc3339(),
            "note": note or "manual_mainline_checkpoint",
            "tool_calls_used": self.state.tool_calls_used,
            "best_benchmark": benchmark,
            "milestone": self._milestone_name(float((benchmark or {}).get("qps", 0.0) or 0.0)),
            "files": list(relpaths),
        }
        self._mainline_manifest_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        campaign = self._load_campaign_state()
        campaign["active_branch"] = "mainline"
        campaign["last_promotion_at"] = _now_rfc3339()
        self._write_campaign_state(campaign)
        self._write_progress_state()
        return {
            "type": "CheckpointMainline",
            "status": "ok",
            "manifest": payload,
        }

    def _restore_best_candidate(self) -> JsonDict:
        manifest = self._load_best_candidate_manifest()
        if not manifest:
            return _error_result("No best candidate snapshot is available to restore.")

        snapshot_dir = self._best_candidate_dir()
        if not snapshot_dir.exists():
            return _error_result("Best candidate snapshot directory is missing.")

        snapshot_files = {
            str(item)
            for item in manifest.get("files", [])
            if isinstance(item, str) and _is_write_allowed(item) and not _is_readonly(item)
        }
        current_files = set(self._candidate_source_relpaths())
        for rel in sorted(current_files - snapshot_files):
            path = self.work_dir / rel
            if path.exists():
                path.unlink()
        restored = 0
        for rel in sorted(snapshot_files):
            src = snapshot_dir / rel
            if not src.exists():
                continue
            dst = self.work_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        self._write_progress_state()
        return {
            "type": "RestoreBestCandidate",
            "status": "ok",
            "restored_files": restored,
            "manifest": manifest,
        }

    def _restore_mainline(self) -> JsonDict:
        manifest = self._load_mainline_manifest()
        if not manifest:
            return _error_result("No mainline snapshot is available to restore.")
        snapshot_dir = self._mainline_dir()
        if not snapshot_dir.exists():
            return _error_result("Mainline snapshot directory is missing.")
        restored = self._restore_candidate_files(snapshot_dir, files=list(manifest.get("files", [])))
        campaign = self._load_campaign_state()
        campaign["active_branch"] = "mainline"
        self._write_campaign_state(campaign)
        self._write_progress_state()
        return {
            "type": "RestoreMainline",
            "status": "ok",
            "restored_files": restored,
            "manifest": manifest,
        }

    def _fork_experiment(self, label: str) -> JsonDict:
        name = self._slugify_label(label)
        experiment_dir = self._experiment_dir(name)
        relpaths = self._copy_candidate_files(experiment_dir)
        manifest = {
            "label": name,
            "timestamp": _now_rfc3339(),
            "source_branch": self._load_campaign_state().get("active_branch", "mainline"),
            "tool_calls_used": self.state.tool_calls_used,
            "files": list(relpaths),
            "best_benchmark": self.state.best_benchmark or self.state.last_benchmark,
        }
        self._experiment_manifest_path(name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        campaign = self._load_campaign_state()
        experiments = dict(campaign.get("experiments", {}))
        experiments[name] = {
            "manifest_path": str(self._experiment_manifest_path(name)),
            "created_at": manifest["timestamp"],
        }
        campaign["experiments"] = experiments
        campaign["active_branch"] = name
        self._write_campaign_state(campaign)
        self._write_progress_state()
        return {
            "type": "ForkExperiment",
            "status": "ok",
            "manifest": manifest,
        }

    def _promote_experiment(self, label: str) -> JsonDict:
        campaign = self._load_campaign_state()
        name = self._slugify_label(label or str(campaign.get("active_branch", "")))
        manifest = self._load_json_dict(self._experiment_manifest_path(name))
        if not manifest:
            return _error_result(f"Experiment {name!r} does not exist.")
        experiment_dir = self._experiment_dir(name)
        relpaths = tuple(
            rel
            for rel in manifest.get("files", [])
            if isinstance(rel, str) and _is_write_allowed(rel) and not _is_readonly(rel)
        )
        mainline_dir = self._mainline_dir()
        if mainline_dir.exists():
            shutil.rmtree(mainline_dir)
        mainline_dir.mkdir(parents=True, exist_ok=True)
        for rel in relpaths:
            src = experiment_dir / rel
            if not src.exists():
                continue
            dst = mainline_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        mainline_manifest = {
            "timestamp": _now_rfc3339(),
            "note": f"promoted_from_{name}",
            "tool_calls_used": self.state.tool_calls_used,
            "best_benchmark": self.state.best_benchmark or self.state.last_benchmark,
            "milestone": self._milestone_name(float(((self.state.best_benchmark or self.state.last_benchmark or {}).get('qps', 0.0)) or 0.0)),
            "files": list(relpaths),
            "source_experiment": name,
        }
        self._mainline_manifest_path().write_text(
            json.dumps(mainline_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        campaign["active_branch"] = "mainline"
        campaign["last_promotion_at"] = _now_rfc3339()
        self._write_campaign_state(campaign)
        self._write_progress_state()
        return {
            "type": "PromoteExperiment",
            "status": "ok",
            "mainline_manifest": mainline_manifest,
        }

    def _discard_experiment(self, label: str) -> JsonDict:
        campaign = self._load_campaign_state()
        name = self._slugify_label(label or str(campaign.get("active_branch", "")))
        experiment_dir = self._experiment_dir(name)
        manifest = self._load_json_dict(self._experiment_manifest_path(name))
        if manifest is None and not experiment_dir.exists():
            return _error_result(f"Experiment {name!r} does not exist.")
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        experiments = dict(campaign.get("experiments", {}))
        experiments.pop(name, None)
        campaign["experiments"] = experiments
        if campaign.get("active_branch") == name:
            campaign["active_branch"] = "mainline"
            self._write_campaign_state(campaign)
            restore = self._restore_mainline()
        else:
            self._write_campaign_state(campaign)
            restore = None
        self._write_progress_state()
        return {
            "type": "DiscardExperiment",
            "status": "ok",
            "discarded": name,
            "restore": restore,
        }

    def _call_llm(self) -> tuple[JsonDict, JsonDict, int]:
        if self.last_llm_call_at is not None and self.api_interval_ms > 0:
            elapsed_ms = int((time.time() - self.last_llm_call_at) * 1000)
            if elapsed_ms < self.api_interval_ms:
                time.sleep((self.api_interval_ms - elapsed_ms) / 1000.0)

        request_body = {
            "model": self.model_id,
            "messages": self.messages,
            "tools": self._tool_definitions(),
            "tool_choice": "auto",
        }
        request_body.update(_build_thinking_extra(self.model_id, self.thinking_mode, self.reasoning_effort))
        request_body.update(_build_provider_extra(self.base_url))
        body = json.dumps(request_body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error = "LLM API call failed"
        started = time.time()

        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                backoff_ms = max((2**attempt) * 1000, self.api_interval_ms)
                time.sleep(backoff_ms / 1000.0)
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                choices = payload.get("choices") or []
                if not choices:
                    last_error = "LLM API returned empty choices"
                    continue
                self.last_llm_call_at = time.time()
                choice = choices[0]
                usage = payload.get("usage") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                meta = {
                    "response_id": payload.get("id"),
                    "response_model": payload.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": completion_details.get("reasoning_tokens"),
                }
                return choice["message"], meta, int((time.time() - started) * 1000)
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                last_error = f"LLM API error (status {exc.code}): {body_text}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"LLM API request failed: {exc}"
        raise RuntimeError(last_error)

    def _tool_definitions(self) -> list[JsonDict]:
        defs = list(_official_tool_definitions())
        for spec in self.helper_tools.values():
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
            )
        return defs

    def _dispatch_tool_call(self, tool_name: str, tool_args_json: str) -> JsonDict:
        args = _json_loads(tool_args_json)
        if tool_name == "read_file":
            return self._tool_read_file(str(args.get("path", "")))
        if tool_name == "write_file":
            return self._tool_write_file(str(args.get("path", "")), str(args.get("content", "")))
        if tool_name == "list_files":
            return self._tool_list_files(str(args.get("path", "")))
        if tool_name == "run_benchmark":
            return self._tool_run_benchmark(
                _maybe_int(args.get("concurrency")),
                _maybe_int(args.get("warmup")),
                _maybe_int(args.get("max_queries")),
            )
        if tool_name == "run_profiling":
            return self._tool_run_profiling(_maybe_int(args.get("duration")))
        if tool_name == "run_correctness_test":
            return self._tool_run_correctness_test()
        if tool_name == "build_project":
            return self._tool_build_project()
        if tool_name == "get_status":
            return self._tool_get_status()
        if tool_name == "finish":
            return self._tool_finish(str(args.get("summary", "")))
        if tool_name in self.helper_tools:
            spec = self.helper_tools[tool_name]
            try:
                payload = _json_safe(spec.handler(self.helper_context, args))
                return {
                    "type": "HelperToolResult",
                    "tool": tool_name,
                    "payload": payload,
                }
            except Exception as exc:  # noqa: BLE001
                return _error_result(f"Helper tool {tool_name!r} failed: {exc}")
        return _error_result(f"Unknown tool: {tool_name}")

    def _tool_read_file(self, path: str) -> JsonDict:
        if not _is_read_allowed(path):
            return _error_result(
                f"Access denied: '{path}' is outside the allowed scope. You can only read files under src/, Cargo.toml, benchmarks/, and profiling/."
            )
        full_path = self.work_dir / path
        try:
            return {"type": "ReadFile", "content": full_path.read_text(encoding="utf-8")}
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"Failed to read file '{path}': {exc}")

    def _tool_write_file(self, path: str, content: str) -> JsonDict:
        if not _is_write_allowed(path):
            return _error_result(
                f"Access denied: '{path}' is outside the allowed scope. You can write files under src/ and Cargo.toml."
            )
        if _is_readonly(path):
            return _error_result(f"Permission denied: '{path}' is read-only")
        full_path = self.work_dir / path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return {"type": "WriteFile", "status": "ok", "bytes_written": len(content)}
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"Failed to write file '{path}': {exc}")

    def _tool_list_files(self, path: str) -> JsonDict:
        if not _is_read_allowed(path):
            return _error_result(
                f"Access denied: '{path}' is outside the allowed scope. You can list files under src/, benchmarks/, and profiling/."
            )
        full_path = self.work_dir / path
        try:
            files = sorted(entry.name for entry in full_path.iterdir())
            return {"type": "ListFiles", "files": files}
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"Failed to list directory '{path}': {exc}")

    def _tool_build_project(self) -> JsonDict:
        result = _build_project(self.work_dir, profiling=False)
        if result is None:
            return {"type": "BuildProject", "success": True, "message": "Build succeeded."}
        return {"type": "BuildProject", "success": False, "message": f"Build failed: {result}"}

    def _tool_run_benchmark(self, concurrency: int | None, warmup: int | None, max_queries: int | None) -> JsonDict:
        effective_concurrency = concurrency if concurrency is not None else 4
        effective_warmup = warmup if warmup is not None else 100
        effective_max_queries = max_queries if max_queries is not None else 1000
        blocked_reason = self._benchmark_failure_block_reason(
            concurrency=effective_concurrency,
            warmup=effective_warmup,
            max_queries=effective_max_queries,
        )
        if blocked_reason is not None:
            result = _error_result(blocked_reason)
            self._write_progress_state()
            return result
        previous_best_qps = float((self.state.best_benchmark or {}).get("qps", 0.0) or 0.0)
        result = _run_benchmark_like(
            work_dir=self.work_dir,
            benchmark_bin=self.benchmark_bin,
            data_dir=self.data_dir,
            cpu_cores=self.cpu_cores,
            concurrency=effective_concurrency,
            warmup=effective_warmup,
            max_queries=effective_max_queries,
            save_history=True,
        )
        if result.get("type") == "RunBenchmark":
            self._clear_benchmark_failure_guard()
            self.state.consider_benchmark(result)
            current_best_qps = float((self.state.best_benchmark or {}).get("qps", 0.0) or 0.0)
            if result.get("recall_passed") and current_best_qps > previous_best_qps:
                self._snapshot_best_candidate(result, note="run_benchmark")
            else:
                self._write_progress_state()
        else:
            self._record_benchmark_failure(
                concurrency=effective_concurrency,
                warmup=effective_warmup,
                max_queries=effective_max_queries,
                message=str(result.get("message", "")),
            )
            self._write_progress_state()
        return result

    def _tool_run_correctness_test(self) -> JsonDict:
        return _run_correctness_test_like(
            work_dir=self.work_dir,
            benchmark_bin=self.benchmark_bin,
            data_dir=self.data_dir,
            cpu_cores=self.cpu_cores,
        )

    def _tool_run_profiling(self, duration: int | None) -> JsonDict:
        return _run_profiling_like(
            work_dir=self.work_dir,
            benchmark_bin=self.benchmark_bin,
            data_dir=self.data_dir,
            cpu_cores=self.cpu_cores,
            duration=duration,
        )

    def _tool_get_status(self) -> JsonDict:
        return {"type": "GetStatus", **self.state.get_status()}

    def _tool_finish(self, summary: str) -> JsonDict:
        self.finish_summary = summary
        bench_result = self._tool_run_benchmark(None, None, 0)
        if bench_result.get("type") == "RunBenchmark":
            self.state.consider_benchmark(bench_result)
            return {
                "type": "Finish",
                "status": (
                    f"Evaluation complete. Summary: {summary}. Final QPS: {bench_result['qps']:.2f}, "
                    f"Recall: {bench_result['recall']:.4f}"
                ),
                "final_benchmark": bench_result,
            }
        if self.state.best_benchmark is not None:
            best = self.state.best_benchmark
            self.state.last_benchmark = best
            return {
                "type": "Finish",
                "status": (
                    f"Evaluation complete with final benchmark error: {bench_result.get('message', 'unknown error')}. "
                    f"Using best recorded QPS: {best['qps']:.2f}, Recall: {best['recall']:.4f}. Summary: {summary}"
                ),
                "final_benchmark": best,
            }
        return {
            "type": "Finish",
            "status": f"Evaluation complete with benchmark error: {bench_result.get('message', 'unknown error')}. Summary: {summary}",
            "final_benchmark": None,
        }

    def _load_initial_messages(self) -> list[JsonDict]:
        session_context_path = self.work_dir / "session_context.json"
        if session_context_path.exists():
            payload = json.loads(session_context_path.read_text(encoding="utf-8"))
            self.state = RuntimeState(
                tool_calls_used=int(payload.get("tool_calls_used", 0)),
                tool_calls_total=int(payload.get("tool_calls_total", self.max_tool_calls)),
                start_time=time.time(),
                server_running=False,
                last_benchmark=payload.get("last_benchmark"),
                best_benchmark=payload.get("best_benchmark"),
                call_log=list(payload.get("call_log", [])),
            )
            self.benchmark_failure_guard = payload.get("benchmark_failure_guard", {}) or {}
            return list(payload.get("messages", []))
        system_prompt = (self.bench_repo / DEFAULT_SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
        return [
            _system_message(system_prompt),
            _user_message("Begin. Read the project files and start implementing."),
        ]

    def _save_session_context(self) -> None:
        payload = {
            "tool_calls_used": self.state.tool_calls_used,
            "tool_calls_total": self.state.tool_calls_total,
            "messages": self.messages,
            "last_benchmark": self.state.last_benchmark,
            "best_benchmark": self.state.best_benchmark,
            "call_log": self.state.call_log,
            "benchmark_failure_guard": self.benchmark_failure_guard,
        }
        (self.work_dir / "session_context.json").write_text(json.dumps(payload), encoding="utf-8")
        self._write_progress_state()

    def _save_eval_log(self) -> None:
        payload = {
            "tool_calls_used": self.state.tool_calls_used,
            "tool_calls_total": self.state.tool_calls_total,
            "call_log": self.state.call_log,
            "last_benchmark": self.state.last_benchmark,
            "best_benchmark": self.state.best_benchmark,
            "progress_state": self._progress_state_payload(),
        }
        (self.work_dir / "eval_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _save_final_result(self, reason: str) -> None:
        chosen = None
        chosen_source = "none"
        if self.state.best_benchmark and self.state.best_benchmark.get("recall_passed") and self.state.best_benchmark.get("qps", 0.0) > 0.0:
            chosen = self.state.best_benchmark
            chosen_source = "best_benchmark"
        elif self.state.last_benchmark and self.state.last_benchmark.get("recall_passed") and self.state.last_benchmark.get("qps", 0.0) > 0.0:
            chosen = self.state.last_benchmark
            chosen_source = "last_benchmark"
        result = {
            "model_name": self.model_name,
            "tool_calls_used": self.state.tool_calls_used,
            "total_time_secs": self.state.elapsed_secs(),
            "optimization_summary": self.finish_summary,
            "reason": reason,
            "qps": float(chosen.get("qps", 0.0)) if chosen else 0.0,
            "recall": float(chosen.get("recall", 0.0)) if chosen else 0.0,
            "recall_passed": bool(chosen.get("recall_passed", False)) if chosen else False,
            "result_source": chosen_source,
            "milestone": self._progress_state_payload().get("milestone"),
            "best_candidate_manifest": self._load_best_candidate_manifest(),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.results_dir / f"{self.model_name}.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    def _ensure_benchmark_binary(self) -> Path:
        benchmark_dir = self.bench_repo / "benchmark"
        benchmark_bin = benchmark_dir / "target" / "release" / DEFAULT_BENCHMARK_BIN_NAME
        if benchmark_bin.exists():
            return benchmark_bin.resolve()
        proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECS,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to build benchmark crate: {(proc.stderr or proc.stdout)[-2000:]}"
            )
        if not benchmark_bin.exists():
            raise FileNotFoundError(f"benchmark binary not found after build: {benchmark_bin}")
        return benchmark_bin.resolve()

    def _load_helper_tools(self) -> dict[str, HelperToolSpec]:
        if self.revision.helper_tools_module is None:
            return {}
        spec = importlib.util.spec_from_file_location(
            f"meta_harness_helper_tools_{self.revision.revision_id}",
            self.revision.helper_tools_module,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load helper tool module: {self.revision.helper_tools_module}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register = getattr(module, "register_tools", None)
        if not callable(register):
            raise RuntimeError(
                f"helper_tools_module {self.revision.helper_tools_module} must define register_tools(registry)"
            )
        registry = HelperToolRegistry()
        register(registry)
        return registry.select(self.revision.added_helper_tools)

    def _assistant_tool_calls_message(self, tool_calls: list[JsonDict], reasoning_content: str | None) -> JsonDict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "tool_call_id": None,
            "reasoning_content": reasoning_content,
        }

    def _assistant_content_message(self, content: str, reasoning_content: str | None) -> JsonDict:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": reasoning_content,
        }

    def _tool_result_message(self, tool_call_id: str, result: JsonDict) -> JsonDict:
        return {
            "role": "tool",
            "content": json.dumps(result),
            "tool_calls": None,
            "tool_call_id": tool_call_id,
            "reasoning_content": None,
        }

    def _log_stderr(self, message: str) -> None:
        if self.stderr_handle is not None:
            self.stderr_handle.write(message.rstrip() + "\n")
            self.stderr_handle.flush()


def run_custom_attempt(
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
    runtime = MetaHarnessRuntime(
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
    return runtime.run()


def _official_tool_definitions() -> tuple[JsonDict, ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Accessible paths: src/*, Cargo.toml, benchmarks/*, profiling/*.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path to read"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Writable paths: src/* (except read-only src/main.rs and src/api.rs) and Cargo.toml.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write to"},
                        "content": {"type": "string", "description": "Content to write into the file"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory. Accessible directories: src/, benchmarks/, profiling/.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Directory path to list"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_benchmark",
                "description": "Run the full benchmark: builds project, starts server, loads 1M vectors, runs queries, reports QPS/latency/recall, stops server. Results are saved to benchmarks/ with round numbers. Includes comparison with previous run if available. Default 1000 queries; use max_queries=0 for full 10K.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concurrency": {"type": "integer", "description": "Number of concurrent query threads (default 4)"},
                        "warmup": {"type": "integer", "description": "Number of warmup queries (default 100)"},
                        "max_queries": {"type": "integer", "description": "Maximum number of queries to run (default 1000, 0 = all 10000)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_profiling",
                "description": "Run performance profiling with real benchmark workload: builds project, starts server, runs perf record while benchmark client sends real queries, stops server. Returns top 10 hottest functions with CPU percentage and flamegraph SVG path. Results saved to profiling/ with round numbers. You can list_files('profiling') to see historical flamegraphs.",
                "parameters": {
                    "type": "object",
                    "properties": {"duration": {"type": "integer", "description": "Profiling duration in seconds (default 30)"}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_correctness_test",
                "description": "Run correctness validation: builds project, starts server, runs test, stops server. Returns recall and pass/fail status.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_project",
                "description": "Build the project with cargo build --release. Returns success/failure with compiler error messages. Use this to quickly check if your code compiles before running benchmark or profiling.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Get current agent session status: tool call counts, elapsed time, server status, and last benchmark result.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal that optimization is complete. Triggers the final benchmark run and records the score.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "description": "Summary of optimizations performed"}},
                    "required": ["summary"],
                },
            },
        },
    )


def _build_thinking_extra(model_id: str, thinking_mode: str, reasoning_effort: str) -> JsonDict:
    if model_id.strip().lower() == "qwen/qwen3-coder-next":
        return {}
    mode = thinking_mode.strip().lower().replace("_", "-")
    if mode in {"false", "off", "none", ""}:
        return {}
    if mode in {"true", "openai", "thinking", "openai-thinking"}:
        return {"thinking": {"type": "enabled"}}
    if mode == "kimi":
        return {"enable_thinking": True}
    if mode == "gemini":
        return {"reasoning": {"enabled": True}}
    if mode in {"reasoning", "openai-reasoning", "reasoning-effort", "openrouter-openai"}:
        return {"reasoning": {"effort": reasoning_effort}}
    if mode == "doubao":
        return {"reasoning_effort": reasoning_effort}
    return {}


def _build_provider_extra(base_url: str) -> JsonDict:
    if "openrouter.ai" not in base_url.lower():
        return {}
    return {
        "provider": {
            "ignore": list(DEFAULT_OPENROUTER_PROVIDER_IGNORE),
        }
    }


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _system_message(content: str) -> JsonDict:
    return {
        "role": "system",
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


def _user_message(content: str) -> JsonDict:
    return {
        "role": "user",
        "content": content,
        "tool_calls": None,
        "tool_call_id": None,
        "reasoning_content": None,
    }


def _json_loads(raw: str | None) -> JsonDict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error_result(message: str) -> JsonDict:
    return {"type": "Error", "message": message}


def _normalize_relpath(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return "__invalid__"
    return "/".join(parts)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return {"repr": repr(value)}


def _is_under(prefixes: tuple[str, ...], normalized: str) -> bool:
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in prefixes)


def _is_read_allowed(path: str) -> bool:
    normalized = _normalize_relpath(path)
    if normalized == "Cargo.toml":
        return True
    return _is_under(("src/", "benchmarks/", "profiling/"), normalized)


def _is_write_allowed(path: str) -> bool:
    normalized = _normalize_relpath(path)
    return normalized == "Cargo.toml" or _is_under(("src/",), normalized)


def _is_readonly(path: str) -> bool:
    normalized = _normalize_relpath(path)
    readonly_paths = ("src/main.rs", "src/api.rs", "benchmark/", "scripts/load_data.py")
    for ro in readonly_paths:
        if ro.endswith("/"):
            if normalized == ro.rstrip("/") or normalized.startswith(ro):
                return True
        elif normalized == ro:
            return True
    return False


def _build_project(work_dir: Path, *, profiling: bool) -> str | None:
    env = os.environ.copy()
    if profiling:
        env["CARGO_PROFILE_RELEASE_DEBUG"] = "2"
        env["CARGO_PROFILE_RELEASE_LTO"] = "false"
        env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "16"
    try:
        proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"cargo build --release timed out after {BUILD_TIMEOUT_SECS} seconds"
    if proc.returncode == 0:
        return None
    try:
        subprocess.run(
            ["cargo", "clean", "--release"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        pass
    try:
        proc_retry = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"cargo build --release timed out after {BUILD_TIMEOUT_SECS} seconds"
    if proc_retry.returncode == 0:
        return None
    return f"cargo build --release failed (exit code {proc_retry.returncode}):\n{proc_retry.stderr or proc_retry.stdout}"


def _detect_binary_name(work_dir: Path) -> str:
    cargo_toml = work_dir / "Cargo.toml"
    if not cargo_toml.exists():
        raise FileNotFoundError(f"Cargo.toml not found in '{work_dir}'. Is this a valid Rust project?")
    in_package = False
    for line in cargo_toml.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if trimmed.startswith("["):
            in_package = trimmed == "[package]"
            continue
        if in_package and trimmed.startswith("name") and "=" in trimmed:
            value = trimmed.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(f"Could not find package name in '{cargo_toml}'")


def _allocate_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(work_dir: Path, *, port: int, cpu_cores: str) -> subprocess.Popen[str]:
    binary_name = _detect_binary_name(work_dir)
    binary = (work_dir / "target" / "release" / binary_name).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Server binary not found at '{binary}'. Build the project first.")
    argv: list[str] = []
    if cpu_cores:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise FileNotFoundError("taskset not found but cpu_cores was provided")
        argv.extend([taskset, "-c", cpu_cores])
    argv.append(str(binary))
    return subprocess.Popen(
        argv,
        cwd=work_dir,
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_server_ready(port: int, timeout_secs: int) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(SERVER_POLL_INTERVAL_MS)
    raise TimeoutError(f"Server on port {port} not ready after {timeout_secs} seconds")


def _kill_server(child: subprocess.Popen[str] | None) -> None:
    if child is None:
        return
    try:
        child.kill()
    except OSError:
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _find_base_vectors(data_dir: Path, work_dir: Path) -> Path:
    single = data_dir / "base_vectors.json"
    if single.exists():
        return single
    shards: list[Path] = []
    index = 0
    while True:
        shard = data_dir / f"base_vectors_{index}.json"
        if not shard.exists():
            break
        shards.append(shard)
        index += 1
    if not shards:
        raise FileNotFoundError(f"No base_vectors.json or base_vectors_N.json files found in {data_dir}")
    merged_path = work_dir / "base_vectors_merged.json"
    if merged_path.exists():
        return merged_path
    all_vectors: list[Any] = []
    for shard in shards:
        all_vectors.extend(json.loads(shard.read_text(encoding="utf-8")))
    merged_path.write_text(json.dumps(all_vectors), encoding="utf-8")
    return merged_path


def _load_previous_benchmark(work_dir: Path) -> JsonDict | None:
    bench_dir = work_dir / "benchmarks"
    if not bench_dir.exists():
        return None
    files = sorted(bench_dir.glob("benchmark_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _save_benchmark_result(work_dir: Path, result: JsonDict) -> None:
    bench_dir = work_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    round_num = len(list(bench_dir.glob("benchmark_*.json"))) + 1
    path = bench_dir / f"benchmark_{round_num:03d}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _build_comparison(prev: JsonDict, curr: JsonDict) -> JsonDict:
    prev_qps = float(prev.get("qps", 0.0) or 0.0)
    prev_recall = float(prev.get("recall", 0.0) or 0.0)
    curr_qps = float(curr.get("qps", 0.0) or 0.0)
    curr_recall = float(curr.get("recall", 0.0) or 0.0)
    qps_change = ((curr_qps - prev_qps) / prev_qps * 100.0) if prev_qps > 0 else 0.0
    recall_change = ((curr_recall - prev_recall) / prev_recall * 100.0) if prev_recall > 0 else 0.0
    return {
        "previous_qps": prev_qps,
        "qps_change_pct": round(qps_change, 2),
        "previous_recall": prev_recall,
        "recall_change_pct": round(recall_change, 2),
    }


def _run_benchmark_like(
    *,
    work_dir: Path,
    benchmark_bin: Path,
    data_dir: Path,
    cpu_cores: str,
    concurrency: int,
    warmup: int,
    max_queries: int,
    save_history: bool,
) -> JsonDict:
    build_error = _build_project(work_dir, profiling=False)
    if build_error is not None:
        return _error_result(f"Build failed: {build_error}")

    port = _allocate_port()
    child = None
    try:
        child = _start_server(work_dir, port=port, cpu_cores=cpu_cores)
        _wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS)
        base_vectors = _find_base_vectors(data_dir, work_dir)
        query_vectors = data_dir / "query_vectors.json"
        ground_truth = data_dir / "ground_truth.json"
        for path, label in ((query_vectors, "query vectors"), (ground_truth, "ground truth")):
            if not path.exists():
                return _error_result(f"Data file not found: {path} ({label})")
        proc = subprocess.run(
            [
                str(benchmark_bin),
                "--server-url",
                f"http://127.0.0.1:{port}",
                "--concurrency",
                str(concurrency),
                "--warmup",
                str(warmup),
                "--max-queries",
                str(max_queries),
                "--base-vectors",
                str(base_vectors),
                "--query-vectors",
                str(query_vectors),
                "--ground-truth",
                str(ground_truth),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BENCHMARK_TIMEOUT_SECS,
        )
        if proc.returncode != 0:
            return _error_result(
                f"Benchmark exited with code {proc.returncode}: {(proc.stderr or proc.stdout)[:2000]}"
            )
        output = json.loads(proc.stdout)
        bench = dict(output["benchmark"])
        prev = _load_previous_benchmark(work_dir)
        if prev is not None:
            bench["comparison"] = _build_comparison(prev, bench)
        if save_history:
            _save_benchmark_result(work_dir, bench)
        return {"type": "RunBenchmark", **bench}
    except subprocess.TimeoutExpired:
        return _error_result(f"Benchmark timed out after {BENCHMARK_TIMEOUT_SECS} seconds")
    except Exception as exc:  # noqa: BLE001
        return _error_result(str(exc))
    finally:
        _kill_server(child)


def _run_correctness_test_like(*, work_dir: Path, benchmark_bin: Path, data_dir: Path, cpu_cores: str) -> JsonDict:
    build_error = _build_project(work_dir, profiling=False)
    if build_error is not None:
        return _error_result(f"Build failed: {build_error}")
    port = _allocate_port()
    child = None
    try:
        child = _start_server(work_dir, port=port, cpu_cores=cpu_cores)
        _wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS)
        base_vectors = _find_base_vectors(data_dir, work_dir)
        query_vectors = data_dir / "query_vectors.json"
        ground_truth = data_dir / "ground_truth.json"
        proc = subprocess.run(
            [
                str(benchmark_bin),
                "--server-url",
                f"http://127.0.0.1:{port}",
                "--concurrency",
                "1",
                "--warmup",
                "0",
                "--max-queries",
                "100",
                "--base-vectors",
                str(base_vectors),
                "--query-vectors",
                str(query_vectors),
                "--ground-truth",
                str(ground_truth),
                "--recall-threshold",
                str(RECALL_THRESHOLD),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BENCHMARK_TIMEOUT_SECS,
        )
        if proc.returncode != 0:
            return _error_result(
                f"Correctness test exited with code {proc.returncode}: {(proc.stderr or proc.stdout)[:2000]}"
            )
        output = json.loads(proc.stdout)
        bench = output["benchmark"]
        passed = float(bench.get("recall", 0.0)) >= RECALL_THRESHOLD
        message = (
            f"Correctness test PASSED: recall {bench['recall']:.4f} >= threshold {RECALL_THRESHOLD:.4f}"
            if passed
            else f"Correctness test FAILED: recall {bench['recall']:.4f} < threshold {RECALL_THRESHOLD:.4f}"
        )
        return {
            "type": "RunCorrectnessTest",
            "passed": passed,
            "total_queries": int(bench.get("total_queries", 0) or 0),
            "recall": float(bench.get("recall", 0.0) or 0.0),
            "recall_threshold": RECALL_THRESHOLD,
            "failed_queries": [],
            "message": message,
        }
    except subprocess.TimeoutExpired:
        return _error_result(f"Correctness test timed out after {BENCHMARK_TIMEOUT_SECS} seconds")
    except Exception as exc:  # noqa: BLE001
        return _error_result(str(exc))
    finally:
        _kill_server(child)


def _run_profiling_like(
    *,
    work_dir: Path,
    benchmark_bin: Path,
    data_dir: Path,
    cpu_cores: str,
    duration: int | None,
) -> JsonDict:
    build_error = _build_project(work_dir, profiling=True)
    if build_error is not None:
        return _error_result(f"Build failed: {build_error}")
    port = _allocate_port()
    child = None
    perf_child = None
    perf_data = work_dir / "perf.data"
    flamegraph_svg = work_dir / "flamegraph.svg"
    try:
        child = _start_server(work_dir, port=port, cpu_cores=cpu_cores)
        _wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS)
        base_vectors = _find_base_vectors(data_dir, work_dir)
        query_vectors = data_dir / "query_vectors.json"
        ground_truth = data_dir / "ground_truth.json"
        perf_child = subprocess.Popen(
            [
                "perf",
                "record",
                "-F",
                "99",
                "-p",
                str(child.pid),
                "-g",
                "-o",
                str(perf_data),
            ],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                str(benchmark_bin),
                "--server-url",
                f"http://127.0.0.1:{port}",
                "--concurrency",
                "4",
                "--warmup",
                "100",
                "--max-queries",
                str(1000 if duration is None else max(100, duration * 50)),
                "--base-vectors",
                str(base_vectors),
                "--query-vectors",
                str(query_vectors),
                "--ground-truth",
                str(ground_truth),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=BENCHMARK_TIMEOUT_SECS,
        )
        if perf_child.poll() is None:
            perf_child.send_signal(signal.SIGINT)
            try:
                perf_child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                perf_child.kill()
        if not perf_data.exists():
            return _error_result("perf record produced no data file.")
        subprocess.run(
            [
                "sh",
                "-c",
                f"perf script -i {perf_data} | stackcollapse-perf.pl | flamegraph.pl > {flamegraph_svg}",
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        top_functions = _extract_top_functions(work_dir, perf_data)
        profiling_dir = work_dir / "profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        round_num = len(list(profiling_dir.glob("report_*.txt"))) + 1
        saved_fg_path = ""
        if flamegraph_svg.exists():
            saved_fg = profiling_dir / f"flamegraph_{round_num:03d}.svg"
            shutil.copy2(flamegraph_svg, saved_fg)
            saved_fg_path = saved_fg.relative_to(work_dir).as_posix()
        report_path = profiling_dir / f"report_{round_num:03d}.txt"
        report_lines = [f"Profiling Report #{round_num:03d}", "", "Top Functions:"]
        for item in top_functions:
            report_lines.append(f"  {item['percentage']:>6.2f}%  {item['function']}")
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return {
            "type": "RunProfiling",
            "flamegraph_svg_path": saved_fg_path,
            "top_functions": top_functions,
            "total_samples": max(1, int(sum(item["percentage"] * 100 for item in top_functions))),
        }
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Profiling failed: {exc}")
    finally:
        if perf_child is not None and perf_child.poll() is None:
            perf_child.kill()
        _kill_server(child)


def _extract_top_functions(work_dir: Path, perf_data: Path) -> list[JsonDict]:
    proc = subprocess.run(
        [
            "perf",
            "report",
            "-i",
            str(perf_data),
            "--stdio",
            "--no-children",
            "-n",
            "--percent-limit",
            "1.0",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return []
    functions: list[JsonDict] = []
    for line in proc.stdout.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#") or "%" not in trimmed:
            continue
        pct_str = trimmed.split("%", 1)[0].strip()
        try:
            percentage = float(pct_str)
        except ValueError:
            continue
        marker = "[.] " if "[.] " in trimmed else ("[k] " if "[k] " in trimmed else "")
        if not marker:
            continue
        func_name = trimmed.split(marker, 1)[1].strip()
        if func_name:
            functions.append({"function": func_name, "percentage": percentage})
    functions.sort(key=lambda item: item["percentage"], reverse=True)
    return functions[:10]
