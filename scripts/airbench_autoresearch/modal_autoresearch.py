#!/usr/bin/env python3
"""Long-lived Modal-hosted autoresearch loop for AirBench."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
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

    from airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from modal_airbench import (
        DEFAULT_GPU,
        REMOTE_DATA_DIR,
        _run_airbench_candidate_impl,
        cifar_volume,
        image as base_image,
    )

    from loop_core import (
        AutoresearchLoopConfig,
        ensure_auth,
        load_dotenv,
        normalize_target_accuracy,
        run_meta_autoresearch_loop,
    )
else:
    from ..airbench_gepa.airbench_evaluator import AirbenchEvalConfig, evaluate_solver_code
    from ..airbench_gepa.modal_airbench import (
        DEFAULT_GPU,
        REMOTE_DATA_DIR,
        _run_airbench_candidate_impl,
        cifar_volume,
        image as base_image,
    )
    from .loop_core import (
        AutoresearchLoopConfig,
        ensure_auth,
        load_dotenv,
        normalize_target_accuracy,
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
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs = dict(
        run_name=run_name,
        initial_candidate_code=DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"),
        initial_program_text=DEFAULT_PROGRAM_PATH.read_text(encoding="utf-8"),
        initial_strategy_text=DEFAULT_STRATEGY_PATH.read_text(encoding="utf-8"),
        initial_memory_text=DEFAULT_MEMORY_PATH.read_text(encoding="utf-8"),
        initial_record_text=DEFAULT_RECORD_PATH.read_text(encoding="utf-8"),
        model=model,
        strategy_model=strategy_model or None,
        strategy_rounds=strategy_rounds,
        max_attempts=max_attempts,
        target_accuracy=target_accuracy,
        proxy_trials=proxy_trials,
        strict_trials=strict_trials,
        warmup_trials=warmup_trials,
        timeout_seconds=timeout_seconds,
        final_strict_eval=final_strict_eval,
        stream_logs=stream_logs,
    )

    if wait_for_result:
        result = run_autoresearch_loop_remote.remote(**kwargs)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return

    function_call = run_autoresearch_loop_remote.spawn(**kwargs)
    print(
        json.dumps(
            {
                "run_name": run_name,
                "function_call_id": function_call.object_id,
                "remote_run_dir": f"{REMOTE_RUN_ROOT}/{run_name}",
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
