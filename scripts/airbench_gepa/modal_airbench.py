#!/usr/bin/env python3
"""Modal execution harness for AirBench candidate scripts."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "airbench-gepa"
DEFAULT_GPU = "A100-40GB"
REMOTE_DATA_DIR = "/vol/cifar10"
TORCH_WHEEL_INDEX = "https://download.pytorch.org/whl/cu124"

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "torchvision==0.19.1",
    extra_index_url=TORCH_WHEEL_INDEX,
)

cifar_volume = modal.Volume.from_name(
    "airbench-cifar10",
    create_if_missing=True,
    version=2,
)


def _tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _expected_device_marker(requested_gpu: str) -> str | None:
    requested = requested_gpu.upper()
    if "40GB" in requested:
        return "40GB"
    if "80GB" in requested:
        return "80GB"
    return None


def _extract_last_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    payload = json.loads(stdout.strip())
    if not isinstance(payload, dict):
        raise ValueError("candidate output JSON must be an object")
    return payload


def _run_subprocess(
    *,
    argv: list[str],
    cwd: str,
    env: dict[str, str],
    timeout_seconds: int,
    stream_output: bool,
) -> tuple[int | None, str, str, bool]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started_at = time.time()

    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc:
        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_parts, sys.stdout))
        if proc.stderr is not None:
            selector.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_parts, sys.stderr))

        timed_out = False
        while selector.get_map():
            remaining = timeout_seconds - (time.time() - started_at)
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break

            events = selector.select(timeout=remaining)
            if not events and proc.poll() is not None:
                break

            for key, _ in events:
                stream_name, parts, sink = key.data
                chunk = key.fileobj.readline()
                if chunk == "":
                    selector.unregister(key.fileobj)
                    continue
                parts.append(chunk)
                if stream_output:
                    print(chunk, end="", file=sink)

        if timed_out:
            proc.wait()
            return None, "".join(stdout_parts), "".join(stderr_parts), True

        returncode = proc.wait()
        return returncode, "".join(stdout_parts), "".join(stderr_parts), False


def build_script_args(
    *,
    target_accuracy: float,
    trials: int,
    warmup_trials: int,
    data_dir: str = REMOTE_DATA_DIR,
    json_only: bool = True,
    preflight: bool = False,
    verbose: bool = False,
) -> list[str]:
    args = [
        "--data-dir",
        data_dir,
        "--trials",
        str(trials),
        "--warmup-trials",
        str(warmup_trials),
        "--target-accuracy",
        str(target_accuracy),
    ]
    if json_only:
        args.append("--json-only")
    if preflight:
        args.append("--preflight")
    if verbose:
        args.append("--verbose")
    return args


def _run_airbench_candidate_impl(
    solver_code: str,
    script_args: list[str],
    *,
    timeout_seconds: int,
    print_subprocess_logs: bool,
    requested_gpu: str,
) -> dict[str, Any]:
    import torch

    started_at = time.time()
    actual_device_name = torch.cuda.get_device_name(0)
    expected_marker = _expected_device_marker(requested_gpu)
    if expected_marker is not None and expected_marker not in actual_device_name.upper():
        if print_subprocess_logs:
            print(
                "[modal] gpu mismatch "
                f"requested={requested_gpu} actual_device_name={actual_device_name}"
            )
        return {
            "ok": False,
            "failure_type": "gpu_mismatch",
            "message": (
                f"requested GPU {requested_gpu}, but Modal attached "
                f"{actual_device_name}"
            ),
            "runtime_seconds": time.time() - started_at,
            "requested_gpu": requested_gpu,
            "actual_device_name": actual_device_name,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    with tempfile.TemporaryDirectory(prefix="airbench_candidate_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(solver_code, encoding="utf-8")

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if print_subprocess_logs:
            print(
                "[modal] starting candidate "
                f"timeout_seconds={timeout_seconds} script_args={script_args}"
            )

        returncode, stdout, stderr, timed_out = _run_subprocess(
            argv=[sys.executable, str(script_path), *script_args],
            cwd=tmpdir,
            env=env,
            timeout_seconds=timeout_seconds,
            stream_output=print_subprocess_logs,
        )
        if timed_out:
            if print_subprocess_logs:
                print("[modal] candidate timed out")
            return {
                "ok": False,
                "failure_type": "timeout",
                "message": f"candidate timed out after {timeout_seconds}s",
                "runtime_seconds": time.time() - started_at,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }

        runtime_seconds = time.time() - started_at
        if print_subprocess_logs:
            print(
                "[modal] candidate finished "
                f"returncode={returncode} runtime_seconds={runtime_seconds:.2f}"
            )

        # Persist dataset downloads cached under the volume mount.
        cifar_volume.commit()

        if returncode != 0:
            return {
                "ok": False,
                "failure_type": "runtime_error",
                "message": f"candidate exited with code {returncode}",
                "runtime_seconds": runtime_seconds,
                "returncode": returncode,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }

        try:
            payload = _extract_last_json(stdout)
        except Exception as exc:
            return {
                "ok": False,
                "failure_type": "invalid_json",
                "message": f"could not parse candidate JSON output: {exc}",
                "runtime_seconds": runtime_seconds,
                "returncode": returncode,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }

        payload["_modal"] = {
            "app_name": APP_NAME,
            "gpu": requested_gpu,
            "requested_gpu": requested_gpu,
            "actual_device_name": actual_device_name,
            "data_dir": REMOTE_DATA_DIR,
            "runtime_seconds": runtime_seconds,
        }

        return {
            "ok": True,
            "runtime_seconds": runtime_seconds,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "result": payload,
        }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={REMOTE_DATA_DIR: cifar_volume},
    timeout=60 * 30,
    cpu=8,
    memory=32768,
)
def run_airbench_candidate(
    solver_code: str,
    script_args: list[str],
    timeout_seconds: int = 60 * 15,
    print_subprocess_logs: bool = False,
) -> dict[str, Any]:
    return _run_airbench_candidate_impl(
        solver_code,
        script_args,
        timeout_seconds=timeout_seconds,
        print_subprocess_logs=print_subprocess_logs,
        requested_gpu=DEFAULT_GPU,
    )


@app.local_entrypoint(name="smoke")
def smoke(
    seed_path: str = str(Path(__file__).with_name("seeds") / "airbench94_baseline.py"),
    trials: int = 1,
    warmup_trials: int = 1,
    target_accuracy: float = 0.94,
) -> None:
    solver_code = Path(seed_path).read_text(encoding="utf-8")
    response = run_airbench_candidate.remote(
        solver_code=solver_code,
        script_args=build_script_args(
            target_accuracy=target_accuracy,
            trials=trials,
            warmup_trials=warmup_trials,
            json_only=True,
            preflight=True,
            verbose=True,
        ),
        print_subprocess_logs=True,
    )
    print(json.dumps(response, indent=2, sort_keys=True, default=str))
