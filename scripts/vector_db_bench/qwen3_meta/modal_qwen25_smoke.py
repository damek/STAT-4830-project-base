#!/usr/bin/env python3
"""Quick Modal smoke test for Qwen2.5-Coder-32B-Instruct inference.

This mounts the cached Qwen2.5-Coder-32B-Instruct weights from a Modal volume,
launches a local vLLM OpenAI-compatible server on a single A100-80GB, and runs
one short completion request to validate end-to-end inference.

Usage:
  modal run scripts/vector_db_bench/qwen3_meta/modal_qwen25_smoke.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "alpha-vdb-qwen25-smoke"
MODEL_VOLUME_NAME = "vdb-qwen3-models"
MODEL_MOUNT = "/models"
MODEL_SUBDIR = "Qwen--Qwen2.5-Coder-32B-Instruct"
MODEL_PATH = f"{MODEL_MOUNT}/{MODEL_SUBDIR}"
SERVER_PORT = 8000
GPU_SPEC = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.15.0",
        "openai>=1.30.0",
        "httpx>=0.28.0",
    )
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _describe_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if path.is_file():
        return {"exists": True, "type": "file", "size_bytes": path.stat().st_size}
    children = sorted(p.name for p in path.iterdir())
    return {
        "exists": True,
        "type": "dir",
        "child_count": len(children),
        "sample_children": children[:20],
    }


def _wait_for_server(
    *,
    proc: subprocess.Popen[str],
    host: str,
    port: int,
    timeout_seconds: int,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    checks = 0
    while time.time() < deadline:
        checks += 1
        if checks == 1 or checks % 20 == 0:
            print(
                f"[smoke] waiting for vLLM on {host}:{port} "
                f"(check={checks}, pid={proc.pid}, returncode={proc.poll()})",
                flush=True,
            )
            tail = _tail(log_path, lines=40)
            if tail:
                print(f"[smoke] current log tail:\n{tail}", flush=True)

        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM exited before opening port {host}:{port} "
                f"(returncode={returncode}).\n"
                f"log tail:\n{_tail(log_path, lines=120)}"
            )

        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return {
                    "checks": checks,
                    "wait_seconds": timeout_seconds - max(deadline - time.time(), 0.0),
                    "log_tail": _tail(log_path, lines=120),
                }
        time.sleep(0.5)

    raise TimeoutError(
        f"server did not open {host}:{port} within {timeout_seconds}s.\n"
        f"process_returncode={proc.poll()}\n"
        f"log tail:\n{_tail(log_path, lines=160)}"
    )


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=60 * 25,
    memory=65536,
    volumes={MODEL_MOUNT: model_volume},
)
def run_smoke(
    prompt: str = (
        "Write a short Rust function `fn square(x: i64) -> i64` and then in one "
        "sentence explain what it does."
    ),
    max_model_len: int = 4096,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> dict[str, Any]:
    from openai import OpenAI

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"missing model path in volume: {model_path}")

    log_path = Path("/tmp/qwen25_vllm.log")
    log_handle = log_path.open("w", encoding="utf-8")
    server_env = dict(os.environ)
    server_env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        "vllm",
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--served-model-name",
        "Qwen2.5-Coder-32B-Instruct",
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
    ]

    startup_context = {
        "model_path": str(model_path),
        "model_path_info": _describe_path(model_path),
        "server_cmd": server_cmd,
        "cwd": os.getcwd(),
        "gpu_spec": GPU_SPEC,
        "max_model_len": max_model_len,
    }
    print(f"[smoke] startup context:\n{json.dumps(startup_context, indent=2)}", flush=True)

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=server_env,
        bufsize=1,
    )
    print(f"[smoke] launched vLLM pid={server_proc.pid}", flush=True)

    try:
        startup_info = _wait_for_server(
            proc=server_proc,
            host="127.0.0.1",
            port=SERVER_PORT,
            timeout_seconds=1200,
            log_path=log_path,
        )
        print(f"[smoke] vLLM is reachable:\n{json.dumps(startup_info, indent=2)}", flush=True)

        client = OpenAI(base_url=f"http://127.0.0.1:{SERVER_PORT}/v1", api_key="EMPTY")
        started = time.time()
        completion = client.chat.completions.create(
            model="Qwen2.5-Coder-32B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - started
        message = completion.choices[0].message
        return {
            "model_path": str(model_path),
            "startup_info": {
                "checks": startup_info["checks"],
                "wait_seconds": startup_info["wait_seconds"],
            },
            "prompt": prompt,
            "response_text": message.content,
            "response_seconds": elapsed,
            "usage": completion.usage.model_dump() if completion.usage else None,
            "server_log_tail": _tail(log_path, lines=40),
        }
    finally:
        print(f"[smoke] shutting down vLLM pid={server_proc.pid}", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=20)
        log_handle.close()


@app.local_entrypoint()
def main(
    prompt: str = (
        "Write a short Rust function `fn square(x: i64) -> i64` and then in one "
        "sentence explain what it does."
    ),
    max_model_len: int = 4096,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> None:
    result = run_smoke.remote(
        prompt=prompt,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(json.dumps(result, indent=2))
