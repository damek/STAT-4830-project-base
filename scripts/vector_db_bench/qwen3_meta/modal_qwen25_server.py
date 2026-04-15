#!/usr/bin/env python3
"""Persistent warm Modal server for Qwen2.5-Coder-32B-Instruct.

This deploys a long-lived Modal class that starts a local vLLM server once in a
container enter hook, keeps one warm container alive, and exposes simple remote
methods for status checks and no-tools chat completions.

Usage:
  modal deploy scripts/vector_db_bench/qwen3_meta/modal_qwen25_server.py

  # Query the deployed warm server from your machine:
  modal run scripts/vector_db_bench/qwen3_meta/modal_qwen25_server.py \
    --prompt "Write a Rust function that computes Fibonacci iteratively."
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "alpha-vdb-qwen25-server"
CLASS_NAME = "Qwen25Server"
MODEL_VOLUME_NAME = "vdb-qwen3-models"
MODEL_MOUNT = "/models"
MODEL_SUBDIR = "Qwen--Qwen2.5-Coder-32B-Instruct"
MODEL_PATH = f"{MODEL_MOUNT}/{MODEL_SUBDIR}"
SERVER_PORT = 8000
GPU_SPEC = "A100-80GB"
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MIN_CONTAINERS = 1

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
                f"[server] waiting for vLLM on {host}:{port} "
                f"(check={checks}, pid={proc.pid}, returncode={proc.poll()})",
                flush=True,
            )
            tail = _tail(log_path, lines=30)
            if tail:
                print(f"[server] current log tail:\n{tail}", flush=True)

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
                    "log_tail": _tail(log_path, lines=80),
                }
        time.sleep(0.5)

    raise TimeoutError(
        f"server did not open {host}:{port} within {timeout_seconds}s.\n"
        f"process_returncode={proc.poll()}\n"
        f"log tail:\n{_tail(log_path, lines=160)}"
    )


@app.cls(
    image=image,
    gpu=GPU_SPEC,
    timeout=60 * 60,
    memory=65536,
    volumes={MODEL_MOUNT: model_volume},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class Qwen25Server:
    @modal.enter()
    def start_server(self) -> None:
        from openai import OpenAI

        self.model_path = MODEL_PATH
        self.served_model_name = "Qwen2.5-Coder-32B-Instruct"
        self.max_model_len = 4096
        self.startup_timeout_seconds = 1200
        self.enforce_eager = True

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"missing model path in volume: {model_path}")

        self.log_path = Path("/tmp/qwen25_vllm_server.log")
        log_handle = self.log_path.open("w", encoding="utf-8")
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
            self.served_model_name,
            "--max-model-len",
            str(self.max_model_len),
            "--tensor-parallel-size",
            "1",
            "--generation-config",
            "vllm",
        ]
        if self.enforce_eager:
            server_cmd.append("--enforce-eager")

        self.startup_context = {
            "model_path": str(model_path),
            "model_path_info": _describe_path(model_path),
            "server_cmd": server_cmd,
            "cwd": os.getcwd(),
            "gpu_spec": GPU_SPEC,
            "max_model_len": self.max_model_len,
            "served_model_name": self.served_model_name,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "enforce_eager": self.enforce_eager,
            "min_containers": MIN_CONTAINERS,
            "scaledown_window_seconds": SCALEDOWN_WINDOW_SECONDS,
        }
        print(
            f"[server] startup context:\n{json.dumps(self.startup_context, indent=2)}",
            flush=True,
        )

        self.server_proc = subprocess.Popen(
            server_cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env,
            bufsize=1,
        )
        print(f"[server] launched vLLM pid={self.server_proc.pid}", flush=True)
        self.startup_info = _wait_for_server(
            proc=self.server_proc,
            host="127.0.0.1",
            port=SERVER_PORT,
            timeout_seconds=self.startup_timeout_seconds,
            log_path=self.log_path,
        )
        print(
            f"[server] vLLM is reachable:\n{json.dumps(self.startup_info, indent=2)}",
            flush=True,
        )
        self.client = OpenAI(base_url=f"http://127.0.0.1:{SERVER_PORT}/v1", api_key="EMPTY")

    @modal.exit()
    def stop_server(self) -> None:
        if self.server_proc is None:
            return
        print(f"[server] shutting down vLLM pid={self.server_proc.pid}", flush=True)
        self.server_proc.terminate()
        try:
            self.server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.server_proc.kill()
            self.server_proc.wait(timeout=20)

    @modal.method()
    def status(self) -> dict[str, Any]:
        return {
            "startup_context": self.startup_context,
            "startup_info": {
                "checks": self.startup_info["checks"],
                "wait_seconds": self.startup_info["wait_seconds"],
            },
            "server_log_tail": _tail(self.log_path, lines=60),
            "server_running": self.server_proc is not None and self.server_proc.poll() is None,
        }

    @modal.method()
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        started = time.time()
        completion = self.client.chat.completions.create(
            model=self.served_model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - started
        message = completion.choices[0].message
        return {
            "response_text": message.content,
            "response_seconds": elapsed,
            "usage": completion.usage.model_dump() if completion.usage else None,
        }


@app.local_entrypoint()
def main(
    prompt: str = "Write a Rust function that computes Fibonacci iteratively.",
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    status_only: bool = False,
) -> None:
    remote_cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    server = remote_cls()
    if status_only:
        result = server.status.remote()
    else:
        result = server.generate.remote(
            prompt=prompt,
            system_prompt=system_prompt or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    print(json.dumps(result, indent=2))
