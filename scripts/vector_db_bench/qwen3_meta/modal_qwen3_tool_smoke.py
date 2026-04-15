#!/usr/bin/env python3
"""Quick Modal smoke test for Qwen3-Coder-Next inference + tool use.

This script mounts the previously cached Qwen3-Coder-Next weights from a Modal
Volume, launches a local vLLM OpenAI-compatible server with the official
`qwen3_coder` tool parser, and runs one small tool-using interaction.

Usage:
  modal run scripts/vector_db_bench/qwen3_meta/modal_qwen3_tool_smoke.py
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

APP_NAME = "alpha-vdb-qwen3-tool-smoke"
MODEL_VOLUME_NAME = "vdb-qwen3-models"
MODEL_MOUNT = "/models"
MODEL_SUBDIR = "Qwen--Qwen3-Coder-Next-Base"
MODEL_PATH = f"{MODEL_MOUNT}/{MODEL_SUBDIR}"
SERVER_PORT = 8000
GPU_SPEC = "A100-80GB:4"

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


def _wait_for_port(host: str, port: int, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"server did not open {host}:{port} within {timeout_seconds}s")


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
    last_tail = ""
    while time.time() < deadline:
        checks += 1
        if checks == 1 or checks % 20 == 0:
            last_tail = _tail(log_path, lines=40)
            print(
                f"[smoke] waiting for vLLM on {host}:{port} "
                f"(check={checks}, pid={proc.pid}, returncode={proc.poll()})",
                flush=True,
            )
            if last_tail:
                print(f"[smoke] current log tail:\n{last_tail}", flush=True)

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


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "square_number",
                "description": "Return the square of an input integer.",
                "parameters": {
                    "type": "object",
                    "required": ["input_num"],
                    "properties": {
                        "input_num": {
                            "type": "integer",
                            "description": "Integer to square.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_vector_db_surface",
                "description": "Return the editable and protected files for the vector-db-bench task.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


def _execute_tool(name: str, arguments_json: str) -> dict[str, Any]:
    args = json.loads(arguments_json or "{}")
    if name == "square_number":
        value = int(args["input_num"])
        return {"input_num": value, "squared": value * value}
    if name == "list_vector_db_surface":
        return {
            "editable": [
                "skeleton/src/db.rs",
                "skeleton/src/distance.rs",
                "skeleton/Cargo.toml",
            ],
            "protected": [
                "skeleton/src/api.rs",
                "skeleton/src/main.rs",
            ],
        }
    raise ValueError(f"unknown tool: {name}")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=60 * 30,
    memory=65536,
    volumes={MODEL_MOUNT: model_volume},
)
def run_tool_smoke(
    prompt: str = "Use the tools to tell me the editable vector-db-bench files and compute 37 squared. Keep the answer under two sentences.",
    max_model_len: int = 8192,
    max_rounds: int = 4,
) -> dict[str, Any]:
    from openai import OpenAI

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"missing model path in volume: {model_path}")

    log_path = Path("/tmp/qwen3_vllm.log")
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
        "Qwen3-Coder-Next",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        str(max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]

    startup_context = {
        "model_path": str(model_path),
        "model_path_info": _describe_path(model_path),
        "parser_file_present": (model_path / "qwen3coder_tool_parser_vllm.py").exists(),
        "detector_file_present": (model_path / "qwen3_coder_detector_sgl.py").exists(),
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
            timeout_seconds=600,
            log_path=log_path,
        )
        print(f"[smoke] vLLM is reachable:\n{json.dumps(startup_info, indent=2)}", flush=True)

        client = OpenAI(base_url=f"http://127.0.0.1:{SERVER_PORT}/v1", api_key="EMPTY")
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = _tool_schema()
        transcript: list[dict[str, Any]] = []

        final_text = ""
        for round_idx in range(max_rounds):
            completion = client.chat.completions.create(
                model="Qwen3-Coder-Next",
                messages=messages,
                tools=tools,
                temperature=1.0,
                top_p=0.95,
                extra_body={"top_k": 40},
                max_tokens=1024,
            )
            message = completion.choices[0].message

            tool_calls = message.tool_calls or []
            transcript.append(
                {
                    "round": round_idx + 1,
                    "assistant_content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in tool_calls
                    ],
                }
            )

            if not tool_calls:
                final_text = message.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                tool_result = _execute_tool(tc.function.name, tc.function.arguments)
                transcript.append(
                    {
                        "round": round_idx + 1,
                        "tool_result": {
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "result": tool_result,
                        },
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        return {
            "prompt": prompt,
            "final_text": final_text,
            "transcript": transcript,
            "startup_context": startup_context,
            "startup_info": startup_info,
            "server_log_tail": _tail(log_path, lines=120),
        }
    finally:
        print(f"[smoke] shutting down vLLM pid={server_proc.pid}", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_handle.close()


@app.local_entrypoint()
def main(
    prompt: str = "Use the tools to tell me the editable vector-db-bench files and compute 37 squared. Keep the answer under two sentences.",
) -> None:
    result = run_tool_smoke.remote(prompt=prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
