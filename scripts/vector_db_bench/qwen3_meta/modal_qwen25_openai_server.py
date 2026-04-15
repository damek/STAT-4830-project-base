#!/usr/bin/env python3
"""Deploy a persistent warm OpenAI-compatible vLLM server for Qwen2.5-32B.

Usage:
  modal deploy scripts/vector_db_bench/qwen3_meta/modal_qwen25_openai_server.py

  # Print deployed URL and optionally run one test query
  modal run scripts/vector_db_bench/qwen3_meta/modal_qwen25_openai_server.py
"""

import json
import subprocess
from pathlib import Path

import modal

APP_NAME = "alpha-vdb-qwen25-openai"
FUNCTION_NAME = "serve_qwen25_openai"
MODEL_VOLUME_NAME = "vdb-qwen3-models"
MODEL_MOUNT = "/models"
MODEL_SUBDIR = "Qwen--Qwen2.5-Coder-32B-Instruct"
MODEL_PATH = f"{MODEL_MOUNT}/{MODEL_SUBDIR}"
SERVER_PORT = 8000
GPU_SPEC = "A100-80GB"
MIN_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 20 * 60
STARTUP_TIMEOUT_SECONDS = 1200
MAX_MODEL_LEN = 8192

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


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=60 * 60,
    memory=65536,
    volumes={MODEL_MOUNT: model_volume},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
@modal.web_server(port=SERVER_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="qwen25-openai")
def serve_qwen25_openai() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"missing model path in volume: {model_path}")

    server_cmd = [
        "vllm",
        "serve",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--served-model-name",
        "Qwen2.5-Coder-32B-Instruct",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        "1",
        "--generation-config",
        "vllm",
        "--enforce-eager",
    ]
    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "gpu_spec": GPU_SPEC,
                "max_model_len": MAX_MODEL_LEN,
                "server_cmd": server_cmd,
                "min_containers": MIN_CONTAINERS,
                "scaledown_window_seconds": SCALEDOWN_WINDOW_SECONDS,
            },
            indent=2,
        ),
        flush=True,
    )
    subprocess.Popen(server_cmd)


@app.local_entrypoint()
def main(
    prompt: str = "Write a short Rust function that reverses a vector in place.",
    run_test_query: bool = False,
) -> None:
    from openai import OpenAI

    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    web_url = fn.get_web_url()
    payload = {
        "function_name": FUNCTION_NAME,
        "web_url": web_url,
        "openai_base_url": f"{web_url}/v1",
    }

    if run_test_query:
        client = OpenAI(base_url=f"{web_url}/v1", api_key="DUMMY")
        completion = client.chat.completions.create(
            model="Qwen2.5-Coder-32B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=256,
        )
        payload["prompt"] = prompt
        payload["response_text"] = completion.choices[0].message.content
        payload["usage"] = completion.usage.model_dump() if completion.usage else None

    print(json.dumps(payload, indent=2))
