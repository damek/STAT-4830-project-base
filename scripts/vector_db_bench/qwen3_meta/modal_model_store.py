#!/usr/bin/env python3
"""Store Qwen3-Coder-Next weights in a Modal Volume.

Usage:
  modal run scripts/vector_db_bench/qwen3_meta/modal_model_store.py::download_model

Optional:
  modal run scripts/vector_db_bench/qwen3_meta/modal_model_store.py::download_model \
    --repo-id Qwen/Qwen3-Coder-Next-Base \
    --revision main
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "alpha-vdb-qwen3-meta-models"
VOLUME_NAME = "vdb-qwen3-models"
VOLUME_MOUNT = "/models"
DEFAULT_REPO_ID = "Qwen/Qwen3-Coder-Next-Base"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.30.0")
)

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _target_dir(repo_id: str) -> str:
    safe = repo_id.replace("/", "--")
    return str(Path(VOLUME_MOUNT) / safe)


@app.function(
    image=image,
    gpu=None,
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: models_volume},
    memory=8192,
)
def download_model(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> dict:
    from huggingface_hub import snapshot_download

    local_dir = _target_dir(repo_id)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    models_volume.commit()
    return {
        "repo_id": repo_id,
        "revision": revision,
        "volume_name": VOLUME_NAME,
        "local_dir": path,
    }


@app.local_entrypoint()
def main(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> None:
    result = download_model.remote(repo_id=repo_id, revision=revision)
    print(result)
