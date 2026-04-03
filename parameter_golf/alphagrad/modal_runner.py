"""
Modal runner — proxy evaluation on A10G GPU.

Architecture:
  - Local machine calls proxy_eval_remote.remote(config, script_content)
  - Modal runs the base train_gpt.py on A10G for ~2.5 min
  - Returns val_bpb (lower is better)

Data setup (run once before first use):
  python -m alphagrad.modal_runner

This downloads 1 shard of FineWeb sp1024 into a persistent Modal Volume.
Subsequent runs reuse the cached data.

NOTE: Proxy uses the BASE train_gpt.py (no Flash Attention 3) so it runs on A10G.
      Full evals (8xH100, SOTA script) are run manually on RunPod — main.py prints
      the exact command to run.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

# ── Modal app + persistent data volume ──────────────────────────────────────
app = modal.App("alphagrad-parameter-golf")
data_vol = modal.Volume.from_name("pg-fineweb-data", create_if_missing=True)

DATA_REMOTE = "/pg_data"
DATASET_DIR = f"{DATA_REMOTE}/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = f"{DATA_REMOTE}/tokenizers/fineweb_1024_bpe.model"

# ── Container image (no FA3 needed — base script only) ──────────────────────
proxy_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "sentencepiece",
        "numpy",
        "huggingface-hub>=0.24",
        "datasets",
        "tqdm",
    )
)

# ── Data setup (run once) ────────────────────────────────────────────────────
@app.function(
    image=proxy_image,
    volumes={DATA_REMOTE: data_vol},
    timeout=3600,
    cpu=4,
)
def setup_data(train_shards: int = 1) -> None:
    """Download FineWeb sp1024 to the persistent volume. Run once."""
    import os
    from huggingface_hub import hf_hub_download

    REPO_ID = "willdepueoai/parameter-golf"

    datasets_dir = Path(f"{DATA_REMOTE}/datasets/fineweb10B_sp1024")
    tokenizers_dir = Path(f"{DATA_REMOTE}/tokenizers")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    def dl(remote_path: str, local_path: Path) -> None:
        if local_path.exists():
            print(f"  already exists: {local_path.name}")
            return
        print(f"  downloading: {remote_path}")
        tmp = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
        )
        import shutil
        shutil.copy(tmp, local_path)

    # Tokenizer
    dl(
        "datasets/tokenizers/fineweb_1024_bpe.model",
        tokenizers_dir / "fineweb_1024_bpe.model",
    )

    # Validation split
    dl(
        "datasets/datasets/fineweb10B_sp1024/fineweb_val_000000.bin",
        datasets_dir / "fineweb_val_000000.bin",
    )

    # Training shards
    for i in range(train_shards):
        dl(
            f"datasets/datasets/fineweb10B_sp1024/fineweb_train_{i:06d}.bin",
            datasets_dir / f"fineweb_train_{i:06d}.bin",
        )

    data_vol.commit()
    print(f"Data setup complete. {train_shards} training shard(s) downloaded.")


# ── Proxy evaluation ─────────────────────────────────────────────────────────
@app.function(
    image=proxy_image,
    gpu="a10g",
    volumes={DATA_REMOTE: data_vol},
    timeout=600,
)
def proxy_eval_remote(config: dict, script_content: str) -> dict:
    """
    Run training on A10G and return {"bpb": float, "steps": int, "log_tail": str}.

    config:         Env var overrides (merged on top of DATA_PATH / TOKENIZER_PATH).
    script_content: Content of train_gpt.py to run (base script, no FA3).
    """
    import os

    # Patch out torch.compile for proxy evals — A10G doesn't need Triton,
    # we just need a BPB signal in eager mode.
    patched = script_content.replace(
        "torch.compile(base_model, dynamic=False, fullgraph=True)",
        "base_model",
    ).replace(
        "zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)",
        "pass  # torch.compile disabled for proxy eval",
    )

    # Write script to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(patched)
        script_path = f.name

    # Build environment
    env = os.environ.copy()
    env["DATA_PATH"] = DATASET_DIR
    env["TOKENIZER_PATH"] = TOKENIZER_PATH
    for k, v in config.items():
        env[str(k)] = str(v)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", script_path
    ]

    # Allow training wallclock + 120s for final eval/quantization.
    # If it exceeds that, kill and use partial output (intermediate val_bpb).
    wallclock = float(config.get("MAX_WALLCLOCK_SECONDS", 150))
    proc_timeout = int(wallclock) + 120

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=proc_timeout,
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired as e:
        def _decode(b):
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="replace")
            return b or ""
        output = _decode(e.stdout) + "\n" + _decode(e.stderr)

    # Parse last intermediate val_bpb
    matches = re.findall(r"val_bpb:([\d.]+)", output)
    bpb = float(matches[-1]) if matches else float("inf")

    # Parse last step count
    step_matches = re.findall(r"step:(\d+)/\d+", output)
    steps = int(step_matches[-1]) if step_matches else 0

    return {
        "bpb": bpb,
        "steps": steps,
        "log_tail": output[-800:],
    }


# ── Local entrypoint — data setup ────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """Run `python -m alphagrad.modal_runner` to set up the data volume."""
    print("Setting up FineWeb data on Modal volume 'pg-fineweb-data'...")
    print("This downloads 1 training shard (~100M tokens) + validation split.")
    print("Takes ~5-10 minutes. Only needs to run once.\n")
    setup_data.remote(train_shards=1)
    print("\nDone. You can now run the optimizer.")
