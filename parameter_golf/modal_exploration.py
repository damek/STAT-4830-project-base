#!/usr/bin/env python3
"""
Run train_gpt_exploration.py on a Modal GPU and stream logs back locally.

Authentication
--------------
One-time: run ``modal token set --token-id <key>`` to authenticate.

Data volume
-----------
The FineWeb data must already be in the Modal volume ``pg-fineweb-data``.
If not set up yet, run once:
  python -m alphagrad.modal_runner

Usage
-----
  cd parameter_golf
  modal run modal_exploration.py                         # defaults (8×H100, 600s wall)
  modal run modal_exploration.py --iterations 5000
  modal run modal_exploration.py --max-wallclock 300

Output
------
Logs are streamed to your terminal.
``output/final_model.int8.ptz`` is written locally when training finishes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

# ── Shared constants (same volume as alphagrad/modal_runner.py) ──────────────
DATA_REMOTE = "/pg_data"
DATASET_DIR = f"{DATA_REMOTE}/datasets/fineweb10B_sp1024"
TOKENIZER_PATH_REMOTE = f"{DATA_REMOTE}/tokenizers/fineweb_1024_bpe.model"

data_vol = modal.Volume.from_name("pg-fineweb-data", create_if_missing=True)

app = modal.App("pg-exploration")

exploration_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("gcc")  # Triton needs a C compiler to JIT-compile its CUDA driver shim
    .pip_install(
        "sentencepiece",
        "numpy",
        "huggingface-hub>=0.24",
        "datasets",
        "tqdm",
    )
)

# ── Remote function ───────────────────────────────────────────────────────────
@app.function(
    image=exploration_image,
    gpu="h100:8",
    volumes={DATA_REMOTE: data_vol},
    timeout=2400,
)
def run_exploration(script_content: str, config: dict, gpu_type: str = "h100x8") -> dict:
    """
    Run train_gpt_exploration.py on 8×H100 GPUs.

    Args:
        script_content: Full text of train_gpt_exploration.py.
        config:         Dict of env-var overrides (ITERATIONS, MAX_WALLCLOCK_SECONDS, ...).
        gpu_type:       Informational label only.

    Returns:
        {
          "log":      full stdout+stderr string,
          "bpb":      final val_bpb (float or None),
          "artifact": bytes of final_model.int8.ptz (or None if not produced),
        }
    """
    import os as _os

    # torch.compile is fully supported on H100 (Hopper / Triton) — no patching needed.
    patched = script_content

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(patched)
        script_path = f.name

    env = _os.environ.copy()
    env["DATA_PATH"] = DATASET_DIR
    env["TOKENIZER_PATH"] = TOKENIZER_PATH_REMOTE
    for k, v in config.items():
        env[str(k)] = str(v)

    # Match the training script's own default (600s). proc_timeout adds a large buffer
    # for torch.compile JIT compilation (first run) + quantization + round-trip eval.
    wallclock = float(config.get("MAX_WALLCLOCK_SECONDS", 600))
    proc_timeout = int(wallclock) + 600

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", script_path]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=proc_timeout,
            cwd="/tmp",  # script writes final_model.int8.ptz to cwd; we look in /tmp
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired as e:
        def _decode(b: bytes | str | None) -> str:
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="replace")
            return b or ""
        output = _decode(e.stdout) + "\n" + _decode(e.stderr)

    # Parse final val_bpb from log
    matches = re.findall(r"val_bpb:([\d.]+)", output)
    bpb = float(matches[-1]) if matches else None

    # Read artifact if produced
    artifact: bytes | None = None
    ptz_path = "/tmp/final_model.int8.ptz"
    if _os.path.exists(ptz_path):
        with open(ptz_path, "rb") as af:
            artifact = af.read()
    else:
        # training script writes to cwd; also check script dir
        for candidate in [
            _os.path.join(_os.path.dirname(script_path), "final_model.int8.ptz"),
            "final_model.int8.ptz",
        ]:
            if _os.path.exists(candidate):
                with open(candidate, "rb") as af:
                    artifact = af.read()
                break

    return {"log": output, "bpb": bpb, "artifact": artifact}


# ── Local entrypoint ──────────────────────────────────────────────────────────
# Modal parses typed parameters from the CLI automatically.
# Usage:  modal run modal_exploration.py --gpu h100 --iterations 20000
@app.local_entrypoint()
def main(
    gpu: str = "a10g",
    iterations: int = 0,
    max_wallclock: float = 0.0,
    warmdown_iters: int = 0,
    seed: int = 0,
    run_id: str = "",
    val_loss_every: int = 0,
) -> None:
    # Build config dict from non-default values only
    config: dict[str, object] = {}
    if iterations:
        config["ITERATIONS"] = iterations
    if max_wallclock:
        config["MAX_WALLCLOCK_SECONDS"] = max_wallclock
    if warmdown_iters:
        config["WARMDOWN_ITERS"] = warmdown_iters
    if seed:
        config["SEED"] = seed
    if run_id:
        config["RUN_ID"] = run_id
    if val_loss_every:
        config["VAL_LOSS_EVERY"] = val_loss_every

    script_path = Path(__file__).resolve().parent / "train_gpt_exploration.py"
    if not script_path.is_file():
        print(f"ERROR: cannot find {script_path}", file=sys.stderr)
        raise SystemExit(1)
    script_content = script_path.read_text(encoding="utf-8")

    print("Dispatching train_gpt_exploration.py to Modal (8×H100)...")
    print(f"Config overrides: {config or '(none — using script defaults)'}")
    print("Logs will stream here; artifact saved to output/ when done.\n")

    result = run_exploration.remote(script_content, config, gpu)

    print(result["log"])

    if result["bpb"] is not None:
        print(f"\nFinal val_bpb: {result['bpb']:.4f}")
    else:
        print("\nval_bpb not found in log output.")

    if result["artifact"]:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(exist_ok=True)
        label = config.get("RUN_ID", "latest")
        out_path = out_dir / f"final_model_{label}.int8.ptz"
        out_path.write_bytes(result["artifact"])
        size_mb = len(result["artifact"]) / 1e6
        print(f"Artifact saved → {out_path}  ({size_mb:.2f} MB)")
    else:
        print("No artifact (final_model.int8.ptz) was produced by the run.")
