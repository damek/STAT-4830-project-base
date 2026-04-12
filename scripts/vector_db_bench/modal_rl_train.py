#!/usr/bin/env python3
"""Modal GRPO reinforcement learning for Qwen2.5-Coder on vector-db-bench.

Reward signal (two phases, selectable via --reward-mode):
  compile  — 0.0 if cargo build fails, 1.0 if it succeeds.
             Use this first to teach the model to produce compilable Rust.
  qps      — 0.0 if build fails, small partial if recall < threshold,
             normalised QPS (vs. baseline) if fully valid.
             Use this after compile rate is consistently high.

Prerequisites:
  pip install modal
  modal token set ...
  Set VECTOR_DB_BENCH_ROOT to a local clone of KCORES/vector-db-bench.
  (For qps mode, also set VECTOR_DB_BENCH_DATA if data is not inside the repo.)

Usage:
  # Generate prompts first:
  python scripts/vector_db_bench/generate_rl_prompts.py \\
      --bench-repo /path/to/vector-db-bench

  # Phase 1 — compile reward (A10G, ~$1.10/hr):
  modal run scripts/vector_db_bench/modal_rl_train.py \\
      --prompts-path data/vector_db_bench/rl_prompts.jsonl \\
      --reward-mode compile

  # Phase 2 — QPS reward (A100-80GB for 32B):
  modal run scripts/vector_db_bench/modal_rl_train.py \\
      --prompts-path data/vector_db_bench/rl_prompts.jsonl \\
      --reward-mode qps

  # To use 32B instead of 7B, edit GPU_TYPE and MODEL_ID below.

Checkpoints are saved to Modal Volume "vdb-rl-checkpoints".
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config — edit these to switch model / GPU
# ---------------------------------------------------------------------------

GPU_TYPE = "A10G"                      # swap to "A100-80GB" for 32B
MODEL_ID  = "Qwen/Qwen2.5-Coder-7B-Instruct"  # swap to 32B variant if needed

APP_NAME           = "alpha-vdb-rl"
CHECKPOINT_VOLUME  = "vdb-rl-checkpoints"
CARGO_CACHE_VOLUME = "vdb-cargo-cache"
WARM_WS_VOLUME     = "vdb-warm-workspace"   # pre-built dep artifacts
CHECKPOINT_MOUNT   = "/checkpoints"
CARGO_CACHE_MOUNT  = "/root/.cargo/registry"
WARM_WS_MOUNT      = "/opt/vdb-warm"        # where the warm workspace lives
REMOTE_BENCH       = "/opt/vector-db-bench"
REMOTE_DATA        = "/opt/vdb-data"

# Incumbent Cargo.toml — kept as a module-level constant so both the warm-up
# function and the reward evaluator use exactly the same file.
INCUMBENT_CARGO_TOML = """\
[package]
name = "vector-db-skeleton"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1"
parking_lot = "0.12"
crossbeam = "0.8"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
"""

# ---------------------------------------------------------------------------
# Local paths + image — resolved only when running locally
# ---------------------------------------------------------------------------
# When imported inside a Modal container, VECTOR_DB_BENCH_ROOT is not set
# and the bench files are already baked into the image.  We guard all
# local-path-dependent setup with modal.is_local().

def _bench_ignore(p: Path) -> bool:
    """Return True to exclude a path from the bench image layer."""
    skip = {".git", "target", ".idea", ".vscode", "data"}
    return any(part in skip for part in p.parts)


if modal.is_local():
    _bench_root = os.environ.get("VECTOR_DB_BENCH_ROOT", "").strip()
    if not _bench_root:
        raise RuntimeError(
            "Set VECTOR_DB_BENCH_ROOT to the absolute path of your vector-db-bench clone."
        )
    _bench_path = Path(_bench_root).resolve()
    if not _bench_path.is_dir():
        raise RuntimeError(f"VECTOR_DB_BENCH_ROOT is not a directory: {_bench_path}")

    _data_root = os.environ.get("VECTOR_DB_BENCH_DATA", "").strip()
    _data_path = Path(_data_root).resolve() if _data_root else _bench_path / "data"

    rl_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "curl", "ca-certificates", "build-essential",
            "pkg-config", "libssl-dev", "clang", "git",
        )
        .run_commands(
            'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable',
            "bash -lc 'source /root/.cargo/env && rustc --version && cargo --version'",
        )
        .env({"PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .pip_install(
            "torch==2.6.0",
            extra_index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "transformers>=4.46.0",
            "trl>=0.12.0",
            "peft>=0.13.0",
            "bitsandbytes>=0.44.1",
            "datasets>=3.0.0",
            "accelerate>=1.0.0",
            "scipy",
        )
        .add_local_dir(
            str(_bench_path),
            remote_path=REMOTE_BENCH,
            copy=True,
            ignore=_bench_ignore,
        )
    )

    if _data_path.is_dir():
        rl_image = rl_image.add_local_dir(
            str(_data_path),
            remote_path=REMOTE_DATA,
            copy=True,
        )
else:
    # Inside the container — image already built, just need a placeholder.
    _bench_path = Path(REMOTE_BENCH)
    _data_path  = Path(REMOTE_DATA)
    rl_image    = modal.Image.debian_slim(python_version="3.11")

app = modal.App(APP_NAME)
checkpoints_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
cargo_cache_volume = modal.Volume.from_name(CARGO_CACHE_VOLUME, create_if_missing=True)
warm_ws_volume     = modal.Volume.from_name(WARM_WS_VOLUME,     create_if_missing=True)

# ---------------------------------------------------------------------------
# Reward evaluation helpers (run inside the Modal container)
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from model output.

    Models frequently wrap JSON in ```json ... ``` blocks.
    We strip the outer fences and return the inner content so json.loads works.
    """
    import re
    t = text.strip()
    # Match ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Also handle a single leading ``` without a closing one (truncated output)
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
    return t.strip()


def _evaluate_compile(
    completion: str,
    *,
    skeleton_dir: Path,
    incumbent_files: dict[str, str],
    warm_ws_dir: "Path | None" = None,
    build_timeout: int = 120,
) -> float:
    """Returns 1.0 if cargo build succeeds, 0.0 if build fails, -1.0 for invalid JSON/format.

    If warm_ws_dir points to a pre-built workspace (with target/ populated), each
    evaluation hardlink-copies the warm workspace so only user code is recompiled
    (~20s instead of 3-5 min for a cold build).
    """
    import json
    import shutil
    import subprocess
    import tempfile

    cleaned = _strip_fences(completion)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return -1.0

    if not isinstance(payload, dict) or "files" not in payload:
        return -1.0

    candidate_files: dict[str, str] = payload.get("files", {})
    if not candidate_files:
        return -1.0

    # Lock Cargo.toml: always use the incumbent's version regardless of what the
    # model produces.  Models hallucinate crate versions that don't exist on crates.io;
    # we pre-seed the incumbent with the crates the model is allowed to use.
    candidate_files_filtered = {
        k: v for k, v in candidate_files.items()
        if k.replace("\\", "/").lstrip("/") not in ("Cargo.toml", "Cargo.lock")
    }
    merged = {**incumbent_files, **{
        k.replace("\\", "/").lstrip("/"): v
        for k, v in candidate_files_filtered.items()
        if isinstance(v, str)
    }}

    with tempfile.TemporaryDirectory(prefix="vdb_rl_") as tmpdir:
        workspace = Path(tmpdir) / "workspace"

        # Always start from a fresh skeleton copy, then graft in the pre-built
        # target/ from the warm workspace.  This avoids the cp -al cross-filesystem
        # hardlink failure (Modal Volume → /tmp tmpfs are different filesystems).
        # A regular cp -r of target/ is slower than hardlinks but still saves the
        # full dep-compilation cost (~2-3 min → ~20s incremental build).
        shutil.copytree(skeleton_dir, workspace)
        if warm_ws_dir is not None and (warm_ws_dir / "target").exists():
            result = subprocess.run(
                ["cp", "-r", str(warm_ws_dir / "target"), str(workspace / "target")],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                print(f"[compile] warm target copy failed: {result.stderr[:200]}", flush=True)

        for rel, content in merged.items():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=build_timeout,
        )
        if result.returncode != 0:
            err_snippet = (result.stderr or "").strip()[:500]
            print(f"[compile] FAIL stderr: {err_snippet}", flush=True)
            return 0.0
        return 1.0


def _evaluate_qps(
    completion: str,
    *,
    skeleton_dir: Path,
    benchmark_dir: Path,
    data_dir: Path,
    incumbent_files: dict[str, str],
    baseline_qps: float,
    recall_threshold: float = 0.95,
    build_timeout: int = 300,
    bench_timeout: int = 300,
) -> float:
    """Full compile + server + benchmark reward.
    -1.0  invalid JSON / no files
     0.0  build fails
     0.1  builds + recall < threshold
     qps / baseline_qps  if fully valid
    """
    import json
    import re
    import shutil
    import socket
    import subprocess
    import tempfile
    import time

    cleaned = _strip_fences(completion)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return -1.0

    if not isinstance(payload, dict) or "files" not in payload:
        return -1.0

    candidate_files: dict[str, str] = payload.get("files", {})
    if not candidate_files:
        return -1.0

    candidate_files_filtered = {
        k: v for k, v in candidate_files.items()
        if k.replace("\\", "/").lstrip("/") not in ("Cargo.toml", "Cargo.lock")
    }
    merged = {**incumbent_files, **{
        k.replace("\\", "/").lstrip("/"): v
        for k, v in candidate_files_filtered.items()
        if isinstance(v, str)
    }}

    with tempfile.TemporaryDirectory(prefix="vdb_rl_") as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        shutil.copytree(skeleton_dir, workspace)
        for rel, content in merged.items():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

        br = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=workspace, capture_output=True, text=True, timeout=build_timeout,
        )
        if br.returncode != 0:
            return 0.0

        bb = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=benchmark_dir, capture_output=True, text=True, timeout=build_timeout,
        )
        if bb.returncode != 0:
            return 0.0

        server_bin = workspace / "target" / "release" / "vector-db-skeleton"
        bench_bin  = benchmark_dir / "target" / "release" / "vector-db-benchmark"
        if not server_bin.is_file() or not bench_bin.is_file():
            return 0.0

        port = 18080
        server_proc = subprocess.Popen(
            [str(server_bin)], cwd=workspace, capture_output=True, text=True,
        )
        try:
            deadline = time.time() + 30
            ready = False
            while time.time() < deadline:
                with socket.socket() as s:
                    s.settimeout(0.3)
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        ready = True
                        break
                time.sleep(0.2)
                if server_proc.poll() is not None:
                    return 0.0
            if not ready:
                return 0.0

            be = subprocess.run(
                [
                    str(bench_bin),
                    "--server-url", f"http://127.0.0.1:{port}",
                    "--concurrency", "4", "--warmup", "100",
                    "--base-vectors",  str(data_dir / "base_vectors.json"),
                    "--query-vectors", str(data_dir / "query_vectors.json"),
                    "--ground-truth",  str(data_dir / "ground_truth.json"),
                    "--recall-threshold", str(recall_threshold),
                    "--seed", "42", "--max-queries", "2000",
                ],
                capture_output=True, text=True, timeout=bench_timeout,
            )
        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

        if be.returncode != 0:
            return 0.0

        try:
            raw = be.stdout.strip()
            data = json.loads(raw) if raw.startswith("{") else json.loads(
                re.search(r"(\{.*\})", raw, re.DOTALL).group(1)
            )
            benchmark   = data.get("benchmark", {})
            qps         = float(benchmark.get("qps", 0.0))
            recall_ok   = bool(benchmark.get("recall_passed", False))
            anti_cheat  = bool((data.get("anti_cheat") or {}).get("passed", False))
        except Exception:
            return 0.0

        if not recall_ok or not anti_cheat:
            return 0.1
        return max(qps / baseline_qps, 0.0)


# ---------------------------------------------------------------------------
# Cargo warm-up function — run once to pre-build all dependencies
# ---------------------------------------------------------------------------

@app.function(
    image=rl_image,
    gpu=None,                           # no GPU needed for compilation
    timeout=60 * 20,                    # 20 min cap for cold dep build
    volumes={
        WARM_WS_MOUNT:    warm_ws_volume,
        CARGO_CACHE_MOUNT: cargo_cache_volume,
    },
    memory=8192,
)
def warmup_cargo() -> bool:
    """Pre-build all Cargo dependencies into a warm workspace on a Modal Volume.

    Subsequent calls are no-ops (checks a sentinel file).  Returns True if a
    build was performed, False if the warm workspace was already present.
    """
    import shutil
    import subprocess

    warm_dir  = Path(WARM_WS_MOUNT) / "ws"
    sentinel  = Path(WARM_WS_MOUNT) / "built.sentinel"

    if sentinel.exists():
        print("[warmup] warm workspace already built — skipping", flush=True)
        return False

    print("[warmup] cold-building all Cargo dependencies...", flush=True)
    skeleton_dir = Path(REMOTE_BENCH) / "skeleton"
    if warm_dir.exists():
        shutil.rmtree(warm_dir)
    shutil.copytree(skeleton_dir, warm_dir)

    # Write the locked incumbent Cargo.toml (same crates the reward fn uses).
    (warm_dir / "Cargo.toml").write_text(INCUMBENT_CARGO_TOML, encoding="utf-8")

    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=warm_dir,
        capture_output=True,
        text=True,
        timeout=1000,
    )
    if result.returncode != 0:
        print(f"[warmup] FAILED:\n{result.stderr[:2000]}", flush=True)
        return False

    sentinel.write_text("done", encoding="utf-8")
    warm_ws_volume.commit()
    print("[warmup] done — warm workspace committed to volume", flush=True)
    return True


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=rl_image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 12,
    volumes={
        CHECKPOINT_MOUNT:    checkpoints_volume,
        CARGO_CACHE_MOUNT:   cargo_cache_volume,
        WARM_WS_MOUNT:       warm_ws_volume,
        "/checkpoints-sft":  modal.Volume.from_name("vdb-sft-checkpoints", create_if_missing=True),
    },
    memory=32768,
)
def train_grpo(
    prompts_jsonl: str,
    *,
    model_id: str = MODEL_ID,
    sft_adapter_name: str = "",         # load SFT LoRA adapter as warm-start
    reward_mode: str = "compile",
    baseline_qps: float = 100.0,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    num_generations: int = 2,
    max_new_tokens: int = 2048,
    max_prompt_length: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    num_train_steps: int = 200,
    output_name: str = "qwen25-coder-vdb-rl",
) -> dict:
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as P

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    bench         = P(REMOTE_BENCH)
    skeleton_dir  = bench / "skeleton"
    benchmark_dir = bench / "benchmark"
    data_dir      = P(REMOTE_DATA)
    warm_ws_dir   = P(WARM_WS_MOUNT) / "ws"   # pre-built dep artifacts

    # Ensure warm workspace exists; build it now if this is the first run.
    _sentinel = P(WARM_WS_MOUNT) / "built.sentinel"
    if not _sentinel.exists():
        print("[grpo] warm workspace not found — triggering warmup_cargo...", flush=True)
        warmup_cargo.remote()
        warm_ws_volume.reload()
    if (warm_ws_dir / "target").exists():
        print(f"[grpo] warm workspace ready at {warm_ws_dir}", flush=True)
    else:
        print("[grpo] WARNING: warm workspace missing target/, falling back to cold builds", flush=True)
        warm_ws_dir = None

    # Seeded baseline incumbent files (matches SEEDED_BASELINE_FILES in harness).
    # We pre-seed rayon and parking_lot so the model can write parallel / low-lock
    # implementations without needing to discover crate versions on its own.
    # hnsw_rs / instant-distance are intentionally omitted — the model often
    # hallucates wrong semver; implementations should be hand-rolled in pure Rust.
    # Use the module-level constant so incumbent Cargo.toml matches the warm workspace.
    incumbent_files: dict[str, str] = {
        "Cargo.toml": INCUMBENT_CARGO_TOML,
        "src/db.rs": """\
use crate::api::*;
use crate::distance::l2_distance;
use std::cmp::Ordering;
use std::sync::RwLock;

const VECTOR_DIM: usize = 128;

struct Storage { ids: Vec<u64>, vectors: Vec<f32> }

pub struct VectorDB { storage: RwLock<Storage> }

impl VectorDB {
    pub fn new() -> Self {
        Self { storage: RwLock::new(Storage { ids: Vec::new(), vectors: Vec::new() }) }
    }
    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != VECTOR_DIM { return; }
        let mut s = self.storage.write().unwrap();
        s.ids.push(id); s.vectors.extend_from_slice(&vector);
    }
    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut s = self.storage.write().unwrap();
        let mut n = 0usize;
        for (id, v) in vectors {
            if v.len() != VECTOR_DIM { continue; }
            s.ids.push(id); s.vectors.extend_from_slice(&v); n += 1;
        }
        n
    }
    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        if query.len() != VECTOR_DIM || top_k == 0 { return Vec::new(); }
        let s = self.storage.read().unwrap();
        if s.ids.is_empty() { return Vec::new(); }
        let mut scored: Vec<(u64, f64)> = s.ids.iter().copied().enumerate()
            .map(|(i, id)| (id, l2_distance(query, &s.vectors[i*VECTOR_DIM..(i+1)*VECTOR_DIM])))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(top_k as usize)
            .map(|(id, distance)| SearchResult { id, distance }).collect()
    }
}
""",
        "src/distance.rs": """\
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| { let d = (*x as f64)-(*y as f64); d*d }).sum()
}
""",
    }

    # ── Dataset ──────────────────────────────────────────────────────────────
    records = [json.loads(l) for l in prompts_jsonl.splitlines() if l.strip()]
    if not records:
        raise ValueError("prompts_jsonl is empty — run generate_rl_prompts.py first")
    dataset = Dataset.from_list(records)
    print(f"[grpo] {len(dataset)} prompts, reward_mode={reward_mode}", flush=True)

    # ── Reward function ───────────────────────────────────────────────────────
    _max_parallel = 2  # keep low to avoid cargo registry lock contention

    def reward_fn(completions: list, **kwargs) -> list[float]:
        # TRL 1.0 may pass completions as list[str] or list[list[dict]] depending
        # on the chat template application. Normalise to plain strings.
        def _to_str(c) -> str:
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                # Chat-format: extract the last assistant message content.
                for msg in reversed(c):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return str(msg.get("content", ""))
                # Fallback: join all content fields
                return " ".join(str(m.get("content", m)) for m in c if isinstance(m, dict))
            return str(c)

        # Debug: log format on first call only.
        if completions:
            print(
                f"[reward] completion type={type(completions[0]).__name__} "
                f"preview={str(completions[0])[:120]}",
                flush=True,
            )

        str_completions = [_to_str(c) for c in completions]

        if reward_mode == "compile":
            fn = lambda c: _evaluate_compile(
                c, skeleton_dir=skeleton_dir, incumbent_files=incumbent_files,
                warm_ws_dir=warm_ws_dir,
            )
        else:
            fn = lambda c: _evaluate_qps(
                c, skeleton_dir=skeleton_dir, benchmark_dir=benchmark_dir,
                data_dir=data_dir, incumbent_files=incumbent_files,
                baseline_qps=baseline_qps,
            )
        with ThreadPoolExecutor(max_workers=min(len(str_completions), _max_parallel)) as ex:
            rewards = list(ex.map(fn, str_completions))
        compile_ok = sum(1 for r in rewards if r > 0)
        print(
            f"[reward] n={len(rewards)} compile_ok={compile_ok}"
            f"  {[round(r, 2) for r in rewards]}",
            flush=True,
        )
        return rewards

    # ── Model + QLoRA ─────────────────────────────────────────────────────────
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # Optional SFT warm-start: load a pre-trained LoRA adapter as the starting
    # point for RL.  The adapter must exist in the vdb-sft-checkpoints volume.
    SFT_CKPT_MOUNT = "/checkpoints-sft"
    if sft_adapter_name:
        sft_adapter_path = f"{SFT_CKPT_MOUNT}/{sft_adapter_name}"
        if P(sft_adapter_path).exists():
            print(f"[grpo] loading SFT adapter from {sft_adapter_path}", flush=True)
            model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
        else:
            print(f"[grpo] WARNING: SFT adapter not found at {sft_adapter_path}, ignoring", flush=True)
            sft_adapter_name = ""

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Only add a new LoRA config if we didn't load a pre-trained adapter.
    lora_config = None if sft_adapter_name else LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules="all-linear", lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
    )

    # ── GRPO config ───────────────────────────────────────────────────────────
    output_dir = f"{CHECKPOINT_MOUNT}/{output_name}"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=num_train_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        num_generations=num_generations,
        max_completion_length=max_new_tokens,   # renamed in TRL 1.0
        temperature=0.9,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("[grpo] starting training", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    checkpoints_volume.commit()

    result = {
        "output_dir": output_dir,
        "model_id": model_id,
        "reward_mode": reward_mode,
        "num_prompts": len(dataset),
        "num_train_steps": num_train_steps,
    }
    print(f"[grpo] done: {result}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    prompts_path: str = "",
    model_id: str = MODEL_ID,
    sft_adapter_name: str = "",
    reward_mode: str = "compile",
    baseline_qps: float = 100.0,
    lora_rank: int = 64,
    num_generations: int = 2,
    num_train_steps: int = 200,
    output_name: str = "qwen25-coder-vdb-rl",
    warmup_only: bool = False,
) -> None:
    """
    Run with --warmup-only to pre-build the Cargo workspace without training.
    This is a one-time setup step that takes ~5-10 min and speeds up all
    subsequent reward evaluations from 3-5 min to ~20 sec.

      modal run scripts/vector_db_bench/modal_rl_train.py --warmup-only
    """
    import json

    if warmup_only:
        print("[main] running warmup_cargo only...")
        built = warmup_cargo.remote()
        print(f"[main] warmup done (was_new={built})")
        return

    path = Path(prompts_path)
    if not path.exists():
        raise SystemExit(
            f"Prompts file not found: {path}\n"
            "Run: python scripts/vector_db_bench/generate_rl_prompts.py --bench-repo ..."
        )

    if reward_mode == "qps" and not _data_path.is_dir():
        raise SystemExit(
            "qps reward mode requires benchmark data.\n"
            "Set VECTOR_DB_BENCH_DATA or ensure data/ exists inside VECTOR_DB_BENCH_ROOT."
        )

    prompts_jsonl = path.read_text(encoding="utf-8")
    n = sum(1 for l in prompts_jsonl.splitlines() if l.strip())
    print(f"[main] {n} prompts, reward_mode={reward_mode}, gpu={GPU_TYPE}")

    result = train_grpo.remote(
        prompts_jsonl,
        model_id=model_id,
        sft_adapter_name=sft_adapter_name,
        reward_mode=reward_mode,
        baseline_qps=baseline_qps,
        lora_rank=lora_rank,
        num_generations=num_generations,
        num_train_steps=num_train_steps,
        output_name=output_name,
    )
    print(f"[main] done:\n{json.dumps(result, indent=2)}")
    print(f"\nRetrieve adapter:")
    print(f"  modal volume get {CHECKPOINT_VOLUME} {result['output_dir'].lstrip('/')} ./local-rl-adapter/")
