#!/usr/bin/env python3
"""Generate compile-clean SFT rollouts from Qwen2.5-Coder on vector-db-bench prompts.

Runs Qwen in inference-only mode (no gradients) on the RL prompt set, evaluates
each completion with the compile reward, and saves compile-successful (prompt,
completion) pairs as chat-format JSONL suitable for modal_sft_train.py.

This is the "C" part of the A+C efficiency plan:
  A = warm cargo workspace (fast incremental builds in modal_rl_train.py)
  C = SFT warm-start — fine-tune on compile-clean rollouts, then RL from adapter

Workflow:
  # 1. Pre-warm cargo (one-time, ~5-10 min):
  export VECTOR_DB_BENCH_ROOT=/path/to/vector-db-bench
  modal run scripts/vector_db_bench/modal_rl_train.py --warmup-only

  # 2. Generate SFT data (A10G, ~2-4 hr for 8 rollouts/prompt):
  modal run scripts/vector_db_bench/modal_sft_rollouts.py \\
      --prompts-path data/vector_db_bench/rl_prompts.jsonl \\
      --rollouts-per-prompt 8

  # 3. SFT train (A10G, ~30-60 min):
  modal run scripts/vector_db_bench/modal_sft_train.py \\
      --dataset-path data/vector_db_bench/sft_data.jsonl

  # 4. RL from SFT checkpoint:
  modal run scripts/vector_db_bench/modal_rl_train.py \\
      --prompts-path data/vector_db_bench/rl_prompts.jsonl \\
      --sft-adapter-name qwen25-coder-vdb-sft
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Shared constants — must match modal_rl_train.py
# ---------------------------------------------------------------------------

MODEL_ID           = "Qwen/Qwen2.5-Coder-7B-Instruct"
CARGO_CACHE_VOLUME = "vdb-cargo-cache"
WARM_WS_VOLUME     = "vdb-warm-workspace"
CARGO_CACHE_MOUNT  = "/root/.cargo/registry"
WARM_WS_MOUNT      = "/opt/vdb-warm"
REMOTE_BENCH       = "/opt/vector-db-bench"

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

INCUMBENT_DB_RS = """\
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
"""

INCUMBENT_DISTANCE_RS = """\
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| { let d = (*x as f64)-(*y as f64); d*d }).sum()
}
"""

# ---------------------------------------------------------------------------
# Modal image + app
# ---------------------------------------------------------------------------

def _bench_ignore(p: Path) -> bool:
    skip = {".git", "target", ".idea", ".vscode", "data"}
    return any(part in skip for part in p.parts)


if modal.is_local():
    _bench_root = os.environ.get("VECTOR_DB_BENCH_ROOT", "").strip()
    if not _bench_root:
        raise RuntimeError("Set VECTOR_DB_BENCH_ROOT to your vector-db-bench clone.")
    _bench_path = Path(_bench_root).resolve()

    rollout_image = (
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
            "accelerate>=1.0.0",
            "bitsandbytes>=0.44.1",
        )
        .add_local_dir(
            str(_bench_path),
            remote_path=REMOTE_BENCH,
            copy=True,
            ignore=_bench_ignore,
        )
    )
else:
    rollout_image = modal.Image.debian_slim(python_version="3.11")

rollout_app        = modal.App("alpha-vdb-sft-rollouts")
cargo_cache_volume = modal.Volume.from_name(CARGO_CACHE_VOLUME, create_if_missing=True)
warm_ws_volume     = modal.Volume.from_name(WARM_WS_VOLUME,     create_if_missing=True)

# ---------------------------------------------------------------------------
# Helpers (duplicated from modal_rl_train so this file is self-contained)
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    import re
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
    return t.strip()


def _evaluate_compile_local(
    completion: str,
    *,
    skeleton_dir: "Path",
    incumbent_files: dict,
    warm_ws_dir: "Path | None",
    build_timeout: int = 120,
) -> float:
    import json, re, shutil, subprocess, tempfile
    cleaned = _strip_fences(completion)
    try:
        payload = json.loads(cleaned)
    except Exception:
        return -1.0
    if not isinstance(payload, dict) or "files" not in payload:
        return -1.0
    candidate_files = payload.get("files", {})
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
    with tempfile.TemporaryDirectory(prefix="vdb_sft_") as tmpdir:
        workspace = Path(tmpdir) / "ws"
        if warm_ws_dir is not None and (warm_ws_dir / "target").exists():
            r = subprocess.run(["cp", "-al", str(warm_ws_dir), str(workspace)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                # cp -al may have partially created the directory; clean up before fallback.
                if workspace.exists():
                    shutil.rmtree(workspace)
                shutil.copytree(skeleton_dir, workspace)
        else:
            shutil.copytree(skeleton_dir, workspace)
        for rel, content in merged.items():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content if content.endswith("\n") else content + "\n",
                            encoding="utf-8")
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=workspace, capture_output=True, text=True, timeout=build_timeout,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip()[:300]
            print(f"[compile] FAIL: {err}", flush=True)
            return 0.0
        return 1.0


# ---------------------------------------------------------------------------
# Main rollout generation function
# ---------------------------------------------------------------------------

@rollout_app.function(
    image=rollout_image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={
        CARGO_CACHE_MOUNT: cargo_cache_volume,
        WARM_WS_MOUNT:     warm_ws_volume,
    },
    memory=32768,
)
def generate_rollouts(
    prompts_jsonl: str,
    *,
    model_id: str = MODEL_ID,
    rollouts_per_prompt: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_p: float = 0.95,
    build_timeout: int = 120,
) -> str:
    """Generate rollouts, evaluate compile reward, return compile-clean JSONL."""
    import json
    from pathlib import Path as P

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bench        = P(REMOTE_BENCH)
    skeleton_dir = bench / "skeleton"
    warm_ws_dir  = P(WARM_WS_MOUNT) / "ws"
    if not (warm_ws_dir / "target").exists():
        print("[rollouts] warm workspace not found — builds will be slow (~3-5 min each)", flush=True)
        warm_ws_dir = None
    else:
        print(f"[rollouts] warm workspace ready — incremental builds (~20s each)", flush=True)

    incumbent_files = {
        "Cargo.toml":       INCUMBENT_CARGO_TOML,
        "src/db.rs":        INCUMBENT_DB_RS,
        "src/distance.rs":  INCUMBENT_DISTANCE_RS,
    }

    # ── Load model in bfloat16 (no QLoRA — inference only) ───────────────────
    print(f"[rollouts] loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("[rollouts] model loaded", flush=True)

    # ── Rollout loop ──────────────────────────────────────────────────────────
    records    = [json.loads(l) for l in prompts_jsonl.splitlines() if l.strip()]
    sft_lines: list[str] = []
    total_tried = total_ok = 0

    for rec_idx, record in enumerate(records):
        prompt_messages = record["prompt"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        for attempt in range(rollouts_per_prompt):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids    = output_ids[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_ids, skip_special_tokens=True)

            total_tried += 1
            reward = _evaluate_compile_local(
                completion,
                skeleton_dir=skeleton_dir,
                incumbent_files=incumbent_files,
                warm_ws_dir=warm_ws_dir,
                build_timeout=build_timeout,
            )

            tag = "✓COMPILE" if reward > 0 else ("✗JSON" if reward < 0 else "✗BUILD")
            print(
                f"[rollouts] prompt {rec_idx+1}/{len(records)} "
                f"attempt {attempt+1}/{rollouts_per_prompt} "
                f"reward={reward:.1f} {tag}",
                flush=True,
            )

            if reward > 0:
                total_ok += 1
                clean = _strip_fences(completion)
                sft_lines.append(json.dumps({
                    "messages": [
                        {"role": "user",      "content": prompt_messages[-1]["content"]},
                        {"role": "assistant", "content": clean},
                    ]
                }, ensure_ascii=False))

    rate = total_ok / max(total_tried, 1) * 100
    print(f"[rollouts] finished — {total_ok}/{total_tried} compiled ({rate:.1f}%)", flush=True)
    return "\n".join(sft_lines)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@rollout_app.local_entrypoint()
def main(
    prompts_path: str,
    output_path: str = "data/vector_db_bench/sft_data.jsonl",
    model_id: str = MODEL_ID,
    rollouts_per_prompt: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
) -> None:
    path = Path(prompts_path)
    if not path.exists():
        raise SystemExit(f"Prompts file not found: {path}")

    prompts_jsonl = path.read_text(encoding="utf-8")
    n = sum(1 for l in prompts_jsonl.splitlines() if l.strip())
    print(f"[main] {rollouts_per_prompt} rollouts × {n} prompts = {rollouts_per_prompt * n} total")

    sft_jsonl = generate_rollouts.remote(
        prompts_jsonl,
        model_id=model_id,
        rollouts_per_prompt=rollouts_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    num = sum(1 for l in sft_jsonl.splitlines() if l.strip())
    out.write_text(sft_jsonl, encoding="utf-8")
    print(f"[main] {num} compile-clean examples → {out}")
    if num == 0:
        print("[main] 0 examples — run --warmup-only first or increase --rollouts-per-prompt")
