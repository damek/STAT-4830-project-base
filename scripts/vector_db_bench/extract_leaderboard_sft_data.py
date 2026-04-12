#!/usr/bin/env python3
"""Extract SFT training data from leaderboard submissions.

Each leaderboard entry contains a working Rust implementation that was benchmarked
and scored. We turn every entry above a QPS threshold into a (prompt, completion)
SFT pair so the model can learn from proven strategies (IVF, HNSW, SIMD, rayon).

Output format is identical to modal_sft_rollouts.py — JSONL with one record per
example, each containing a `messages` list with user + assistant turns.

Usage:
  python scripts/vector_db_bench/extract_leaderboard_sft_data.py \\
      --bench-repo third_party/vector-db-bench \\
      --min-qps 100 \\
      --output data/vector_db_bench/leaderboard_sft_data.jsonl

  # Merge with rollout data:
  cat data/vector_db_bench/leaderboard_sft_data.jsonl \\
      data/vector_db_bench/sft_data.jsonl \\
      > data/vector_db_bench/combined_sft_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "vector_db_bench" / "leaderboard_sft_data.jsonl"

# Files we expose to the model as editable (matches our RL setup).
EDITABLE_EXTS = {".rs", ".toml"}
PROTECTED = {"src/api.rs", "src/main.rs"}

# The locked incumbent Cargo.toml we use in RL training.
# Model completions replace individual .rs files on top of this.
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

BASELINE_QPS = 55.0
SOTA_QPS = 3548.0


def _build_prompt(incumbent_files: dict[str, str], entry_qps: float, entry_recall: float) -> str:
    """Build the user-turn prompt for an SFT example."""
    editable_sorted = sorted(incumbent_files.keys())
    mutable_text = "\n\n".join(
        f"## {rel}\n```{'toml' if rel.endswith('.toml') else 'rust'}\n{content.rstrip()}\n```"
        for rel, content in sorted(incumbent_files.items())
    )
    return (
        "You are one worker in a local Codex CLI optimization harness for vector-db-bench.\n"
        "Return a JSON object with keys 'summary' and 'files'.\n"
        f"The 'files' object must include full updated contents for EVERY current editable path "
        f"(listed below: {', '.join(editable_sorted)}).\n"
        "You may ADD new editable Rust modules under src/ (e.g. src/ivf.rs) by including extra keys in 'files'; "
        "new files must be wired from db.rs/Cargo.toml as needed and must stay on the allowed paths "
        "(src/**/*.rs, Cargo.toml, build.rs only).\n"
        "Do not change protected API/server files. Preserve recall >= 0.95 and anti-cheat.\n"
        "Your code must compile in Rust release mode.\n\n"
        "# Performance Baseline\n\n"
        f"Incumbent implementation (brute-force flat scan):\n"
        f"- QPS: {BASELINE_QPS:.0f} queries/second\n"
        f"- Recall: 0.999 (well above the 0.95 threshold)\n\n"
        f"Leaderboard leader (single-turn, no agentic loop):\n"
        f"- QPS: {SOTA_QPS:.0f} queries/second\n"
        f"- Recall: 0.959\n\n"
        f"Gap: {SOTA_QPS / BASELINE_QPS:.0f}x improvement over incumbent.\n"
        "This gap comes from indexing (IVF, HNSW), parallelism (rayon), and SIMD distance kernels.\n"
        "Strategies that trade a small amount of recall for large QPS gains are the right direction.\n\n"
        "Improve the implementation to maximize QPS while maintaining recall >= 0.95.\n\n"
        f"Current mutable files:\n{mutable_text}\n\n"
        "Return only JSON."
    )


def _load_src_files(entry_dir: Path, prefer_best_qps: bool = True) -> dict[str, str] | None:
    """Load editable source files from a leaderboard entry directory.

    Returns a dict of rel_path -> content, or None if the directory is invalid.
    Prefers src_best_qps/ over src/ when available.
    """
    src_dir = entry_dir / ("src_best_qps" if prefer_best_qps and (entry_dir / "src_best_qps").is_dir() else "src")
    if not src_dir.is_dir():
        src_dir = entry_dir / "src"
    if not src_dir.is_dir():
        return None

    files: dict[str, str] = {}
    for f in sorted(src_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f"src/{f.relative_to(src_dir).as_posix()}"
        if rel in PROTECTED:
            continue
        if f.suffix not in EDITABLE_EXTS:
            continue
        files[rel] = f.read_text(encoding="utf-8")

    if not files:
        return None
    return files


def _infer_strategy(files: dict[str, str], entry_name: str) -> str:
    """Heuristically describe the main strategy used by an entry."""
    db_content = files.get("src/db.rs", "").lower()
    tags = []
    if "hnsw" in db_content or "hnsw" in entry_name.lower():
        tags.append("HNSW graph index")
    if "ivf" in db_content or "centroid" in db_content or "cluster" in db_content:
        tags.append("IVF clustering")
    if "avx" in db_content or "simd" in db_content or "mm256" in db_content:
        tags.append("AVX2/SIMD distance")
    if "par_iter" in db_content or "rayon" in db_content:
        tags.append("rayon parallelism")
    if "parking_lot" in db_content:
        tags.append("parking_lot RwLock")
    if "quantiz" in db_content or "u8" in db_content:
        tags.append("quantization")
    return ", ".join(tags) if tags else "optimized implementation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SFT data from leaderboard submissions")
    parser.add_argument(
        "--bench-repo", type=Path, required=True,
        help="Path to vector-db-bench repo (contains leaderboard/ and results/)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-qps", type=float, default=100.0,
        help="Only include entries with QPS >= this threshold (default: 100).",
    )
    parser.add_argument(
        "--include-all", action="store_true",
        help="Include all entries regardless of QPS (useful for compile-only SFT).",
    )
    args = parser.parse_args()

    bench_repo = args.bench_repo.resolve()
    leaderboard_dir = bench_repo / "leaderboard"
    results_file = bench_repo / "results" / "leaderboard.json"

    if not leaderboard_dir.is_dir():
        print(f"[extract] leaderboard dir not found: {leaderboard_dir}", file=sys.stderr)
        sys.exit(1)
    if not results_file.is_file():
        print(f"[extract] results file not found: {results_file}", file=sys.stderr)
        sys.exit(1)

    results = json.loads(results_file.read_text(encoding="utf-8"))
    # Build lookup: entry_dir -> {qps, recall}
    result_by_dir: dict[str, dict] = {}
    for r in results:
        entry_dir = r.get("entry_dir", "")
        if entry_dir and r.get("qps", 0) > result_by_dir.get(entry_dir, {}).get("qps", -1):
            result_by_dir[entry_dir] = r

    incumbent_files = {
        "Cargo.toml":      INCUMBENT_CARGO_TOML,
        "src/db.rs":       INCUMBENT_DB_RS,
        "src/distance.rs": INCUMBENT_DISTANCE_RS,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_qps = skipped_empty = 0

    with args.output.open("w", encoding="utf-8") as out:
        for entry_name in sorted(leaderboard_dir.iterdir()):
            if not entry_name.is_dir():
                continue
            name = entry_name.name
            meta = result_by_dir.get(name, {})
            qps = meta.get("qps", 0.0)
            recall = meta.get("recall", 0.0)

            if not args.include_all and qps < args.min_qps:
                skipped_qps += 1
                continue

            src_files = _load_src_files(entry_name)
            if src_files is None:
                skipped_empty += 1
                continue

            strategy = _infer_strategy(src_files, name)
            summary = (
                f"Implementation from {name} achieving {qps:.0f} QPS at recall={recall:.3f}. "
                f"Key strategies: {strategy}."
            )

            # Completion: JSON with summary + all editable files from this entry.
            completion_files = dict(src_files)  # src/*.rs files from the entry
            # Don't include Cargo.toml in completion — we lock it in training.
            completion_obj = {"summary": summary, "files": completion_files}
            completion = json.dumps(completion_obj, ensure_ascii=False)

            prompt = _build_prompt(incumbent_files, qps, recall)
            record = {
                "messages": [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "entry_name": name,
                "qps": qps,
                "recall": recall,
                "strategy": strategy,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{written:2d}] {name:45s}  QPS={qps:>8.0f}  {strategy}")

    print(f"\n[extract] wrote {written} examples → {args.output}")
    print(f"[extract] skipped {skipped_qps} below min-qps={args.min_qps:.0f}, {skipped_empty} with no src")


if __name__ == "__main__":
    main()
