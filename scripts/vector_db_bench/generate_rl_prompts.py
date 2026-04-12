#!/usr/bin/env python3
"""Generate a JSONL prompt dataset for GRPO RL training on vector-db-bench.

Produces one prompt per optimization brief (20 diverse strategies).
Each prompt is the exact worker prompt the harness uses, formatted as
a single-turn chat message suitable for TRL GRPOTrainer.

Usage:
  python scripts/vector_db_bench/generate_rl_prompts.py \\
      --bench-repo /path/to/vector-db-bench \\
      --output data/vector_db_bench/rl_prompts.jsonl

  # Optional: use a specific incumbent instead of the seeded baseline
  python scripts/vector_db_bench/generate_rl_prompts.py \\
      --bench-repo /path/to/vector-db-bench \\
      --incumbent-dir data/vector_db_bench/codex_cli_runs/my_run/initial_candidate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import shared prompt construction + data structures from the harness.
sys.path.insert(0, str(Path(__file__).parent))
from codex_cli_harness import (
    DEFAULT_MEMORY_TEMPLATE,
    DEFAULT_REVIEWER_NOTES,
    SEEDED_BASELINE_FILES,
    WorkerBrief,
    _bootstrap_seed_surface,
    _read_readonly_context,
    _worker_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "vector_db_bench" / "rl_prompts.jsonl"
DEFAULT_PROGRAM = Path(__file__).with_name("program.md")

# ---------------------------------------------------------------------------
# Diverse optimization briefs — one prompt per brief
# ---------------------------------------------------------------------------
# These cover the main ANN / systems optimization families.
# Duplicate families with different hypotheses give the model multiple angles.

# ---------------------------------------------------------------------------
# NOTE ON DEPENDENCIES
# --------------------
# The Cargo.toml already includes: axum, tokio, serde, serde_json, rayon,
# parking_lot, crossbeam.  Do NOT instruct the model to add new crate
# dependencies — it tends to hallucinate versions that don't exist on crates.io.
# All briefs should work with only what's pre-seeded.
# ---------------------------------------------------------------------------

_BRIEFS: list[WorkerBrief] = [
    # ── Indexing ────────────────────────────────────────────────────────────
    WorkerBrief(
        title="IVF-Flat with K-Means",
        family="indexing",
        hypothesis="Partitioning vectors into clusters reduces the brute-force search space from N to ~N/k per query.",
        instructions=(
            "Implement an Inverted File Index in src/db.rs. "
            "During bulk_insert, run k-means with k=256 centroids (random init, 20 iterations). "
            "Assign each vector to its nearest centroid. "
            "At query time, probe the top 8 closest centroids and do exact L2 over their members. "
            "Fall back to full brute-force if fewer than 256 vectors have been inserted. "
            "Preserve the existing HTTP API — only src/db.rs is the target. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="IVF-Flat with higher nprobe",
        family="indexing",
        hypothesis="More cluster probes trade compute for better recall; 256 clusters, nprobe=16 is a safe starting point.",
        instructions=(
            "Implement IVF with k=256 centroids and nprobe=16. "
            "Use a flat (exact L2) distance scan within each probed cluster. "
            "Sort centroids by distance to query in O(k) time, then probe top-nprobe. "
            "Store centroid assignments in a Vec<Vec<usize>> inverted list. "
            "rayon is already in Cargo.toml — use par_iter for the within-cluster scan. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="HNSW Graph Index (pure Rust)",
        family="indexing",
        hypothesis="HNSW provides O(log N) approximate nearest-neighbor search with high recall.",
        instructions=(
            "Implement HNSW from scratch in src/hnsw.rs — do NOT add external crates. "
            "Parameters: M=16, ef_construction=200, ef_search=80. "
            "Use a BinaryHeap for the priority queue and a Vec<Vec<(u32, f32)>> for adjacency lists. "
            "Wire the HNSW index from src/db.rs: insert into HNSW on bulk_insert, search via HNSW on query. "
            "Keep a Vec<u64> of ids so HNSW node indices map back to API vector IDs. "
            "rayon is available for parallel distance computations. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs", "src/hnsw.rs"),
    ),
    WorkerBrief(
        title="IVF + Scalar Quantization with Reranking",
        family="indexing",
        hypothesis="Compressing stored vectors to u8 speeds coarse search; reranking the top candidates with f32 restores recall.",
        instructions=(
            "In src/db.rs: "
            "(1) On insert, quantize each f32 vector to u8 per-dimension (store min/scale per dimension). "
            "(2) Use IVF with k=256 clusters on the quantized vectors. "
            "(3) At query time, probe top-16 clusters using u8 distances. "
            "(4) Rerank the top-500 approximate candidates with exact f32 L2. "
            "Add a src/quant.rs module for encode/decode helpers. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs", "src/quant.rs"),
    ),
    # ── Distance kernels ─────────────────────────────────────────────────────
    WorkerBrief(
        title="SIMD AVX2 L2 Distance Kernel",
        family="distance",
        hypothesis="Explicit AVX2 vectorization processes 8 floats per cycle vs 1 in scalar, yielding 4-8x distance throughput.",
        instructions=(
            "Rewrite l2_distance in src/distance.rs using std::arch::x86_64 AVX2 + FMA intrinsics. "
            "Process 8 f32 values per loop iteration (_mm256_sub_ps, _mm256_fmadd_ps). "
            "Handle the remaining tail (<8) with scalar code. "
            "Add build.rs at the skeleton root to emit cargo:rustc-flags=-C target-feature=+avx2,+fma. "
            "Wrap intrinsic code in #[target_feature(enable=\"avx2,fma\")] unsafe fn."
        ),
        target_files=("src/distance.rs", "Cargo.toml"),
    ),
    WorkerBrief(
        title="Auto-vectorized L2 with explicit loops",
        family="distance",
        hypothesis="LLVM auto-vectorizes simple loop patterns; restructuring the iterator into a for loop over chunks can unlock wider SIMD.",
        instructions=(
            "Replace the iterator chain in l2_distance with a manual loop over 8-element chunks. "
            "Accumulate into a [f32; 8] partial-sum array to encourage SIMD widening. "
            "Sum the partial array at the end. Return f64. "
            "Also add #[inline(always)] to l2_distance and any hot distance helpers. "
            "In Cargo.toml [profile.release] set opt-level=3 and target-cpu=native via RUSTFLAGS."
        ),
        target_files=("src/distance.rs", "Cargo.toml"),
    ),
    # ── Parallelism ──────────────────────────────────────────────────────────
    WorkerBrief(
        title="Rayon Parallel Brute-Force Search",
        family="parallelism",
        hypothesis="Rayon data parallelism amortizes memory bandwidth across all cores, scaling QPS with thread count.",
        instructions=(
            "rayon is already in Cargo.toml — do not add it again. "
            "In the search method in src/db.rs, replace the sequential iterator with par_chunks(VECTOR_DIM) "
            "to compute per-vector distances in parallel. "
            "Collect results, sort, and take top_k. "
            "Use a global Rayon thread pool configured via rayon::ThreadPoolBuilder::new().build_global(). "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="Concurrent Read via parking_lot RwLock + Rayon",
        family="parallelism",
        hypothesis="parking_lot::RwLock is faster than std RwLock; holding it only for the clone and then computing distances unlocked removes the serialisation bottleneck.",
        instructions=(
            "Replace std::sync::RwLock with parking_lot::RwLock (parking_lot is already in Cargo.toml). "
            "Snapshot the storage (clone ids + Arc<Vec<f32>>) before computing distances, "
            "so the lock is not held during the distance loop. "
            "Use rayon par_iter on the snapshot for parallel distance computation. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    # ── Hybrid ───────────────────────────────────────────────────────────────
    WorkerBrief(
        title="IVF + Rayon parallel cluster probing",
        family="hybrid",
        hypothesis="Combining IVF cluster pruning with rayon parallelism over probed clusters maximises both algorithmic and hardware efficiency.",
        instructions=(
            "Implement IVF (k=256, nprobe=8) in src/db.rs. "
            "At query time, sort centroids by distance, then use rayon::par_iter over the top-nprobe clusters, "
            "computing exact L2 within each cluster on a separate thread. "
            "Merge thread-local top-k heaps into a global result. "
            "rayon is already in Cargo.toml. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="IVF + AVX2 distance kernel",
        family="hybrid",
        hypothesis="IVF shrinks the candidate set; AVX2 speeds each distance — combining them multiplies the throughput gain.",
        instructions=(
            "Implement IVF (k=128, nprobe=16) in src/db.rs. "
            "Replace the inner distance call with an AVX2 SIMD kernel in src/distance.rs "
            "(use std::arch::x86_64 intrinsics, #[target_feature(enable=\"avx2,fma\")]). "
            "Add build.rs to pass -C target-feature=+avx2,+fma via cargo:rustc-flags. "
            "rayon is already in Cargo.toml — use it for parallel centroid-distance computation during k-means. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs", "src/distance.rs", "build.rs"),
    ),
    WorkerBrief(
        title="HNSW (pure Rust) + Rayon parallel searches",
        family="hybrid",
        hypothesis="Running multiple HNSW searches in parallel on separate query threads saturates all cores.",
        instructions=(
            "Implement HNSW from scratch in src/hnsw.rs (do NOT add external crates). "
            "Use M=16, ef_construction=200, ef_search=50. "
            "Build the index on bulk_insert. "
            "Wrap the HNSW index in Arc<parking_lot::RwLock<...>> so multiple threads can search concurrently "
            "(parking_lot is already in Cargo.toml). "
            "rayon is available for parallel candidate scoring. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs", "src/hnsw.rs"),
    ),
    # ── Memory layout ────────────────────────────────────────────────────────
    WorkerBrief(
        title="Cache-aligned SoA with prefetch",
        family="memory",
        hypothesis="32-byte aligned vector storage eliminates unaligned-load penalties and allows software prefetch to hide DRAM latency.",
        instructions=(
            "In src/db.rs, change the flat vectors Vec<f32> to be 32-byte aligned "
            "using manual alignment via std::alloc::alloc / Layout (no new crates). "
            "Add _mm_prefetch (std::arch::x86_64::_MM_HINT_T0) calls ahead of the distance loop. "
            "Keep the existing SoA (ids + flat vectors) layout. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    # ── Build profile ────────────────────────────────────────────────────────
    WorkerBrief(
        title="Aggressive release profile + LTO",
        family="profile",
        hypothesis="Fat LTO and codegen-units=1 allow LLVM to inline across crate boundaries, typically gaining 10-25% throughput for free.",
        instructions=(
            "In Cargo.toml, set [profile.release]: opt-level=3, lto='fat', codegen-units=1, panic='abort', strip='none'. "
            "Add RUSTFLAGS env hint via build.rs: println!(\"cargo:rustc-env=RUSTFLAGS=-C target-cpu=native\"). "
            "Also mark the distance fn and hot search loops with #[inline(always)]. "
            "Do not change any logic — only build configuration."
        ),
        target_files=("Cargo.toml",),
    ),
    WorkerBrief(
        title="Native CPU target + PGO hint via build.rs",
        family="profile",
        hypothesis="Compiling for the exact CPU microarchitecture enables AVX2/FMA without explicit intrinsics via auto-vectorisation.",
        instructions=(
            "Add build.rs that emits: println!(\"cargo:rustc-flags=-C target-cpu=native -C opt-level=3\"). "
            "In Cargo.toml: lto=true, codegen-units=1, opt-level=3. "
            "Add #[target_feature(enable=\"avx2\")] to l2_distance and the hot search loop so LLVM vectorises them. "
            "No logic changes — this is pure compile-time tuning."
        ),
        target_files=("Cargo.toml",),
    ),
    # ── Quantization ─────────────────────────────────────────────────────────
    WorkerBrief(
        title="Binary quantization + Hamming reranking",
        family="compression",
        hypothesis="Binary quantization reduces distance computation to popcount, which is a single CPU instruction.",
        instructions=(
            "In src/db.rs, on insert, pack each 128-dim f32 vector into 128 bits (16 bytes) using sign thresholding. "
            "At query time, scan with Hamming distance (XOR + popcount) to find top-500 binary candidates. "
            "Rerank with exact f32 L2. "
            "Add a src/binary.rs module for pack/hamming helpers. "
            "This requires recall >= 0.95 after reranking — tune the rerank candidate count accordingly."
        ),
        target_files=("src/db.rs", "Cargo.toml"),
    ),
    WorkerBrief(
        title="Product Quantization (PQ) with 8 subspaces",
        family="compression",
        hypothesis="PQ reduces 128-dim f32 distance to 8 table lookups, enabling very fast approximate search.",
        instructions=(
            "Implement PQ in src/db.rs (or src/pq.rs): "
            "8 subspaces of 16 dimensions each, 256 codewords per subspace trained with k-means. "
            "Encode each vector to 8 bytes on insert. "
            "At query time, precompute 8 distance tables (256 entries each) then scan codes with table lookups. "
            "Rerank top-k with exact f32. Update Cargo.toml if new crates are needed."
        ),
        target_files=("src/db.rs", "Cargo.toml"),
    ),
    # ── Concurrency model ────────────────────────────────────────────────────
    WorkerBrief(
        title="Lock-free read path with crossbeam",
        family="concurrency",
        hypothesis="Removing the RwLock from the read path eliminates contention under high query concurrency.",
        instructions=(
            "Replace the RwLock<Storage> with an atomic pointer approach using crossbeam (already in Cargo.toml). "
            "Use crossbeam::atomic::AtomicCell or wrap Storage in Arc and swap it with Arc::clone + Arc::swap. "
            "Writes (bulk_insert) build a new Storage Arc, then atomically replace the current. "
            "Reads (search) clone the current Arc cheaply without locking. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="Tokio async batching with query coalescing",
        family="concurrency",
        hypothesis="Batching multiple incoming queries and processing them as a single SIMD-friendly slab can improve throughput.",
        instructions=(
            "In src/db.rs, add a background Tokio task that receives queries via tokio::sync::mpsc channel, "
            "accumulates them for up to 1ms or 32 queries, then processes the batch with rayon par_iter. "
            "Return results via tokio::sync::oneshot channels. "
            "tokio is already in Cargo.toml. rayon is already in Cargo.toml. "
            "Expose a public VectorDB::search_async API that the existing handler can call. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    # ── Mixed / creative ─────────────────────────────────────────────────────
    WorkerBrief(
        title="Two-level IVF (IVF-IVF coarse-to-fine)",
        family="indexing",
        hypothesis="A two-level index reduces candidate set more aggressively than single IVF at similar recall cost.",
        instructions=(
            "Implement a two-level coarse-to-fine IVF in src/db.rs: "
            "Level 1: 32 super-centroids. Level 2: 16 sub-centroids per super-centroid (512 total). "
            "At query time, find top-4 super-centroids, then top-8 sub-centroids within those, "
            "scan exact L2 over members of selected sub-clusters. "
            "rayon is already in Cargo.toml — use it for parallel sub-cluster scanning. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs",),
    ),
    WorkerBrief(
        title="Randomised projection trees (RP-Tree) index",
        family="indexing",
        hypothesis="Random projection trees partition space recursively, yielding fast approximate search with configurable recall.",
        instructions=(
            "Implement a set of 8 random projection trees in src/db.rs (or src/rptree.rs). "
            "Each tree splits on a random unit vector at median. Build all trees on bulk_insert. "
            "At query time, descend all 8 trees in parallel (rayon is already in Cargo.toml), "
            "union the leaf candidates, rerank with exact L2. Tune tree depth (~14) for 1M vectors. "
            "IMPORTANT: Do not add any new crate dependencies to Cargo.toml."
        ),
        target_files=("src/db.rs", "src/rptree.rs"),
    ),
]


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _build_memory_text(
    baseline_qps: float,
    baseline_recall: float,
    sota_qps: float,
    sota_recall: float,
) -> str:
    """Build the memory text block injected into every prompt.

    Tells the model where the incumbent stands and what the leaderboard leader
    achieves, so it can reason about how much improvement is possible and which
    strategies are worth pursuing.
    """
    return (
        "# Performance Baseline\n\n"
        f"Incumbent implementation (brute-force flat scan):\n"
        f"- QPS: {baseline_qps:.0f} queries/second\n"
        f"- Recall: {baseline_recall:.3f} (well above the 0.95 threshold)\n\n"
        f"Leaderboard leader (single-turn, no agentic loop):\n"
        f"- QPS: {sota_qps:.0f} queries/second\n"
        f"- Recall: {sota_recall:.3f}\n\n"
        f"Gap: {sota_qps / max(baseline_qps, 1):.0f}x improvement over incumbent.\n"
        "This gap comes from indexing (IVF, HNSW), parallelism (rayon), and SIMD "
        "distance kernels — not from cheating recall. Your implementation must "
        "maintain recall >= 0.95 to be scored.\n"
        "Strategies that trade a small amount of recall for large QPS gains "
        "(e.g., approximate search with reranking) are the right direction."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GRPO RL prompt dataset")
    parser.add_argument(
        "--bench-repo", type=Path, required=True,
        help="Path to a local clone of KCORES/vector-db-bench",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--program-path", type=Path, default=DEFAULT_PROGRAM,
        help="Path to program.md",
    )
    parser.add_argument(
        "--incumbent-dir", type=Path, default=None,
        help="Directory with incumbent files (e.g. a run's initial_candidate/). "
             "Defaults to the seeded baseline.",
    )
    parser.add_argument(
        "--baseline-qps", type=float, default=55.0,
        help="QPS of the brute-force incumbent (default: 55, measured from leaderboard data).",
    )
    parser.add_argument(
        "--baseline-recall", type=float, default=0.999,
        help="Recall of the brute-force incumbent (default: 0.999).",
    )
    parser.add_argument(
        "--sota-qps", type=float, default=3548.0,
        help="QPS of the current leaderboard leader (default: 3548, claude-opus-4.6 single-turn).",
    )
    parser.add_argument(
        "--sota-recall", type=float, default=0.959,
        help="Recall of the leaderboard leader (default: 0.959).",
    )
    args = parser.parse_args()

    bench_repo = args.bench_repo.resolve()
    skeleton_dir = bench_repo / "skeleton"
    if not skeleton_dir.is_dir():
        print(f"[generate_rl_prompts] missing skeleton dir: {skeleton_dir}", file=sys.stderr)
        sys.exit(1)

    # Incumbent files: actual skeleton or seeded baseline
    if args.incumbent_dir is not None:
        incumbent_dir = args.incumbent_dir.resolve()
        if not incumbent_dir.is_dir():
            print(f"[generate_rl_prompts] --incumbent-dir not found: {incumbent_dir}", file=sys.stderr)
            sys.exit(1)
        incumbent_files: dict[str, str] = {}
        for f in sorted(incumbent_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(incumbent_dir).as_posix()
            if rel.endswith(".rs") or rel in ("Cargo.toml", "build.rs"):
                incumbent_files[rel] = _read_text(f)
        print(f"[generate_rl_prompts] loaded {len(incumbent_files)} files from --incumbent-dir")
    else:
        incumbent_files = _bootstrap_seed_surface(skeleton_dir)
        # Override Cargo.toml with the enriched version that pre-seeds useful crates.
        # This must match the incumbent_files dict used in modal_rl_train.py's reward fn.
        incumbent_files["Cargo.toml"] = (
            "[package]\n"
            'name = "vector-db-skeleton"\n'
            'version = "0.1.0"\n'
            'edition = "2021"\n'
            "\n"
            "[dependencies]\n"
            'axum = "0.7"\n'
            'tokio = { version = "1", features = ["full"] }\n'
            'serde = { version = "1", features = ["derive"] }\n'
            'serde_json = "1"\n'
            'rayon = "1"\n'
            'parking_lot = "0.12"\n'
            'crossbeam = "0.8"\n'
            "\n"
            "[profile.release]\n"
            "lto = true\n"
            "codegen-units = 1\n"
            "opt-level = 3\n"
        )
        print(f"[generate_rl_prompts] using seeded baseline ({len(incumbent_files)} files) with enriched Cargo.toml")

    program_text = _read_text(args.program_path)
    readonly_context = _read_readonly_context(skeleton_dir)
    memory_text = _build_memory_text(
        baseline_qps=args.baseline_qps,
        baseline_recall=args.baseline_recall,
        sota_qps=args.sota_qps,
        sota_recall=args.sota_recall,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for brief in _BRIEFS:
            # Filter out target_files not in the incumbent (model can add new files, but
            # the prompt should only list existing ones as primary targets).
            valid_targets = tuple(t for t in brief.target_files if t in incumbent_files)
            if not valid_targets:
                valid_targets = tuple(brief.target_files[:1])  # keep at least one

            trimmed_brief = WorkerBrief(
                title=brief.title,
                family=brief.family,
                hypothesis=brief.hypothesis,
                instructions=brief.instructions,
                target_files=valid_targets,
            )

            prompt_text = _worker_prompt(
                program_text=program_text,
                memory_text=memory_text,
                reviewer_notes_text=DEFAULT_REVIEWER_NOTES,
                mutable_files=incumbent_files,
                readonly_context=readonly_context,
                recent_rows=[],
                brief=trimmed_brief,
            )

            record = {
                # TRL GRPOTrainer expects a "prompt" column.
                # Using chat format so the tokenizer applies the correct template.
                "prompt": [{"role": "user", "content": prompt_text}],
                "brief_title": brief.title,
                "brief_family": brief.family,
                "incumbent_files": list(incumbent_files.keys()),
                # Baseline numbers for the QPS reward normalisation in modal_rl_train.py.
                "baseline_qps": args.baseline_qps,
                "sota_qps": args.sota_qps,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[generate_rl_prompts] wrote {written} prompts → {args.output}")
    for b in _BRIEFS:
        print(f"  {b.family:16s}  {b.title}")


if __name__ == "__main__":
    main()
