from __future__ import annotations

import json
import re
from typing import Any


def _read_optional(context, path: str) -> str:
    payload = context.read_file(path)
    if payload.get("type") == "ReadFile":
        return str(payload.get("content", ""))
    return ""


def _classify_architecture(db_src: str) -> dict[str, Any]:
    lowered = db_src.lower()
    signals = []
    if any(token in lowered for token in ("centroid", "cluster", "kmeans", "probe", "nprobe", "ivf")):
        signals.append("cluster_terms")
    if "select_nth_unstable" in lowered:
        signals.append("partial_topk")
    if any(token in lowered for token in ("unsafe", "std::arch", "avx", "simd")):
        signals.append("simd")
    if any(token in lowered for token in ("rwlock", "mutex")):
        signals.append("locking")

    if "cluster_terms" in signals:
        label = "ivf_like"
        next_move = "stabilize cluster assignment, probing, and exact reranking before low-level tuning"
    elif "partial_topk" in signals:
        label = "exact_partial_topk"
        next_move = "pivot toward candidate pruning or clustered shortlist generation"
    else:
        label = "bruteforce_or_unclear"
        next_move = "build a shortlist-generating ANN structure instead of polishing full scan"
    return {
        "label": label,
        "signals": signals,
        "next_move": next_move,
    }


def register_tools(registry) -> None:
    def review_run_state(context, arguments):
        progress = context.get_progress_state()
        architecture = _classify_architecture(_read_optional(context, "src/db.rs"))
        return {
            "status": "ok",
            "progress": progress,
            "architecture": architecture,
            "advice": [
                "Protect the best valid state aggressively.",
                "Prefer structural ANN progress over late exact-search micro-optimizations.",
                "Use helper tools to checkpoint after each substantial architectural change.",
            ],
        }

    def checkpoint_candidate(context, arguments):
        label = str(arguments.get("label", "checkpoint"))
        profile = bool(arguments.get("profile", False))
        restore_on_failure = bool(arguments.get("restore_on_failure", True))

        build_result = context.build_project()
        if not build_result.get("success", False):
            if restore_on_failure:
                restore_result = context.restore_best_candidate()
            else:
                restore_result = None
            return {
                "status": "build_failed",
                "label": label,
                "build": build_result,
                "restore": restore_result,
            }

        correctness = context.run_correctness_test()
        correctness_passed = bool(correctness.get("recall_passed", False))
        benchmark = context.run_benchmark(max_queries=1000)
        benchmark_valid = bool(benchmark.get("recall_passed", False)) and float(benchmark.get("qps", 0.0) or 0.0) > 0.0

        profile_result = None
        if profile and benchmark_valid:
            profile_result = context.run_profiling(duration=20)

        if restore_on_failure and (not correctness_passed or not benchmark_valid):
            restore_result = context.restore_best_candidate()
        else:
            restore_result = None

        checkpoint_manifest = context.checkpoint_best_candidate(note=label)
        return {
            "status": "ok" if correctness_passed and benchmark_valid else "invalid_candidate",
            "label": label,
            "build": build_result,
            "correctness": correctness,
            "benchmark": benchmark,
            "profile": profile_result,
            "restore": restore_result,
            "checkpoint": checkpoint_manifest,
        }

    def restore_best_candidate(context, arguments):
        return context.restore_best_candidate()

    def assess_architecture(context, arguments):
        db_src = _read_optional(context, "src/db.rs")
        distance_src = _read_optional(context, "src/distance.rs")
        architecture = _classify_architecture(db_src)
        simd_signals = []
        if re.search(r"\b(avx|simd|std::arch|unsafe)\b", distance_src.lower()):
            simd_signals.append("distance_kernel_tuning_present")
        return {
            "status": "ok",
            "architecture": architecture,
            "distance_signals": simd_signals,
        }

    def benchmark_policy(context, arguments):
        progress = context.get_progress_state()
        best_qps = float(((progress.get("best_benchmark") or {}).get("qps", 0.0)) or 0.0)
        tool_calls_used = int(progress.get("tool_calls_used", 0) or 0)
        if best_qps < 100.0:
            recommendation = "Use quick 1K-query checkpoints while changing architecture. Do not spend full 10K benchmarks on exact-search polish."
            next_action = "quick_checkpoint"
        elif best_qps < 1000.0:
            recommendation = "Run quick checkpoints after structural changes and occasional profiling. Save full benchmarks for credible ANN candidates."
            next_action = "mixed_validation"
        else:
            recommendation = "Use full benchmarks to stabilize promising ANN candidates and confirm milestone jumps."
            next_action = "full_validation"
        if tool_calls_used > 0:
            recommendation += f" Tool calls used so far: {tool_calls_used}."
        return {
            "status": "ok",
            "best_qps": best_qps,
            "recommendation": recommendation,
            "next_action": next_action,
        }

    def profile_summary(context, arguments):
        run_profile = bool(arguments.get("run_profile", False))
        if run_profile:
            profile_payload = context.run_profiling(duration=int(arguments.get("duration", 20) or 20))
            return {
                "status": "ok",
                "profile": profile_payload,
                "note": "Prefer the top-functions text output over raw SVG inspection.",
            }
        files_payload = context.list_files("profiling")
        files = files_payload.get("files", []) if files_payload.get("type") == "ListFiles" else []
        txt_files = sorted(name for name in files if name.endswith(".txt") or name.endswith(".json"))
        latest = txt_files[-1] if txt_files else None
        latest_payload = _read_optional(context, f"profiling/{latest}") if latest else ""
        return {
            "status": "ok",
            "available_files": files,
            "latest_text_summary_file": latest,
            "latest_text_summary": latest_payload,
        }

    registry.add_tool(
        name="review_run_state",
        description="Summarize the current best valid state, milestone, and likely next move.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=review_run_state,
    )
    registry.add_tool(
        name="checkpoint_candidate",
        description="Run build, correctness, and a quick benchmark, then checkpoint or restore as needed.",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "profile": {"type": "boolean"},
                "restore_on_failure": {"type": "boolean"},
            },
            "required": [],
        },
        handler=checkpoint_candidate,
    )
    registry.add_tool(
        name="restore_best_candidate",
        description="Restore the best known valid worker code state.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=restore_best_candidate,
    )
    registry.add_tool(
        name="assess_architecture",
        description="Classify the current implementation and recommend the next structural move.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=assess_architecture,
    )
    registry.add_tool(
        name="benchmark_policy",
        description="Recommend whether to use quick or full benchmark validation next.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=benchmark_policy,
    )
    registry.add_tool(
        name="profile_summary",
        description="Run profiling or summarize the latest profiling text artifacts without reading raw SVGs.",
        parameters={
            "type": "object",
            "properties": {
                "run_profile": {"type": "boolean"},
                "duration": {"type": "integer"},
            },
            "required": [],
        },
        handler=profile_summary,
    )
