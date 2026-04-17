from __future__ import annotations

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
    if any(token in lowered for token in ("select_nth_unstable", "binaryheap", "online top-k", "heap")):
        signals.append("topk_selection")
    if any(token in lowered for token in ("unsafe", "std::arch", "avx", "simd")):
        signals.append("simd")
    if any(token in lowered for token in ("rwlock", "mutex")):
        signals.append("locking")
    if any(token in lowered for token in ("par_iter", "rayon")):
        signals.append("rayon")

    if "cluster_terms" in signals:
        label = "ivf_like"
        next_move = "stabilize probing, contiguous list layout, and online top-k before more low-level tuning"
    elif "topk_selection" in signals:
        label = "exact_partial_topk"
        next_move = "fork an ANN experiment and stop polishing full scan"
    else:
        label = "bruteforce_or_unclear"
        next_move = "establish one valid baseline, then pivot to shortlist generation"
    return {
        "label": label,
        "signals": signals,
        "next_move": next_move,
    }


def _recent_benchmark_events(context) -> list[dict[str, Any]]:
    events = []
    for entry in context.get_call_log():
        if entry.get("tool") == "run_benchmark":
            events.append(entry)
    return events


def _recent_benchmark_summary(context) -> dict[str, Any]:
    events = _recent_benchmark_events(context)
    if not events:
        return {
            "count": 0,
            "recent_errors": [],
            "best_valid_qps": 0.0,
            "last_valid_qps": 0.0,
        }
    recent_errors = []
    best_valid_qps = 0.0
    last_valid_qps = 0.0
    for entry in events:
        output = entry.get("output") or {}
        if output.get("type") == "RunBenchmark" and output.get("recall_passed"):
            qps = float(output.get("qps", 0.0) or 0.0)
            best_valid_qps = max(best_valid_qps, qps)
            last_valid_qps = qps
        elif output.get("type") == "Error":
            recent_errors.append(str(output.get("message", ""))[:220])
    return {
        "count": len(events),
        "recent_errors": recent_errors[-5:],
        "best_valid_qps": best_valid_qps,
        "last_valid_qps": last_valid_qps,
    }


def register_tools(registry) -> None:
    def review_run_state(context, arguments):
        progress = context.get_progress_state()
        campaign = context.get_campaign_state()
        architecture = _classify_architecture(_read_optional(context, "src/db.rs"))
        return {
            "status": "ok",
            "progress": progress,
            "campaign": campaign,
            "architecture": architecture,
            "advice": [
                "Keep the mainline valid and small experiments explicit.",
                "Use promote_experiment only after a meaningful valid improvement.",
                "If identical benchmark failures recur, change code or parameters before retrying.",
            ],
        }

    def checkpoint_candidate(context, arguments):
        label = str(arguments.get("label", "checkpoint"))
        profile = bool(arguments.get("profile", False))
        restore_on_failure = bool(arguments.get("restore_on_failure", True))

        build_result = context.build_project()
        if not build_result.get("success", False):
            restore_result = context.restore_mainline() if restore_on_failure else None
            return {
                "status": "build_failed",
                "label": label,
                "build": build_result,
                "restore": restore_result,
            }

        correctness = context.run_correctness_test()
        benchmark = context.run_benchmark(max_queries=int(arguments.get("max_queries", 1000) or 1000))
        profile_result = context.run_profiling(duration=int(arguments.get("duration", 20) or 20)) if profile else None

        valid = bool(correctness.get("recall_passed", False)) and bool(benchmark.get("recall_passed", False)) and float(benchmark.get("qps", 0.0) or 0.0) > 0.0
        best_checkpoint = context.checkpoint_best_candidate(note=label)
        campaign = context.get_campaign_state()
        current_mainline = (((campaign.get("mainline_manifest") or {}).get("best_benchmark") or {}).get("qps", 0.0) or 0.0)
        promoted = None
        restore_result = None
        if valid and float(benchmark.get("qps", 0.0) or 0.0) >= float(current_mainline):
            promoted = context.checkpoint_mainline(note=label)
        elif restore_on_failure and not valid:
            restore_result = context.restore_mainline()

        return {
            "status": "ok" if valid else "invalid_candidate",
            "label": label,
            "build": build_result,
            "correctness": correctness,
            "benchmark": benchmark,
            "profile": profile_result,
            "best_checkpoint": best_checkpoint,
            "promoted_mainline": promoted,
            "restore": restore_result,
        }

    def restore_best_candidate(context, arguments):
        return context.restore_best_candidate()

    def checkpoint_mainline(context, arguments):
        return context.checkpoint_mainline(note=str(arguments.get("label", "manual_mainline")))

    def restore_mainline(context, arguments):
        return context.restore_mainline()

    def fork_experiment(context, arguments):
        return context.fork_experiment(label=str(arguments.get("label", "experiment")))

    def promote_experiment(context, arguments):
        return context.promote_experiment(label=str(arguments.get("label", "")))

    def discard_experiment(context, arguments):
        return context.discard_experiment(label=str(arguments.get("label", "")))

    def assess_architecture(context, arguments):
        db_src = _read_optional(context, "src/db.rs")
        distance_src = _read_optional(context, "src/distance.rs")
        architecture = _classify_architecture(db_src)
        distance_signals = []
        if re.search(r"\b(avx|simd|std::arch|unsafe)\b", distance_src.lower()):
            distance_signals.append("distance_kernel_tuning_present")
        return {
            "status": "ok",
            "architecture": architecture,
            "distance_signals": distance_signals,
        }

    def benchmark_policy(context, arguments):
        progress = context.get_progress_state()
        history = _recent_benchmark_summary(context)
        guard = progress.get("benchmark_failure_guard") or {}
        best_qps = float(((progress.get("best_benchmark") or {}).get("qps", 0.0)) or 0.0)
        architecture = _classify_architecture(_read_optional(context, "src/db.rs"))

        if guard and int(guard.get("count", 0) or 0) >= 3:
            recommendation = "Do not benchmark again until code or benchmark parameters change. Investigate the repeated failure signature first."
            next_action = "fix_before_benchmark"
        elif architecture.get("label") != "ivf_like":
            recommendation = "Use only smoke or quick checkpoints while moving toward IVF. Avoid full benchmarks on exact-search branches."
            next_action = "cheap_validation_only"
        elif best_qps < 500.0:
            recommendation = "Use quick checkpoints to stabilize the first valid IVF path before larger validation."
            next_action = "quick_ivf_validation"
        else:
            recommendation = "Use mixed quick and occasional full validation near milestone transitions."
            next_action = "mixed_validation"
        return {
            "status": "ok",
            "best_qps": best_qps,
            "history": history,
            "failure_guard": guard,
            "recommendation": recommendation,
            "next_action": next_action,
        }

    def benchmark_history_summary(context, arguments):
        progress = context.get_progress_state()
        history = _recent_benchmark_summary(context)
        return {
            "status": "ok",
            "history": history,
            "failure_guard": progress.get("benchmark_failure_guard") or {},
        }

    def profile_summary(context, arguments):
        run_profile = bool(arguments.get("run_profile", False))
        if run_profile:
            profile_payload = context.run_profiling(duration=int(arguments.get("duration", 20) or 20))
            return {
                "status": "ok",
                "profile": profile_payload,
                "note": "Prefer text summaries over raw SVG inspection.",
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

    def research_notes(context, arguments):
        return {
            "status": "ok",
            "research_notes": _read_optional(context, "src/research_notes.md"),
        }

    def plan_next_step(context, arguments):
        architecture = _classify_architecture(_read_optional(context, "src/db.rs"))
        history = _recent_benchmark_summary(context)
        guard = context.get_progress_state().get("benchmark_failure_guard") or {}
        if guard and int(guard.get("count", 0) or 0) >= 3:
            plan = "Stop benchmarking. Fix the repeated failure mode or change benchmark parameters, then checkpoint again."
        elif architecture.get("label") != "ivf_like":
            plan = "Keep one valid exact fallback, then fork an IVF experiment immediately."
        elif history.get("best_valid_qps", 0.0) < 500.0:
            plan = "Make the first valid IVF path stable and promote it if it beats mainline."
        else:
            plan = "Tune contiguous list layout, online top-k, and scan ratio before further low-level polish."
        return {
            "status": "ok",
            "plan": plan,
            "architecture": architecture,
            "history": history,
            "failure_guard": guard,
        }

    registry.add_tool(name="review_run_state", description="Summarize current progress, mainline state, and likely next move.", parameters={"type": "object", "properties": {}, "required": []}, handler=review_run_state)
    registry.add_tool(name="checkpoint_candidate", description="Validate the current branch, checkpoint it, and optionally promote it to mainline.", parameters={"type": "object", "properties": {"label": {"type": "string"}, "profile": {"type": "boolean"}, "restore_on_failure": {"type": "boolean"}, "max_queries": {"type": "integer"}, "duration": {"type": "integer"}}, "required": []}, handler=checkpoint_candidate)
    registry.add_tool(name="restore_best_candidate", description="Restore the latest best-candidate snapshot.", parameters={"type": "object", "properties": {}, "required": []}, handler=restore_best_candidate)
    registry.add_tool(name="checkpoint_mainline", description="Promote the current worker code to the persistent mainline snapshot.", parameters={"type": "object", "properties": {"label": {"type": "string"}}, "required": []}, handler=checkpoint_mainline)
    registry.add_tool(name="restore_mainline", description="Restore the persistent promoted mainline branch.", parameters={"type": "object", "properties": {}, "required": []}, handler=restore_mainline)
    registry.add_tool(name="fork_experiment", description="Create a named experiment branch snapshot from the current worker state.", parameters={"type": "object", "properties": {"label": {"type": "string"}}, "required": []}, handler=fork_experiment)
    registry.add_tool(name="promote_experiment", description="Promote a named experiment branch into mainline.", parameters={"type": "object", "properties": {"label": {"type": "string"}}, "required": []}, handler=promote_experiment)
    registry.add_tool(name="discard_experiment", description="Discard an experiment branch and optionally restore mainline.", parameters={"type": "object", "properties": {"label": {"type": "string"}}, "required": []}, handler=discard_experiment)
    registry.add_tool(name="assess_architecture", description="Classify the current implementation and recommend the next structural move.", parameters={"type": "object", "properties": {}, "required": []}, handler=assess_architecture)
    registry.add_tool(name="benchmark_policy", description="Recommend whether the campaign should benchmark now or fix something first.", parameters={"type": "object", "properties": {}, "required": []}, handler=benchmark_policy)
    registry.add_tool(name="benchmark_history_summary", description="Summarize recent benchmark calls and repeated failure signatures.", parameters={"type": "object", "properties": {}, "required": []}, handler=benchmark_history_summary)
    registry.add_tool(name="profile_summary", description="Run or summarize profiling without forcing raw SVG reads.", parameters={"type": "object", "properties": {"run_profile": {"type": "boolean"}, "duration": {"type": "integer"}}, "required": []}, handler=profile_summary)
    registry.add_tool(name="research_notes", description="Return the current worker-facing research notes.", parameters={"type": "object", "properties": {}, "required": []}, handler=research_notes)
    registry.add_tool(name="plan_next_step", description="Turn the current campaign state into the next concrete implementation objective.", parameters={"type": "object", "properties": {}, "required": []}, handler=plan_next_step)
