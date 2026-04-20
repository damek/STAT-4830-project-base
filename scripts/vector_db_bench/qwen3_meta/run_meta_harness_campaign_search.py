#!/usr/bin/env python3
"""Run a Codex-supervised long-horizon Meta-Harness campaign search.

This mode keeps a persistent worker mainline across revisions and lets Codex
periodically revise the harness around that evolving worker codebase.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from meta_harness_common import DEFAULT_DOTENV_PATH, load_dotenv, write_json


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_CAMPAIGN_REVISIONS_ROOT = SCRIPT_PATH.with_name("meta_harness_campaign") / "revisions"
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_BLANK_SEED_SOURCE = DEFAULT_BENCH_REPO / "skeleton"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_QWEN_BURST_TOOL_CALLS = 50
DEFAULT_SEARCH_ITERATIONS = 20
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_campaign_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
SEARCH_COLUMNS = [
    "iteration",
    "parent_revision",
    "candidate_revision",
    "accepted",
    "incumbent_best_qps_before",
    "candidate_best_qps",
    "candidate_best_milestone",
    "candidate_goal_reached",
    "codex_returncode",
    "codex_runtime_seconds",
    "candidate_campaign_run_root",
    "incumbent_campaign_run_root_after",
    "mainline_snapshot_dir_after",
    "notes",
]


@dataclass(frozen=True)
class CodexExecResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    last_message: str
    runtime_seconds: float


@dataclass(frozen=True)
class CampaignSummary:
    revision_id: str
    cycles_requested: int
    cycles_completed: int
    valid_cycles: int
    best_qps: float
    best_cycle: int | None
    best_recall: float
    median_valid_qps: float
    mean_valid_qps: float
    mean_elapsed_secs: float
    goal_qps: float
    goal_reached: bool
    best_milestone: str
    mainline_snapshot_dir: str | None
    mainline_manifest_path: str | None
    run_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--revisions-root", type=Path, default=DEFAULT_CAMPAIGN_REVISIONS_ROOT)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--base-revision-id", type=str, default="campaign_000")
    parser.add_argument("--reuse-incumbent-run-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS)
    parser.add_argument("--cycles-per-revision", type=int, default=1)

    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_QWEN_BURST_TOOL_CALLS)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)
    parser.add_argument("--goal-qps", type=float, default=4000.0)

    parser.add_argument("--model-name", type=str, default="qwen3-coder-next")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-id", type=str, default="qwen/qwen3-coder-next")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--thinking-mode", type=str, default="none")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)

    parser.add_argument("--codex-executable", type=str, default="codex")
    parser.add_argument("--codex-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--codex-sandbox", choices=("read-only", "workspace-write", "danger-full-access"), default="workspace-write")
    parser.add_argument("--codex-model", type=str, default="")
    parser.add_argument("--codex-oss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--codex-local-provider", choices=("", "ollama", "lmstudio"), default="")
    parser.add_argument("--codex-enable-web-search", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=SEARCH_COLUMNS, delimiter="\t")
    writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def _flush_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.flush()


def _close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def _require_codex(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Could not find Codex CLI executable {executable!r} on PATH")
    return resolved


def _run_codex_exec(*, executable: str, prompt: str, cwd: Path, output_path: Path, events_path: Path, timeout_seconds: int, sandbox: str, model: str, use_oss: bool, local_provider: str, enable_web_search: bool) -> CodexExecResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        executable,
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--json",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "--output-last-message",
        str(output_path),
    ]
    if model:
        argv.extend(["-m", model])
    if enable_web_search:
        argv.extend(["--enable", "web_search_request"])
    if use_oss:
        argv.append("--oss")
    if local_provider:
        argv.extend(["--local-provider", local_provider])
    argv.append("-")

    started_at = time.time()
    proc = subprocess.run(argv, input=prompt, text=True, capture_output=True, cwd=cwd, timeout=timeout_seconds)
    events_path.write_text(proc.stdout, encoding="utf-8")
    last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    return CodexExecResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        last_message=last_message,
        runtime_seconds=time.time() - started_at,
    )


def _load_campaign_summary(run_root: Path) -> CampaignSummary:
    payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    return CampaignSummary(
        revision_id=str(payload["revision_id"]),
        cycles_requested=int(payload["cycles_requested"]),
        cycles_completed=int(payload["cycles_completed"]),
        valid_cycles=int(payload["valid_cycles"]),
        best_qps=float(payload["best_qps"]),
        best_cycle=(int(payload["best_cycle"]) if payload.get("best_cycle") is not None else None),
        best_recall=float(payload["best_recall"]),
        median_valid_qps=float(payload["median_valid_qps"]),
        mean_valid_qps=float(payload["mean_valid_qps"]),
        mean_elapsed_secs=float(payload["mean_elapsed_secs"]),
        goal_qps=float(payload["goal_qps"]),
        goal_reached=bool(payload["goal_reached"]),
        best_milestone=str(payload["best_milestone"]),
        mainline_snapshot_dir=str(payload.get("mainline_snapshot_dir")) if payload.get("mainline_snapshot_dir") else None,
        mainline_manifest_path=str(payload.get("mainline_manifest_path")) if payload.get("mainline_manifest_path") else None,
        run_root=str(payload["run_root"]),
    )


def _is_better(candidate: CampaignSummary, incumbent: CampaignSummary) -> bool:
    left = (
        int(candidate.goal_reached),
        candidate.best_qps,
        candidate.valid_cycles,
        candidate.median_valid_qps,
        -candidate.mean_elapsed_secs,
    )
    right = (
        int(incumbent.goal_reached),
        incumbent.best_qps,
        incumbent.valid_cycles,
        incumbent.median_valid_qps,
        -incumbent.mean_elapsed_secs,
    )
    return left > right


def _rewrite_revision_identity(revision_toml: Path, *, revision_id: str, description: str, notes: str) -> None:
    content = revision_toml.read_text(encoding="utf-8")
    replacements = {
        "id": repr(revision_id),
        "description": repr(description),
        "notes": repr(notes),
    }
    for key, value in replacements.items():
        pattern = rf"(?m)^{re.escape(key)}\s*=\s*.*$"
        if re.search(pattern, content):
            content = re.sub(pattern, f"{key} = {value}", content, count=1)
        else:
            content += f"\n{key} = {value}\n"
    revision_toml.write_text(content, encoding="utf-8")


def _seed_candidate_revision(*, workspace: Path, parent_revision_id: str, candidate_revision_id: str) -> Path:
    revisions_root = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "meta_harness_campaign" / "revisions"
    parent_dir = revisions_root / parent_revision_id
    candidate_dir = revisions_root / candidate_revision_id
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_dir, candidate_dir)
    _rewrite_revision_identity(
        candidate_dir / "revision.toml",
        revision_id=candidate_revision_id,
        description=f"Codex-authored campaign candidate based on {parent_revision_id}",
        notes=f"Candidate generated by the campaign search loop from parent {parent_revision_id}.",
    )
    notes_path = candidate_dir / "harness_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Harness Notes\n\n- architectural bet: fill this in\n- intended causal mechanism: fill this in\n",
            encoding="utf-8",
        )
    return candidate_dir


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _tail_file(src: Path, dst: Path, lines: int = 120) -> None:
    if not src.exists():
        return
    content = src.read_text(encoding="utf-8", errors="replace").splitlines()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(content[-lines:]) + "\n", encoding="utf-8")


def _write_context_bundle(*, workspace: Path, search_run_root: Path, incumbent_revision_id: str, incumbent_summary: CampaignSummary, incumbent_run_root: Path, iteration: int) -> Path:
    context_root = workspace / ".meta_harness_campaign_search_context"
    if context_root.exists():
        shutil.rmtree(context_root)
    context_root.mkdir(parents=True, exist_ok=True)

    write_json(
        context_root / "task.json",
        {
            "iteration": iteration,
            "incumbent_revision_id": incumbent_revision_id,
            "incumbent_campaign_run_root": str(incumbent_run_root),
            "objective": {
                "primary": "Run a long-lived hill-climbing campaign that gets qwen/qwen3-coder-next to a valid 4000+ QPS vector-db-bench solution.",
                "secondary": [
                    "Keep the worker productive across many cycles instead of restarting from scratch.",
                    "Use Codex as a hands-on supervisor that revises strategy, tools, research notes, and branching discipline.",
                    "Produce staircase-like progress across the campaign trajectory.",
                ],
                "search_style": "Treat this as an autonomous research campaign, not a fresh-start benchmark comparison.",
                "codex_role": [
                    "Inspect the persistent campaign trace and current mainline worker state.",
                    "Research strong public vector-db solution patterns when useful.",
                    "Design detailed teacher artifacts, benchmark policy, and helper tools for the next campaign segment.",
                    "Act as the conductor across many short worker bursts and help Qwen as much as possible short of directly writing the worker Rust solution.",
                ],
            },
            "campaign_constraints": {
                "worker_model": "qwen/qwen3-coder-next",
                "worker_writes_solution_code": True,
                "persistent_mainline_across_revisions": True,
                "persistent_experiment_branches": True,
                "online_research_allowed": True,
                "goal_qps": incumbent_summary.goal_qps,
                "qwen_burst_tool_call_budget": args.max_tool_calls,
                "qwen_role": "actuator",
                "codex_role": "conductor",
            },
            "worker_file_scope": {
                "readable": ["src/*", "Cargo.toml", "benchmarks/*", "profiling/*"],
                "writable": ["src/* (except protected files)", "Cargo.toml"],
                "seed_file_guidance": "Mount worker-facing files inside src/.",
            },
            "available_harness_surfaces": [
                "extra user messages",
                "teacher files under src/",
                "helper tools via revision-local helper_tools.py",
                "campaign mainline / branch semantics",
                "benchmark and profile governance",
                "research notes and supervisor summaries",
                "zero-completion retry policy",
            ],
        },
    )
    write_json(context_root / "incumbent_summary.json", incumbent_summary.__dict__)
    _copy_if_exists(incumbent_run_root / "results.tsv", context_root / "incumbent_results.tsv")
    _copy_if_exists(incumbent_run_root / "summary.json", context_root / "incumbent_run_summary.json")
    _copy_if_exists(incumbent_run_root / "campaign_config.json", context_root / "campaign_config.json")
    _copy_if_exists(incumbent_run_root / "trajectory.jsonl", context_root / "trajectory.jsonl")
    _copy_if_exists(search_run_root / "search_results.tsv", context_root / "search_results.tsv")
    workspace_root = incumbent_run_root / "workspace"
    _copy_if_exists(workspace_root / "session_context.json", context_root / "session_context.json")
    _copy_if_exists(workspace_root / "eval_log.json", context_root / "eval_log.json")
    _copy_if_exists(workspace_root / ".meta_harness" / "progress_state.json", context_root / "progress_state.json")
    _copy_if_exists(workspace_root / ".meta_harness" / "mainline_manifest.json", context_root / "mainline_manifest.json")
    _copy_if_exists(workspace_root / ".meta_harness" / "campaign_state.json", context_root / "campaign_state.json")
    if incumbent_summary.mainline_snapshot_dir:
        _copy_tree_if_exists(Path(incumbent_summary.mainline_snapshot_dir), context_root / "mainline_snapshot")
    src_dir = workspace_root / "src"
    if src_dir.exists():
        for path in sorted(src_dir.glob("*.md")):
            _copy_if_exists(path, context_root / "worker_src" / path.name)
    for cycle_summary in sorted(incumbent_run_root.glob("cycles/cycle_*/cycle_summary.json")):
        rel = cycle_summary.relative_to(incumbent_run_root)
        _copy_if_exists(cycle_summary, context_root / rel)
    for stderr_log in sorted(incumbent_run_root.glob("cycles/cycle_*/results/run_eval.stderr.log")):
        rel = stderr_log.relative_to(incumbent_run_root).with_suffix(".stderr.tail.txt")
        _tail_file(stderr_log, context_root / rel)
    for agent_log in sorted(incumbent_run_root.glob("workspace/agent_log.jsonl")):
        rel = agent_log.relative_to(incumbent_run_root).with_suffix(".tail.jsonl")
        _tail_file(agent_log, context_root / rel, lines=250)
    return context_root


def _build_codex_prompt(*, candidate_revision_id: str, parent_revision_id: str, context_root: Path) -> str:
    return textwrap.dedent(
        f"""\
        You are authoring a new campaign-mode Meta-Harness revision for vector-db-bench.

        Objective:
        - help qwen/qwen3-coder-next reach a valid 4000+ QPS solution through a long-running hill-climbing campaign
        - inspect the persistent campaign record and current mainline worker snapshot
        - design the strongest supervisor package you can without directly writing the worker Rust solution
        - be hands-on: research public high-performing solution patterns when useful, design the strategy, add highly specific helper tools, and tighten the orchestration around the worker

        Parent revision:
        - {parent_revision_id}

        Candidate revision to author:
        - {candidate_revision_id}

        Hard framing:
        - qwen writes the Rust solution
        - qwen is the constrained actuator
        - qwen should operate under the same per-burst tool-call budget as the baseline worker condition
        - you are the conductor across many bursts and revisions
        - you design the optimization environment around qwen
        - worker mainline persists across revisions in this campaign mode
        - your revision will be evaluated on top of the incumbent mainline snapshot, not from a blank codebase
        - branch/promote/restore mechanics are first-class
        - helper tools, teacher files, summaries, and benchmark governance are all valid
        - creativity is encouraged if it helps Qwen reach state-of-the-art behavior

        Strongly preferred revision shapes:
        - structured teacher package under src/
        - helper tools that improve branch control, architecture review, benchmark policy, or supervisor summaries
        - solution-specific research notes or checklists for the intended ANN / IVF-style direction
        - orchestration changes that reduce thrash and help the worker climb through architectural phases

        Worker file scope:
        - readable: src/*, Cargo.toml, benchmarks/*, profiling/*
        - writable: src/* except protected files, plus Cargo.toml
        - seed worker-facing files under src/

        Record to inspect first:
        - {context_root}/task.json
        - {context_root}/incumbent_summary.json
        - {context_root}/incumbent_results.tsv
        - {context_root}/trajectory.jsonl
        - {context_root}/mainline_manifest.json
        - {context_root}/campaign_state.json
        - {context_root}/progress_state.json
        - {context_root}/mainline_snapshot/
        - any cycle summaries, stderr tails, and worker markdown files under {context_root}
        - scripts/vector_db_bench/qwen3_meta/META_HARNESS_CAMPAIGN_SPEC.md

        Revision standard:
        - optimize for long-horizon staircase progress, not tiny local deltas
        - optimize for speed of convergence, not just local polish
        - assume qwen gets a short burst and you must front-load the highest-leverage guidance into the harness
        - research is allowed when useful
        - it is good to be directive and opinionated if that helps the worker move faster
        - preserve the principle that qwen, not you, writes the Rust solution

        Required output:
        - edit files in this workspace to create a concrete candidate revision
        - ensure scripts/vector_db_bench/qwen3_meta/meta_harness_campaign/revisions/{candidate_revision_id}/revision.toml exists and is accurate
        - if you add helper tools or runtime behavior, implement matching support in this workspace
        - write/update harness_notes.md with the architectural bet and intended causal mechanism

        In your final message:
        - summarize the revision in 3-6 bullets
        - list the files you changed
        """
    )


def _run_campaign_in_workspace(*, workspace: Path, revision_id: str, campaign_run_root: Path, mainline_snapshot_dir: str | None, mainline_manifest_path: str | None, args: argparse.Namespace) -> CampaignSummary:
    script_path = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "run_meta_harness_campaign.py"
    cmd = [
        "python3",
        str(script_path),
        "--revision-id",
        revision_id,
        "--run-root",
        str(campaign_run_root),
        "--revisions-root",
        str((workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "meta_harness_campaign" / "revisions").resolve()),
        "--bench-repo",
        str(args.bench_repo.resolve()),
        "--blank-seed-source",
        str(args.blank_seed_source.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--cycles",
        str(args.cycles_per_revision),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--cpu-cores",
        str(args.cpu_cores),
        "--goal-qps",
        str(args.goal_qps),
        "--model-name",
        args.model_name,
        "--base-url",
        args.base_url,
        "--model-id",
        args.model_id,
        "--thinking-mode",
        args.thinking_mode,
        "--reasoning-effort",
        args.reasoning_effort,
        "--api-interval-ms",
        str(args.api_interval_ms),
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if mainline_snapshot_dir:
        cmd.extend(["--mainline-snapshot", mainline_snapshot_dir])
    if mainline_manifest_path:
        cmd.extend(["--mainline-manifest", mainline_manifest_path])
    subprocess.run(cmd, cwd=workspace, check=True, text=True)
    return _load_campaign_summary(campaign_run_root)


def _validate_workspace(workspace: Path) -> None:
    subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            "scripts/vector_db_bench/qwen3_meta/meta_harness_common.py",
            "scripts/vector_db_bench/qwen3_meta/meta_harness_runtime.py",
            "scripts/vector_db_bench/qwen3_meta/run_meta_harness_campaign.py",
            "scripts/vector_db_bench/qwen3_meta/run_meta_harness_campaign_search.py",
        ],
        cwd=workspace,
        check=True,
        text=True,
    )


def _create_worktree(repo_root: Path, path: Path, ref: str) -> None:
    if path.exists():
        shutil.rmtree(path)
    subprocess.run(["git", "worktree", "add", "--detach", str(path), ref], cwd=repo_root, check=True, text=True)


def _remove_worktree(repo_root: Path, path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=repo_root, check=True, text=True)


def _git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()


def _commit_workspace_changes(workspace: Path, revision_id: str) -> str:
    subprocess.run(["git", "add", "scripts/vector_db_bench/qwen3_meta"], cwd=workspace, check=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=workspace, check=True, text=True, capture_output=True)
    if not status.stdout.strip():
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True).stdout.strip()
    subprocess.run(["git", "commit", "-m", f"Campaign meta-harness {revision_id}"], cwd=workspace, check=True, text=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY") or ""
    if not api_key:
        raise SystemExit("API key missing. Pass --api-key or set OPENROUTER_API_KEY in the environment/.env.")
    codex_executable = _require_codex(args.codex_executable)

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "search_results.tsv"
    summary_path = run_root / "summary.json"
    config_path = run_root / "search_config.json"
    worktrees_root = run_root / "worktrees"
    prompts_root = run_root / "codex_prompts"
    outputs_root = run_root / "codex_outputs"
    campaigns_root = run_root / "campaigns"

    write_json(
        config_path,
        {
            "base_revision_id": args.base_revision_id,
            "iterations": args.iterations,
            "cycles_per_revision": args.cycles_per_revision,
            "goal_qps": args.goal_qps,
            "max_tool_calls": args.max_tool_calls,
            "run_root": str(run_root),
            "revisions_root": str(args.revisions_root.resolve()),
            "bench_repo": str(args.bench_repo.resolve()),
            "blank_seed_source": str(args.blank_seed_source.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "codex_enable_web_search": args.codex_enable_web_search,
            "persistent_worker_carryover": True,
        },
    )

    if args.reuse_incumbent_run_root is not None:
        incumbent_run_root = args.reuse_incumbent_run_root.resolve()
        incumbent_summary = _load_campaign_summary(incumbent_run_root)
        incumbent_revision_id = incumbent_summary.revision_id
    else:
        incumbent_run_root = campaigns_root / args.base_revision_id
        incumbent_summary = _run_campaign_in_workspace(
            workspace=REPO_ROOT,
            revision_id=args.base_revision_id,
            campaign_run_root=incumbent_run_root,
            mainline_snapshot_dir=None,
            mainline_manifest_path=None,
            args=args,
        )
        incumbent_revision_id = incumbent_summary.revision_id

    writer = _build_results_writer(results_path)
    current_ref = _git_head(REPO_ROOT)
    try:
        for iteration in range(1, args.iterations + 1):
            candidate_revision_id = f"rev_{iteration:03d}"
            prior_incumbent_revision_id = incumbent_revision_id
            prior_incumbent_best_qps = incumbent_summary.best_qps
            workspace = worktrees_root / candidate_revision_id
            codex_result = None
            candidate_summary = None
            accepted = False
            try:
                _create_worktree(REPO_ROOT, workspace, current_ref)
                _seed_candidate_revision(
                    workspace=workspace,
                    parent_revision_id=incumbent_revision_id,
                    candidate_revision_id=candidate_revision_id,
                )
                context_root = _write_context_bundle(
                    workspace=workspace,
                    search_run_root=run_root,
                    incumbent_revision_id=incumbent_revision_id,
                    incumbent_summary=incumbent_summary,
                    incumbent_run_root=incumbent_run_root,
                    iteration=iteration,
                )
                prompt = _build_codex_prompt(
                    candidate_revision_id=candidate_revision_id,
                    parent_revision_id=incumbent_revision_id,
                    context_root=context_root,
                )
                prompt_path = prompts_root / f"{candidate_revision_id}.md"
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt, encoding="utf-8")
                codex_result = _run_codex_exec(
                    executable=codex_executable,
                    prompt=prompt,
                    cwd=workspace,
                    output_path=outputs_root / f"{candidate_revision_id}_last_message.txt",
                    events_path=outputs_root / f"{candidate_revision_id}_events.jsonl",
                    timeout_seconds=args.codex_timeout_seconds,
                    sandbox=args.codex_sandbox,
                    model=args.codex_model,
                    use_oss=args.codex_oss,
                    local_provider=args.codex_local_provider,
                    enable_web_search=args.codex_enable_web_search,
                )
                write_json(
                    outputs_root / f"{candidate_revision_id}_exec.json",
                    {
                        "argv": codex_result.argv,
                        "returncode": codex_result.returncode,
                        "runtime_seconds": codex_result.runtime_seconds,
                        "stderr": codex_result.stderr,
                        "last_message": codex_result.last_message,
                        "events_path": str(outputs_root / f"{candidate_revision_id}_events.jsonl"),
                    },
                )
                if codex_result.returncode != 0:
                    raise RuntimeError(f"codex exec failed with return code {codex_result.returncode}")

                _validate_workspace(workspace)
                candidate_commit = _commit_workspace_changes(workspace, candidate_revision_id)
                candidate_run_root = campaigns_root / candidate_revision_id
                candidate_summary = _run_campaign_in_workspace(
                    workspace=workspace,
                    revision_id=candidate_revision_id,
                    campaign_run_root=candidate_run_root,
                    mainline_snapshot_dir=incumbent_summary.mainline_snapshot_dir,
                    mainline_manifest_path=incumbent_summary.mainline_manifest_path,
                    args=args,
                )
                accepted = _is_better(candidate_summary, incumbent_summary)
                if accepted:
                    current_ref = candidate_commit
                    incumbent_summary = candidate_summary
                    incumbent_revision_id = candidate_summary.revision_id
                    incumbent_run_root = candidate_run_root
            finally:
                if workspace.exists():
                    _remove_worktree(REPO_ROOT, workspace)

            row = {
                "iteration": iteration,
                "parent_revision": prior_incumbent_revision_id,
                "candidate_revision": candidate_revision_id,
                "accepted": str(accepted).lower(),
                "incumbent_best_qps_before": f"{prior_incumbent_best_qps:.2f}",
                "candidate_best_qps": f"{candidate_summary.best_qps:.2f}" if candidate_summary else "0.00",
                "candidate_best_milestone": candidate_summary.best_milestone if candidate_summary else "none",
                "candidate_goal_reached": str(candidate_summary.goal_reached).lower() if candidate_summary else "false",
                "codex_returncode": codex_result.returncode if codex_result else -1,
                "codex_runtime_seconds": f"{codex_result.runtime_seconds:.2f}" if codex_result else "0.00",
                "candidate_campaign_run_root": candidate_summary.run_root if candidate_summary else "",
                "incumbent_campaign_run_root_after": str(incumbent_run_root),
                "mainline_snapshot_dir_after": incumbent_summary.mainline_snapshot_dir or "",
                "notes": codex_result.last_message[:500] if codex_result else "",
            }
            writer.writerow(row)
            _flush_results_writer(writer)
            write_json(
                summary_path,
                {
                    "current_incumbent_revision_id": incumbent_revision_id,
                    "current_incumbent_run_root": str(incumbent_run_root),
                    "current_incumbent_summary": incumbent_summary.__dict__,
                    "run_root": str(run_root),
                },
            )
    finally:
        _close_results_writer(writer)

    write_json(
        summary_path,
        {
            "current_incumbent_revision_id": incumbent_revision_id,
            "current_incumbent_run_root": str(incumbent_run_root),
            "current_incumbent_summary": incumbent_summary.__dict__,
            "run_root": str(run_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
