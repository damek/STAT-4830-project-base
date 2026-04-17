#!/usr/bin/env python3
"""Run a SOTA-seeking Codex outer loop for vector-db-bench.

This path is separate from the benchmark-faithful best-of-3 search. It uses:
- long-horizon worker episodes
- worker incumbent carryover across revisions
- structured teacher artifacts
- helper tools and online research

The target is a valid 4000+ QPS solution, not minimal benchmark fidelity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from meta_harness_common import DEFAULT_DOTENV_PATH, load_dotenv, write_json


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SOTA_REVISIONS_ROOT = SCRIPT_PATH.with_name("meta_harness_sota") / "revisions"
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_BLANK_SEED_SOURCE = DEFAULT_BENCH_REPO / "skeleton"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_CPU_CORES = "0-3"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_sota_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
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
    "candidate_eval_run_root",
    "incumbent_eval_run_root_after",
    "incumbent_snapshot_dir_after",
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
class SotaEvalSummary:
    revision_id: str
    episodes_requested: int
    episodes_completed: int
    valid_episodes: int
    best_qps: float
    best_episode: int | None
    best_recall: float
    median_valid_qps: float
    mean_valid_qps: float
    mean_elapsed_secs: float
    goal_qps: float
    goal_reached: bool
    best_milestone: str
    incumbent_snapshot_dir: str | None
    run_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--revisions-root", type=Path, default=DEFAULT_SOTA_REVISIONS_ROOT)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--base-revision-id", type=str, default="sota_000")
    parser.add_argument("--reuse-incumbent-eval-run-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)

    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-tool-calls", type=int, default=300)
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


def _load_eval_summary(run_root: Path) -> SotaEvalSummary:
    payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    return SotaEvalSummary(
        revision_id=str(payload["revision_id"]),
        episodes_requested=int(payload["episodes_requested"]),
        episodes_completed=int(payload["episodes_completed"]),
        valid_episodes=int(payload["valid_episodes"]),
        best_qps=float(payload["best_qps"]),
        best_episode=(int(payload["best_episode"]) if payload.get("best_episode") is not None else None),
        best_recall=float(payload["best_recall"]),
        median_valid_qps=float(payload["median_valid_qps"]),
        mean_valid_qps=float(payload["mean_valid_qps"]),
        mean_elapsed_secs=float(payload["mean_elapsed_secs"]),
        goal_qps=float(payload["goal_qps"]),
        goal_reached=bool(payload["goal_reached"]),
        best_milestone=str(payload["best_milestone"]),
        incumbent_snapshot_dir=str(payload.get("incumbent_snapshot_dir")) if payload.get("incumbent_snapshot_dir") else None,
        run_root=str(payload["run_root"]),
    )


def _is_better(candidate: SotaEvalSummary, incumbent: SotaEvalSummary) -> bool:
    left = (
        int(candidate.goal_reached),
        candidate.best_qps,
        candidate.valid_episodes,
        candidate.median_valid_qps,
        -candidate.mean_elapsed_secs,
    )
    right = (
        int(incumbent.goal_reached),
        incumbent.best_qps,
        incumbent.valid_episodes,
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
    revisions_root = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "meta_harness_sota" / "revisions"
    parent_dir = revisions_root / parent_revision_id
    candidate_dir = revisions_root / candidate_revision_id
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_dir, candidate_dir)
    _rewrite_revision_identity(
        candidate_dir / "revision.toml",
        revision_id=candidate_revision_id,
        description=f"Codex-authored SOTA candidate based on {parent_revision_id}",
        notes=f"Candidate generated by the SOTA Meta-Harness search loop from parent {parent_revision_id}.",
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


def _tail_file(src: Path, dst: Path, lines: int = 120) -> None:
    if not src.exists():
        return
    content = src.read_text(encoding="utf-8", errors="replace").splitlines()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(content[-lines:]) + "\n", encoding="utf-8")


def _write_context_bundle(*, workspace: Path, search_run_root: Path, incumbent_revision_id: str, incumbent_summary: SotaEvalSummary, incumbent_eval_root: Path, iteration: int) -> Path:
    context_root = workspace / ".meta_harness_sota_search_context"
    if context_root.exists():
        shutil.rmtree(context_root)
    context_root.mkdir(parents=True, exist_ok=True)

    write_json(
        context_root / "task.json",
        {
            "iteration": iteration,
            "incumbent_revision_id": incumbent_revision_id,
            "incumbent_eval_run_root": str(incumbent_eval_root),
            "objective": {
                "primary": "Get qwen/qwen3-coder-next to a valid 4000+ QPS vector-db-bench solution.",
                "secondary": [
                    "Accelerate architectural progress toward ANN/IVF-style search.",
                    "Keep the best valid worker incumbent alive across episodes.",
                    "Reduce wasted loops and unstable branches.",
                ],
                "search_style": "Design the strongest possible teacher, tooling, and orchestration package short of directly writing the worker Rust solution yourself.",
                "codex_role": [
                    "Research public high-performing solutions and architectural patterns when useful.",
                    "Design detailed strategy, design, and implementation artifacts for Qwen to follow.",
                    "Create helper tools and summaries that are specific to the intended solution class.",
                    "Use the full harness surface creatively to maximize Qwen's chance of reaching 4000+ QPS.",
                ],
            },
            "sota_constraints": {
                "worker_model": "qwen/qwen3-coder-next",
                "worker_writes_solution_code": True,
                "goal_qps": incumbent_summary.goal_qps,
                "worker_state_carryover_allowed": True,
                "long_horizon_episodes": True,
                "online_research_allowed": True,
            },
            "worker_file_scope": {
                "readable": ["src/*", "Cargo.toml", "benchmarks/*", "profiling/*"],
                "writable": ["src/* (except protected files)", "Cargo.toml"],
                "seed_file_guidance": "Mount worker-facing seed files inside src/, e.g. src/strategy.md.",
            },
            "available_harness_surfaces": [
                "extra user messages",
                "teacher files under src/ such as strategy.md, design_spec.md, implementation_plan.md, incumbent_record.md, milestones.md",
                "helper tools via revision-local helper_tools.py",
                "zero-completion retry policy",
                "benchmark/profiling summary tools",
                "checkpoint and restore logic",
                "long-horizon orchestration around the worker incumbent",
            ],
        },
    )
    write_json(context_root / "incumbent_summary.json", incumbent_summary.__dict__)
    _copy_if_exists(incumbent_eval_root / "results.tsv", context_root / "incumbent_results.tsv")
    _copy_if_exists(incumbent_eval_root / "incumbent_snapshot_manifest.json", context_root / "incumbent_snapshot_manifest.json")
    _copy_if_exists(incumbent_eval_root / "incumbent_progress_state.json", context_root / "incumbent_progress_state.json")
    for episode_summary in sorted(incumbent_eval_root.glob("episode_*/episode_summary.json")):
        rel = episode_summary.relative_to(incumbent_eval_root)
        _copy_if_exists(episode_summary, context_root / rel)
    for stderr_log in sorted(incumbent_eval_root.glob("episode_*/run_eval.stderr.log")):
        rel = stderr_log.relative_to(incumbent_eval_root).with_suffix(".stderr.tail.txt")
        _tail_file(stderr_log, context_root / rel)
    for agent_log in sorted(incumbent_eval_root.glob("episode_*/workdir/agent_log.jsonl")):
        rel = agent_log.relative_to(incumbent_eval_root).with_suffix(".tail.jsonl")
        _tail_file(agent_log, context_root / rel, lines=200)
    _copy_if_exists(search_run_root / "search_results.tsv", context_root / "search_results.tsv")
    return context_root


def _build_codex_prompt(*, candidate_revision_id: str, parent_revision_id: str, context_root: Path) -> str:
    return textwrap.dedent(
        f"""\
        You are authoring a new SOTA-seeking Meta-Harness revision for vector-db-bench.

        Objective:
        - get Qwen to a valid 4000+ QPS vector-db-bench solution
        - use the incumbent run record and online research when useful
        - design the strongest teacher, tooling, and orchestration package you can without directly writing the worker Rust solution yourself
        - be hands-on: research strong public solution patterns, design the system around Qwen, and create any teacher artifacts or helper tools that materially improve its odds

        Parent revision:
        - {parent_revision_id}

        Candidate revision to author:
        - {candidate_revision_id}

        Hard framing:
        - Qwen writes the Rust solution
        - you design the optimization environment around Qwen
        - worker state carryover across episodes is allowed and desirable
        - long-horizon search is allowed
        - helper tools, teacher files, summaries, and orchestration changes are all valid
        - creativity is encouraged if it helps Qwen reach a state-of-the-art solution class
        - your job is to help Qwen as much as possible short of directly writing the Rust solution for it

        Strongly preferred revision shapes:
        - structured teacher package under src/ (strategy.md, design_spec.md, implementation_plan.md, incumbent_record.md, milestones.md)
        - helper tools that compress checkpointing, rollback, architecture review, or benchmark policy
        - solution-specific tools or summaries that help Qwen implement and validate the intended ANN / IVF-style architecture
        - orchestration changes that help Qwen escape exact-search local optima and move toward ANN/IVF-like architectures

        Worker file scope:
        - readable: src/*, Cargo.toml, benchmarks/*, profiling/*
        - writable: src/* except protected files, plus Cargo.toml
        - seed worker-facing files under src/

        Record to inspect first:
        - {context_root}/task.json
        - {context_root}/incumbent_summary.json
        - {context_root}/incumbent_results.tsv
        - {context_root}/incumbent_snapshot_manifest.json
        - {context_root}/incumbent_progress_state.json
        - episode summaries, stderr tails, and agent log tails under {context_root}
        - scripts/vector_db_bench/qwen3_meta/META_HARNESS_SOTA_SPEC.md

        Revision standard:
        - pursue state-of-the-art behavior, not small local deltas
        - teacher artifacts may be detailed and directive
        - helper tools may be opinionated if they materially improve Qwen's search process
        - use online research when it helps you design a stronger harness revision
        - feel free to build a very capable, highly specific harness if that is what the intended solution requires
        - preserve the principle that Qwen, not you, writes the final Rust solution

        Required output:
        - edit files in this workspace to create a concrete candidate revision
        - ensure scripts/vector_db_bench/qwen3_meta/meta_harness_sota/revisions/{candidate_revision_id}/revision.toml exists and is accurate
        - if you add helper tools or runtime behavior, implement the matching support in this workspace
        - write/update harness_notes.md with the architectural bet and intended causal mechanism

        In your final message:
        - summarize the revision in 3-6 bullets
        - list the files you changed
        """
    )


def _run_eval_in_workspace(*, workspace: Path, revision_id: str, eval_run_root: Path, incumbent_snapshot_dir: str | None, args: argparse.Namespace) -> SotaEvalSummary:
    script_path = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "run_meta_harness_sota_eval.py"
    cmd = [
        "python3",
        str(script_path),
        "--revision-id",
        revision_id,
        "--run-root",
        str(eval_run_root),
        "--revisions-root",
        str((workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "meta_harness_sota" / "revisions").resolve()),
        "--bench-repo",
        str(args.bench_repo.resolve()),
        "--blank-seed-source",
        str(args.blank_seed_source.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--episodes",
        str(args.episodes),
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
    if incumbent_snapshot_dir:
        cmd.extend(["--worker-incumbent-snapshot", incumbent_snapshot_dir])
    subprocess.run(cmd, cwd=workspace, check=True, text=True)
    return _load_eval_summary(eval_run_root)


def _validate_workspace(workspace: Path) -> None:
    subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            "scripts/vector_db_bench/qwen3_meta/meta_harness_common.py",
            "scripts/vector_db_bench/qwen3_meta/meta_harness_runtime.py",
            "scripts/vector_db_bench/qwen3_meta/run_meta_harness_sota_eval.py",
            "scripts/vector_db_bench/qwen3_meta/run_meta_harness_sota_search.py",
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
    subprocess.run(["git", "commit", "-m", f"SOTA meta-harness {revision_id}"], cwd=workspace, check=True, text=True)
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
    evals_root = run_root / "evaluations"

    write_json(
        config_path,
        {
            "base_revision_id": args.base_revision_id,
            "iterations": args.iterations,
            "episodes": args.episodes,
            "goal_qps": args.goal_qps,
            "max_tool_calls": args.max_tool_calls,
            "run_root": str(run_root),
            "revisions_root": str(args.revisions_root.resolve()),
            "bench_repo": str(args.bench_repo.resolve()),
            "blank_seed_source": str(args.blank_seed_source.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "codex_enable_web_search": args.codex_enable_web_search,
        },
    )

    if args.reuse_incumbent_eval_run_root is not None:
        incumbent_eval_root = args.reuse_incumbent_eval_run_root.resolve()
        incumbent_summary = _load_eval_summary(incumbent_eval_root)
        incumbent_revision_id = incumbent_summary.revision_id
    else:
        incumbent_eval_root = evals_root / args.base_revision_id
        incumbent_summary = _run_eval_in_workspace(
            workspace=REPO_ROOT,
            revision_id=args.base_revision_id,
            eval_run_root=incumbent_eval_root,
            incumbent_snapshot_dir=None,
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
            context_root = None
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
                    incumbent_eval_root=incumbent_eval_root,
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
                candidate_eval_root = evals_root / candidate_revision_id
                candidate_summary = _run_eval_in_workspace(
                    workspace=workspace,
                    revision_id=candidate_revision_id,
                    eval_run_root=candidate_eval_root,
                    incumbent_snapshot_dir=incumbent_summary.incumbent_snapshot_dir,
                    args=args,
                )
                accepted = _is_better(candidate_summary, incumbent_summary)
                if accepted:
                    current_ref = candidate_commit
                    incumbent_summary = candidate_summary
                    incumbent_revision_id = candidate_summary.revision_id
                    incumbent_eval_root = candidate_eval_root
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
                "candidate_eval_run_root": candidate_summary.run_root if candidate_summary else "",
                "incumbent_eval_run_root_after": str(incumbent_eval_root),
                "incumbent_snapshot_dir_after": incumbent_summary.incumbent_snapshot_dir or "",
                "notes": codex_result.last_message[:500] if codex_result else "",
            }
            writer.writerow(row)
            _flush_results_writer(writer)
            write_json(
                summary_path,
                {
                    "current_incumbent_revision_id": incumbent_revision_id,
                    "current_incumbent_eval_run_root": str(incumbent_eval_root),
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
            "current_incumbent_eval_run_root": str(incumbent_eval_root),
            "current_incumbent_summary": incumbent_summary.__dict__,
            "run_root": str(run_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
