#!/usr/bin/env python3
"""Run a Codex-authored Meta-Harness search loop for vector-db-bench.

The loop is:
1. start from a harness incumbent revision (default: h0)
2. have `codex exec` author a candidate harness revision in an isolated worktree
3. evaluate that candidate on 3 fresh official-style attempts
4. accept or reject the candidate against the incumbent
5. repeat

Important: the 3 benchmark attempts are intentionally serial by default. Running
attempts in parallel on one benchmark host would distort the benchmark-faithful
`CPU_CORES=0-3` measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from meta_harness_common import DEFAULT_DOTENV_PATH, DEFAULT_REVISIONS_ROOT, load_dotenv, write_json


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_BENCH_REPO = REPO_ROOT / "third_party" / "vector-db-bench"
DEFAULT_BLANK_SEED_SOURCE = DEFAULT_BENCH_REPO / "skeleton"
DEFAULT_DATA_DIR = DEFAULT_BENCH_REPO / "data"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta" / "meta_harness_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_CPU_CORES = "0-3"
SEARCH_COLUMNS = [
    "iteration",
    "parent_revision",
    "candidate_revision",
    "accepted",
    "incumbent_valid_before",
    "incumbent_best_qps_before",
    "candidate_valid",
    "candidate_best_qps",
    "candidate_median_qps",
    "candidate_mean_elapsed_secs",
    "codex_returncode",
    "codex_runtime_seconds",
    "candidate_eval_run_root",
    "incumbent_eval_run_root_after",
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
class EvalSummary:
    revision_id: str
    attempts_requested: int
    attempts_completed: int
    valid_attempts: int
    best_qps: float
    best_attempt: int | None
    best_recall: float
    median_valid_qps: float
    mean_valid_qps: float
    mean_elapsed_secs: float
    run_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--revisions-root", type=Path, default=DEFAULT_REVISIONS_ROOT)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--base-revision-id", type=str, default="h0")
    parser.add_argument("--reuse-incumbent-eval-run-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)

    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--blank-seed-source", type=Path, default=DEFAULT_BLANK_SEED_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-tool-calls", type=int, default=50)
    parser.add_argument("--cpu-cores", type=str, default=DEFAULT_CPU_CORES)

    parser.add_argument("--model-name", type=str, default="qwen3-coder-next")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-id", type=str, default="qwen/qwen3-coder-next")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--thinking-mode", type=str, default="openrouter-openai")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-interval-ms", type=int, default=0)

    parser.add_argument("--codex-executable", type=str, default="codex")
    parser.add_argument("--codex-timeout-seconds", type=int, default=60 * 15)
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--codex-model", type=str, default="")
    parser.add_argument("--codex-oss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--codex-local-provider", choices=("", "ollama", "lmstudio"), default="")
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


def _git(repo: Path, *args: str, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=capture_output,
        check=True,
    )


def _current_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _run_codex_exec(
    *,
    executable: str,
    prompt: str,
    cwd: Path,
    output_path: Path,
    timeout_seconds: int,
    sandbox: str,
    model: str,
    use_oss: bool,
    local_provider: str,
) -> CodexExecResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        executable,
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "--output-last-message",
        str(output_path),
    ]
    if model:
        argv.extend(["-m", model])
    if use_oss:
        argv.append("--oss")
    if local_provider:
        argv.extend(["--local-provider", local_provider])
    argv.append("-")

    started_at = time.time()
    proc = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    return CodexExecResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        last_message=last_message,
        runtime_seconds=time.time() - started_at,
    )


def _load_eval_summary(run_root: Path) -> EvalSummary:
    payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    return EvalSummary(
        revision_id=str(payload["revision_id"]),
        attempts_requested=int(payload["attempts_requested"]),
        attempts_completed=int(payload["attempts_completed"]),
        valid_attempts=int(payload["valid_attempts"]),
        best_qps=float(payload["best_qps"]),
        best_attempt=(int(payload["best_attempt"]) if payload.get("best_attempt") is not None else None),
        best_recall=float(payload["best_recall"]),
        median_valid_qps=float(payload["median_valid_qps"]),
        mean_valid_qps=float(payload["mean_valid_qps"]),
        mean_elapsed_secs=float(payload["mean_elapsed_secs"]),
        run_root=str(payload["run_root"]),
    )


def _is_better(candidate: EvalSummary, incumbent: EvalSummary) -> bool:
    left = (
        candidate.valid_attempts,
        candidate.best_qps,
        candidate.median_valid_qps,
        -candidate.mean_elapsed_secs,
    )
    right = (
        incumbent.valid_attempts,
        incumbent.best_qps,
        incumbent.median_valid_qps,
        -incumbent.mean_elapsed_secs,
    )
    return left > right


def _write_revision_toml(*, revision_dir: Path, revision_id: str, description: str, notes: str, attempts_per_eval: int) -> None:
    payload = textwrap.dedent(
        f"""\
        id = {revision_id!r}
        description = {description!r}
        attempts_per_eval = {attempts_per_eval}
        official_tools_only = true

        extra_user_messages = []
        added_helper_tools = []
        seed_files_dir = ""
        notes = {notes!r}
        """
    )
    (revision_dir / "revision.toml").write_text(payload, encoding="utf-8")


def _seed_candidate_revision(*, workspace: Path, parent_revision_id: str, candidate_revision_id: str, attempts: int) -> Path:
    revisions_root = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "meta_harness" / "revisions"
    parent_dir = revisions_root / parent_revision_id
    candidate_dir = revisions_root / candidate_revision_id
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_dir, candidate_dir)
    _write_revision_toml(
        revision_dir=candidate_dir,
        revision_id=candidate_revision_id,
        description=f"Codex-authored candidate based on {parent_revision_id}",
        notes=f"Candidate generated by the Meta-Harness search loop from parent {parent_revision_id}.",
        attempts_per_eval=attempts,
    )
    notes_path = candidate_dir / "harness_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Harness Notes\n\n- hypothesis: fill this in\n- intended changes: fill this in\n",
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


def _write_context_bundle(
    *,
    workspace: Path,
    search_run_root: Path,
    incumbent_revision_id: str,
    incumbent_summary: EvalSummary,
    incumbent_eval_root: Path,
    iteration: int,
) -> Path:
    context_root = workspace / ".meta_harness_search_context"
    if context_root.exists():
        shutil.rmtree(context_root)
    context_root.mkdir(parents=True, exist_ok=True)

    write_json(
        context_root / "task.json",
        {
            "iteration": iteration,
            "incumbent_revision_id": incumbent_revision_id,
            "incumbent_eval_run_root": str(incumbent_eval_root),
            "benchmark_faithful_constraints": {
                "official_tools_unchanged": True,
                "fresh_blank_scaffold_every_attempt": True,
                "max_tool_calls": 50,
                "cpu_cores": "0-3",
                "recall_threshold": 0.95,
            },
        },
    )
    write_json(context_root / "incumbent_summary.json", incumbent_summary.__dict__)
    _copy_if_exists(incumbent_eval_root / "results.tsv", context_root / "incumbent_results.tsv")
    for attempt_summary in sorted(incumbent_eval_root.glob("attempt_*/attempt_summary.json")):
        rel = attempt_summary.relative_to(incumbent_eval_root)
        _copy_if_exists(attempt_summary, context_root / rel)
    for stderr_log in sorted(incumbent_eval_root.glob("attempt_*/run_eval.stderr.log")):
        rel = stderr_log.relative_to(incumbent_eval_root).with_suffix(".stderr.tail.txt")
        _tail_file(stderr_log, context_root / rel)
    _copy_if_exists(search_run_root / "search_results.tsv", context_root / "search_results.tsv")
    return context_root


def _build_codex_prompt(*, candidate_revision_id: str, parent_revision_id: str, context_root: Path) -> str:
    return textwrap.dedent(
        f"""\
        You are authoring a new Meta-Harness revision for vector-db-bench.

        Goal:
        - improve Qwen's fresh-start benchmark performance under the official 50-tool-call setup
        - do not carry worker solution code across attempts
        - keep the official benchmark tools present and unchanged
        - you may add helper tools only if you also implement runtime support for them

        Parent revision:
        - {parent_revision_id}

        Candidate revision to author:
        - {candidate_revision_id}

        Context to inspect first:
        - {context_root}/task.json
        - {context_root}/incumbent_summary.json
        - {context_root}/incumbent_results.tsv
        - any attempt summaries and stderr tails under {context_root}
        - scripts/vector_db_bench/qwen3_meta/META_HARNESS_ADAPTATION.md
        - scripts/vector_db_bench/qwen3_meta/META_HARNESS_REV_000.md

        Constraints:
        - preserve benchmark-faithful worker settings: blank scaffold, official system prompt, official opening user message, official tool budget, CPU_CORES=0-3
        - do not pre-bake solution code carryover
        - keep changes focused and defensible

        Required output:
        - edit files in this workspace to create a concrete candidate revision
        - at minimum, ensure scripts/vector_db_bench/qwen3_meta/meta_harness/revisions/{candidate_revision_id}/revision.toml exists and is accurate
        - if you change runtime behavior, keep it within the allowed harness surface
        - write/update harness_notes.md in the candidate revision with the hypothesis and intended causal mechanism

        In your final message:
        - summarize the revision in 3-6 bullets
        - list the files you changed
        """
    )


def _run_eval_in_workspace(*, workspace: Path, revision_id: str, eval_run_root: Path, args: argparse.Namespace) -> EvalSummary:
    script_path = workspace / "scripts" / "vector_db_bench" / "qwen3_meta" / "run_meta_harness_eval.py"
    cmd = [
        "python3",
        str(script_path),
        "--revision-id",
        revision_id,
        "--run-root",
        str(eval_run_root),
        "--bench-repo",
        str(args.bench_repo.resolve()),
        "--blank-seed-source",
        str(args.blank_seed_source.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--attempts",
        str(args.attempts),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--cpu-cores",
        str(args.cpu_cores),
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
    subprocess.run(cmd, cwd=workspace, check=True, text=True)
    return _load_eval_summary(eval_run_root)


def _validate_workspace(workspace: Path) -> None:
    subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            "scripts/vector_db_bench/qwen3_meta/meta_harness_common.py",
            "scripts/vector_db_bench/qwen3_meta/run_meta_harness_eval.py",
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


def _commit_workspace_changes(workspace: Path, revision_id: str) -> str:
    subprocess.run(["git", "add", "scripts/vector_db_bench/qwen3_meta"], cwd=workspace, check=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=workspace, check=True, text=True, capture_output=True)
    if not status.stdout.strip():
        return _git(workspace, "rev-parse", "HEAD").stdout.strip()
    subprocess.run(["git", "commit", "-m", f"Meta-harness {revision_id}"], cwd=workspace, check=True, text=True)
    return _git(workspace, "rev-parse", "HEAD").stdout.strip()


def _search_row(
    *,
    iteration: int,
    parent_revision: str,
    candidate_revision: str,
    accepted: bool,
    incumbent_summary_before: EvalSummary,
    candidate_summary: EvalSummary,
    codex_result: CodexExecResult,
    incumbent_eval_root_after: Path,
    notes: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "parent_revision": parent_revision,
        "candidate_revision": candidate_revision,
        "accepted": str(accepted).lower(),
        "incumbent_valid_before": incumbent_summary_before.valid_attempts,
        "incumbent_best_qps_before": f"{incumbent_summary_before.best_qps:.2f}",
        "candidate_valid": candidate_summary.valid_attempts,
        "candidate_best_qps": f"{candidate_summary.best_qps:.2f}",
        "candidate_median_qps": f"{candidate_summary.median_valid_qps:.2f}",
        "candidate_mean_elapsed_secs": f"{candidate_summary.mean_elapsed_secs:.2f}",
        "codex_returncode": codex_result.returncode,
        "codex_runtime_seconds": f"{codex_result.runtime_seconds:.2f}",
        "candidate_eval_run_root": candidate_summary.run_root,
        "incumbent_eval_run_root_after": str(incumbent_eval_root_after),
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv_path)
    _require_codex(args.codex_executable)

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.attempts <= 0:
        raise SystemExit("--attempts must be positive")

    repo_root = REPO_ROOT
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    worktrees_root = run_root / "worktrees"
    evals_root = run_root / "evaluations"
    prompts_root = run_root / "codex_prompts"
    codex_outputs_root = run_root / "codex_outputs"
    results_path = run_root / "search_results.tsv"
    summary_path = run_root / "summary.json"

    incumbent_branch = f"meta-harness/{run_root.name}-incumbent"
    head_commit = _current_head(repo_root)
    subprocess.run(["git", "branch", "-f", incumbent_branch, head_commit], cwd=repo_root, check=True, text=True)
    incumbent_ref = incumbent_branch
    incumbent_revision_id = args.base_revision_id

    if args.reuse_incumbent_eval_run_root is not None:
        incumbent_eval_root = args.reuse_incumbent_eval_run_root.resolve()
        incumbent_summary = _load_eval_summary(incumbent_eval_root)
    else:
        base_eval_root = evals_root / f"{incumbent_revision_id}_base"
        incumbent_summary = _run_eval_in_workspace(
            workspace=repo_root,
            revision_id=incumbent_revision_id,
            eval_run_root=base_eval_root,
            args=args,
        )
        incumbent_eval_root = base_eval_root

    write_json(
        run_root / "search_config.json",
        {
            "base_revision_id": args.base_revision_id,
            "reuse_incumbent_eval_run_root": str(args.reuse_incumbent_eval_run_root) if args.reuse_incumbent_eval_run_root else None,
            "iterations": args.iterations,
            "attempts": args.attempts,
            "bench_repo": str(args.bench_repo.resolve()),
            "blank_seed_source": str(args.blank_seed_source.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "cpu_cores": args.cpu_cores,
            "incumbent_branch": incumbent_branch,
            "initial_incumbent_summary": incumbent_summary.__dict__,
        },
    )

    writer = _build_results_writer(results_path)
    rows: list[dict[str, Any]] = []
    try:
        for iteration in range(1, args.iterations + 1):
            candidate_revision_id = f"rev_{iteration:03d}"
            candidate_worktree = worktrees_root / candidate_revision_id
            incumbent_revision_before = incumbent_revision_id
            incumbent_summary_before = incumbent_summary
            _create_worktree(repo_root, candidate_worktree, incumbent_ref)
            try:
                _seed_candidate_revision(
                    workspace=candidate_worktree,
                    parent_revision_id=incumbent_revision_id,
                    candidate_revision_id=candidate_revision_id,
                    attempts=args.attempts,
                )
                context_root = _write_context_bundle(
                    workspace=candidate_worktree,
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
                output_path = codex_outputs_root / f"{candidate_revision_id}_last_message.txt"
                codex_result = _run_codex_exec(
                    executable=args.codex_executable,
                    prompt=prompt,
                    cwd=candidate_worktree,
                    output_path=output_path,
                    timeout_seconds=args.codex_timeout_seconds,
                    sandbox=args.codex_sandbox,
                    model=args.codex_model,
                    use_oss=args.codex_oss,
                    local_provider=args.codex_local_provider,
                )
                write_json(
                    codex_outputs_root / f"{candidate_revision_id}_exec.json",
                    {
                        "argv": codex_result.argv,
                        "returncode": codex_result.returncode,
                        "stdout": codex_result.stdout,
                        "stderr": codex_result.stderr,
                        "last_message": codex_result.last_message,
                        "runtime_seconds": codex_result.runtime_seconds,
                    },
                )
                if codex_result.returncode != 0:
                    raise RuntimeError(f"codex exec failed for {candidate_revision_id} with code {codex_result.returncode}")

                _validate_workspace(candidate_worktree)
                candidate_eval_root = evals_root / candidate_revision_id
                candidate_summary = _run_eval_in_workspace(
                    workspace=candidate_worktree,
                    revision_id=candidate_revision_id,
                    eval_run_root=candidate_eval_root,
                    args=args,
                )
                accepted = _is_better(candidate_summary, incumbent_summary)
                if accepted:
                    new_commit = _commit_workspace_changes(candidate_worktree, candidate_revision_id)
                    subprocess.run(["git", "branch", "-f", incumbent_branch, new_commit], cwd=repo_root, check=True, text=True)
                    incumbent_ref = incumbent_branch
                    incumbent_revision_id = candidate_revision_id
                    incumbent_summary = candidate_summary
                    incumbent_eval_root = candidate_eval_root
                    notes = "accepted"
                else:
                    notes = "rejected"

                row = _search_row(
                    iteration=iteration,
                    parent_revision=incumbent_revision_before,
                    candidate_revision=candidate_revision_id,
                    accepted=accepted,
                    incumbent_summary_before=incumbent_summary_before,
                    candidate_summary=candidate_summary,
                    codex_result=codex_result,
                    incumbent_eval_root_after=incumbent_eval_root,
                    notes=notes,
                )
                rows.append(row)
                writer.writerow(row)
                _flush_results_writer(writer)
                write_json(
                    summary_path,
                    {
                        "iterations_completed": len(rows),
                        "incumbent_revision_id": incumbent_revision_id,
                        "incumbent_summary": incumbent_summary.__dict__,
                        "run_root": str(run_root),
                    },
                )
            finally:
                _remove_worktree(repo_root, candidate_worktree)
    finally:
        _close_results_writer(writer)

    write_json(
        summary_path,
        {
            "iterations_completed": len(rows),
            "incumbent_revision_id": incumbent_revision_id,
            "incumbent_summary": incumbent_summary.__dict__,
            "run_root": str(run_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
