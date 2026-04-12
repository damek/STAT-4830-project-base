#!/usr/bin/env python3
"""Generate vector-db-bench SFT data using Codex CLI as a teacher model.

This script reuses the existing worker prompts from generate_rl_prompts.py,
runs `codex exec` for each prompt, benchmarks the resulting candidate with the
same local evaluator used by the older Codex harness, and writes accepted
(prompt, completion) pairs as chat-format JSONL for modal_sft_train.py.

Typical workflow:
  python scripts/vector_db_bench/generate_rl_prompts.py \
      --bench-repo third_party/vector-db-bench

  python scripts/vector_db_bench/codex_sft_rollouts.py \
      --bench-repo third_party/vector-db-bench \
      --prompts-path data/vector_db_bench/rl_prompts.jsonl \
      --attempts-per-prompt 1 \
      --only-valid \
      --output data/vector_db_bench/codex_teacher_sft.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from codex_cli_harness import (
    _bootstrap_seed_surface,
    _evaluate_candidate,
    _json_schema_for_worker_output,
    _parse_worker_output,
    _prepare_config,
    _run_codex_exec,
    _score_result,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "data" / "vector_db_bench" / "rl_prompts.jsonl"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "codex_teacher_runs"

RESULT_COLUMNS = [
    "prompt_index",
    "attempt",
    "brief_title",
    "brief_family",
    "status",
    "accepted",
    "build_ok",
    "runtime_ok",
    "valid",
    "recall_passed",
    "anti_cheat_passed",
    "qps",
    "recall",
    "score",
    "failure_type",
    "runtime_seconds",
    "change_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True, help="Path to a local clone of KCORES/vector-db-bench")
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS, help=f"Prompt JSONL from generate_rl_prompts.py (default: {DEFAULT_PROMPTS})")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output", type=Path, default=None, help="Output chat-format JSONL; defaults to <run-dir>/sft_data.jsonl")
    parser.add_argument("--attempts-per-prompt", type=int, default=1, help="How many Codex samples to request per prompt")
    parser.add_argument("--max-prompts", type=int, default=0, help="Optional cap for smoke tests; 0 means all prompts")
    parser.add_argument(
        "--only-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only recall+anti-cheat-valid benchmark results (default: true)",
    )
    parser.add_argument("--min-qps", type=float, default=0.0, help="Reject accepted samples below this QPS threshold")
    parser.add_argument(
        "--eval-phase",
        choices=("proxy", "strict"),
        default="proxy",
        help="Benchmark split to use when filtering teacher rollouts",
    )

    parser.add_argument("--recall-threshold", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proxy-max-queries", type=int, default=2000)
    parser.add_argument("--strict-max-queries", type=int, default=0, help="0 means use all queries")
    parser.add_argument("--build-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--server-bin-name", type=str, default="vector-db-skeleton")
    parser.add_argument("--benchmark-bin-name", type=str, default="vector-db-benchmark")
    parser.add_argument("--cpu-cores", type=str, default="", help="Optional taskset CPU core list, e.g. 0-3")
    parser.add_argument("--base-vectors", type=Path, default=None)
    parser.add_argument("--query-vectors", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)

    parser.add_argument("--codex-executable", type=str, default="codex")
    parser.add_argument("--codex-model", type=str, default="")
    parser.add_argument(
        "--codex-oss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Codex CLI with a local OSS provider.",
    )
    parser.add_argument("--codex-local-provider", choices=("ollama", "lmstudio"), default="")
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--codex-timeout-seconds", type=int, default=60 * 15)
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibility flag for shared config helpers; unused here.",
    )
    return parser.parse_args()


def _build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
    writer.writeheader()
    setattr(writer, "_handle", handle)
    return writer


def _close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def _prompt_text(record: dict[str, Any]) -> str:
    prompt_messages = record.get("prompt")
    if not isinstance(prompt_messages, list) or not prompt_messages:
        raise ValueError("prompt record missing prompt messages")
    last = prompt_messages[-1]
    if not isinstance(last, dict):
        raise ValueError("prompt record has malformed last prompt message")
    text = str(last.get("content", "")).strip()
    if not text:
        raise ValueError("prompt record has empty prompt content")
    return text


def _accepted(eval_result: Any, *, only_valid: bool, min_qps: float) -> bool:
    if only_valid:
        return bool(eval_result.valid and eval_result.qps >= min_qps)
    return bool(eval_result.build_ok and eval_result.runtime_ok and eval_result.qps >= min_qps)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else args.run_dir / "sft_data.jsonl"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.prompts_path.exists():
        raise SystemExit(
            f"Prompts file not found: {args.prompts_path}\n"
            "Run: python scripts/vector_db_bench/generate_rl_prompts.py --bench-repo ..."
        )

    config = _prepare_config(args)
    incumbent_files = _bootstrap_seed_surface(config.skeleton_dir)
    schema_path = args.run_dir / "worker_output_schema.json"
    _write_json(
        schema_path,
        _json_schema_for_worker_output(incumbent_paths=sorted(incumbent_files.keys())),
    )

    raw_records = [
        json.loads(line)
        for line in args.prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_records:
        raise SystemExit(f"Prompt file is empty: {args.prompts_path}")
    if args.max_prompts > 0:
        raw_records = raw_records[: args.max_prompts]

    results_writer = _build_results_writer(args.run_dir / "results.tsv")
    sft_examples: list[dict[str, Any]] = []
    accepted_count = 0
    total_attempts = 0

    try:
        for prompt_index, record in enumerate(raw_records, start=1):
            prompt_dir = args.run_dir / f"prompt_{prompt_index:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_text = _prompt_text(record)
            (prompt_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

            for attempt in range(1, args.attempts_per_prompt + 1):
                total_attempts += 1
                attempt_dir = prompt_dir / f"attempt_{attempt:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)

                exec_result = _run_codex_exec(
                    executable=config.codex_executable,
                    prompt=prompt_text,
                    cwd=config.bench_repo,
                    output_path=attempt_dir / "worker_output.json",
                    timeout_seconds=config.codex_timeout_seconds,
                    sandbox=config.codex_sandbox,
                    model=config.codex_model,
                    use_oss=config.codex_oss,
                    local_provider=config.codex_local_provider,
                    schema_path=schema_path,
                )
                (attempt_dir / "stdout.log").write_text(exec_result.stdout, encoding="utf-8")
                (attempt_dir / "stderr.log").write_text(exec_result.stderr, encoding="utf-8")

                result_row: dict[str, Any] = {
                    "prompt_index": prompt_index,
                    "attempt": attempt,
                    "brief_title": record.get("brief_title", ""),
                    "brief_family": record.get("brief_family", ""),
                    "status": "proposal_error",
                    "accepted": False,
                    "build_ok": False,
                    "runtime_ok": False,
                    "valid": False,
                    "recall_passed": False,
                    "anti_cheat_passed": False,
                    "qps": 0.0,
                    "recall": 0.0,
                    "score": 0.0,
                    "failure_type": "proposal_error",
                    "runtime_seconds": exec_result.runtime_seconds,
                    "change_summary": "",
                }

                if exec_result.returncode != 0:
                    result_row["change_summary"] = (exec_result.stderr or exec_result.stdout).strip()[:1000]
                    results_writer.writerow(result_row)
                    continue

                try:
                    summary, candidate_files = _parse_worker_output(
                        exec_result.last_message,
                        incumbent_files=incumbent_files,
                    )
                except Exception as exc:
                    result_row["failure_type"] = "parse_error"
                    result_row["change_summary"] = str(exc)
                    results_writer.writerow(result_row)
                    continue

                eval_result = _evaluate_candidate(
                    candidate_files=candidate_files,
                    workspace_dir=attempt_dir / "workspace",
                    eval_dir=attempt_dir / f"{args.eval_phase}_eval",
                    config=config,
                    inputs=config.proxy_inputs if args.eval_phase == "proxy" else config.strict_inputs,
                )
                accepted = _accepted(eval_result, only_valid=args.only_valid, min_qps=args.min_qps)
                result_row.update(
                    {
                        "status": "accepted" if accepted else "discarded",
                        "accepted": accepted,
                        "build_ok": eval_result.build_ok,
                        "runtime_ok": eval_result.runtime_ok,
                        "valid": eval_result.valid,
                        "recall_passed": eval_result.recall_passed,
                        "anti_cheat_passed": eval_result.anti_cheat_passed,
                        "qps": eval_result.qps,
                        "recall": eval_result.recall,
                        "score": _score_result(eval_result),
                        "failure_type": eval_result.failure_type,
                        "runtime_seconds": exec_result.runtime_seconds + eval_result.runtime_seconds,
                        "change_summary": summary,
                    }
                )
                results_writer.writerow(result_row)

                if not accepted:
                    continue

                accepted_count += 1
                clean_output = exec_result.last_message.strip()
                sft_examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt_text},
                            {"role": "assistant", "content": clean_output},
                        ],
                        "metadata": {
                            "teacher": "codex-exec",
                            "prompt_index": prompt_index,
                            "attempt": attempt,
                            "brief_title": record.get("brief_title", ""),
                            "brief_family": record.get("brief_family", ""),
                            "eval_phase": args.eval_phase,
                            "only_valid": args.only_valid,
                            "build_ok": eval_result.build_ok,
                            "runtime_ok": eval_result.runtime_ok,
                            "valid": eval_result.valid,
                            "recall_passed": eval_result.recall_passed,
                            "anti_cheat_passed": eval_result.anti_cheat_passed,
                            "qps": eval_result.qps,
                            "recall": eval_result.recall,
                            "summary": summary,
                        },
                    }
                )
    finally:
        _close_results_writer(results_writer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in sft_examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary = {
        "prompts_path": str(args.prompts_path),
        "bench_repo": str(config.bench_repo),
        "run_dir": str(args.run_dir),
        "output_path": str(output_path),
        "eval_phase": args.eval_phase,
        "only_valid": args.only_valid,
        "min_qps": args.min_qps,
        "attempts_per_prompt": args.attempts_per_prompt,
        "prompts_evaluated": len(raw_records),
        "attempts_total": total_attempts,
        "examples_written": len(sft_examples),
        "accept_rate": (len(sft_examples) / total_attempts) if total_attempts else 0.0,
    }
    _write_json(args.run_dir / "summary.json", summary)

    print(f"[codex_sft_rollouts] wrote {len(sft_examples)} examples → {output_path}")
    print(f"[codex_sft_rollouts] accepted {accepted_count}/{total_attempts} attempts")
    print(f"[codex_sft_rollouts] summary → {args.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
