#!/usr/bin/env python3
"""Evaluate a no-tools Qwen baseline on vector-db-bench worker prompts.

This script sends the existing RL/worker prompts directly to a warm OpenAI-
compatible endpoint, expects the same JSON `{"summary": ..., "files": ...}`
contract as the Codex harness, and benchmarks the returned Rust candidate with
the trusted local vector-db-bench evaluator.

The intended use is:
1. keep a warm model server running (for example the deployed Qwen2.5-32B
   endpoint),
2. run this script locally or on the Linux benchmark host,
3. measure format validity, compile rate, proxy-valid rate, and strict-valid
   rate before adding any tool use.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex_cli_harness import (
    BenchEvalResult,
    RunConfig,
    _bootstrap_seed_surface,
    _ensure_benchmark_binary,
    _evaluate_candidate,
    _parse_server_port,
    _parse_worker_output,
    _prepare_inputs,
    _score_result,
    _surface_sha,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = REPO_ROOT / "data" / "vector_db_bench" / "rl_prompts.jsonl"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "qwen3_meta_runs"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-coder-next"
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
PROVIDER_PRESETS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-coder-next",
        "model_max_context": 262_144,
        "env_key": "OPENROUTER_API_KEY",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-coder-next",
        "model_max_context": 262_144,
        "env_key": "DASHSCOPE_API_KEY",
    },
    "custom": {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "model_max_context": 262_144,
        "env_key": "",
    },
}

RESULT_COLUMNS = [
    "prompt_index",
    "attempt",
    "brief_title",
    "brief_family",
    "phase",
    "status",
    "candidate_sha256",
    "valid",
    "recall_passed",
    "anti_cheat_passed",
    "build_ok",
    "runtime_ok",
    "qps",
    "recall",
    "avg_latency_ms",
    "p95_latency_ms",
    "score",
    "failure_type",
    "generation_seconds",
    "eval_runtime_seconds",
    "prompt_tokens",
    "completion_tokens",
    "change_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True, help="Path to a local clone of KCORES/vector-db-bench")
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / f"no_tools_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_PRESETS.keys()),
        default="openrouter",
        help="Inference provider preset for base URL, model, context, and auth env lookup.",
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL ending in /v1")
    parser.add_argument("--api-key", type=str, default="DUMMY")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-max-context",
        type=int,
        default=8192,
        help="Maximum total tokens supported by the serving model endpoint.",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all prompts")
    parser.add_argument("--attempts-per-prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=3072)
    parser.add_argument(
        "--completion-reserve-tokens",
        type=int,
        default=32,
        help="Reserve this many tokens under the model context limit when adapting completion length.",
    )
    parser.add_argument("--request-timeout-seconds", type=int, default=60 * 20)
    parser.add_argument(
        "--strict-on-proxy-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run strict evaluation for proxy-valid candidates (default: true).",
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
    return parser.parse_args()


def load_dotenv(path: Path) -> list[str]:
    if not path.exists():
        return []
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


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


def _flush_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.flush()


def _load_prompt_records(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_prompts > 0:
        return records[:max_prompts]
    return records


def _prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt record missing prompt messages")
    cleaned: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("prompt record contains a non-object message")
        role = str(item.get("role", "")).strip()
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role in prompt record: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("prompt record message has empty content")
        cleaned.append({"role": role, "content": content})
    return cleaned


def _required_incumbent_subset(record: dict[str, Any], incumbent_files: dict[str, str]) -> dict[str, str]:
    raw_paths = record.get("incumbent_files")
    if not isinstance(raw_paths, list) or not raw_paths:
        return dict(incumbent_files)
    subset: dict[str, str] = {}
    for raw in raw_paths:
        rel = str(raw)
        if rel not in incumbent_files:
            raise ValueError(f"prompt record referenced unknown incumbent path: {rel}")
        subset[rel] = incumbent_files[rel]
    return subset


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("model returned empty text")
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        payload = json.loads(fenced.group(1))
        if isinstance(payload, dict):
            return payload

    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("model output did not contain a JSON object")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("model output JSON root must be an object")
    return payload


def _prepare_eval_config(args: argparse.Namespace) -> RunConfig:
    args.run_dir = args.run_dir.resolve()
    bench_repo = args.bench_repo.resolve()
    skeleton_dir = bench_repo / "skeleton"
    benchmark_dir = bench_repo / "benchmark"
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"missing skeleton directory: {skeleton_dir}")
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"missing benchmark directory: {benchmark_dir}")
    benchmark_bin = _ensure_benchmark_binary(benchmark_dir, args.benchmark_bin_name, args.build_timeout_seconds)
    proxy_inputs, strict_inputs = _prepare_inputs(args, args.run_dir)
    return RunConfig(
        bench_repo=bench_repo,
        skeleton_dir=skeleton_dir,
        benchmark_dir=benchmark_dir,
        benchmark_bin=benchmark_bin,
        server_bin_name=args.server_bin_name,
        server_url=args.server_url,
        server_port=_parse_server_port(args.server_url),
        cpu_cores=args.cpu_cores,
        build_timeout_seconds=args.build_timeout_seconds,
        benchmark_timeout_seconds=args.benchmark_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        proxy_inputs=proxy_inputs,
        strict_inputs=strict_inputs,
        concurrency=args.concurrency,
        warmup=args.warmup,
        recall_threshold=args.recall_threshold,
        seed=args.seed,
        codex_executable="",
        codex_timeout_seconds=0,
        codex_sandbox="workspace-write",
        codex_model="",
        codex_oss=False,
        codex_local_provider="",
        modal_show_output=False,
    )


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any] | None, float]:
    started_at = time.time()
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
        )
        message = completion.choices[0].message
        text = message.content or ""
        usage = completion.usage.model_dump() if completion.usage else None
        return text, usage, time.time() - started_at
    except ModuleNotFoundError:
        pass

    import httpx

    headers = {"Authorization": f"Bearer {api_key}"}
    if base_url.startswith("https://openrouter.ai"):
        headers["HTTP-Referer"] = "https://github.com/jonrxu/STAT-4830-AlphaGrad-project"
        headers["X-Title"] = "STAT-4830-AlphaGrad-project"

    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_completion_tokens,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("chat completion response missing choices")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    usage = payload.get("usage")
    if not isinstance(text, str):
        raise ValueError("chat completion content was not a string")
    return text, usage if isinstance(usage, dict) else None, time.time() - started_at


def _query_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    timeout_seconds: int,
    model_max_context: int,
    completion_reserve_tokens: int,
) -> tuple[str, dict[str, Any] | None, float]:
    probe_text, probe_usage, probe_seconds = _post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=1,
        timeout_seconds=timeout_seconds,
    )
    prompt_tokens = int((probe_usage or {}).get("prompt_tokens") or 0)
    available_completion = model_max_context - prompt_tokens - completion_reserve_tokens
    allowed_completion = min(max_completion_tokens, available_completion)
    if allowed_completion <= 0:
        raise ValueError(
            f"prompt already consumes the model context budget: prompt_tokens={prompt_tokens} "
            f"model_max_context={model_max_context} reserve={completion_reserve_tokens}"
        )
    if allowed_completion == 1:
        return probe_text, probe_usage, probe_seconds
    text, usage, runtime_seconds = _post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=allowed_completion,
        timeout_seconds=timeout_seconds,
    )
    if usage is None and probe_usage is not None:
        usage = dict(probe_usage)
    if usage is not None:
        usage["effective_max_completion_tokens"] = allowed_completion
    return text, usage, probe_seconds + runtime_seconds


def _status_row(
    *,
    prompt_index: int,
    attempt: int,
    record: dict[str, Any],
    phase: str,
    status: str,
    candidate_sha: str,
    result: BenchEvalResult | None,
    generation_seconds: float,
    usage: dict[str, Any] | None,
    change_summary: str,
    failure_type: str | None = None,
) -> dict[str, Any]:
    if result is None:
        return {
            "prompt_index": prompt_index,
            "attempt": attempt,
            "brief_title": record.get("brief_title", ""),
            "brief_family": record.get("brief_family", ""),
            "phase": phase,
            "status": status,
            "candidate_sha256": candidate_sha,
            "valid": False,
            "recall_passed": False,
            "anti_cheat_passed": False,
            "build_ok": False,
            "runtime_ok": False,
            "qps": 0.0,
            "recall": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "score": 0.0,
            "failure_type": failure_type,
            "generation_seconds": generation_seconds,
            "eval_runtime_seconds": 0.0,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "change_summary": change_summary,
        }

    return {
        "prompt_index": prompt_index,
        "attempt": attempt,
        "brief_title": record.get("brief_title", ""),
        "brief_family": record.get("brief_family", ""),
        "phase": phase,
        "status": status,
        "candidate_sha256": candidate_sha,
        "valid": result.valid,
        "recall_passed": result.recall_passed,
        "anti_cheat_passed": result.anti_cheat_passed,
        "build_ok": result.build_ok,
        "runtime_ok": result.runtime_ok,
        "qps": result.qps,
        "recall": result.recall,
        "avg_latency_ms": result.avg_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "score": _score_result(result),
        "failure_type": result.failure_type,
        "generation_seconds": generation_seconds,
        "eval_runtime_seconds": result.runtime_seconds,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "change_summary": change_summary,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generation_rows = [row for row in rows if row["phase"] == "generation"]
    proxy_rows = [row for row in rows if row["phase"] == "proxy"]
    strict_rows = [row for row in rows if row["phase"] == "strict"]
    strict_valid = [row for row in strict_rows if row["valid"]]
    proxy_valid = [row for row in proxy_rows if row["valid"]]
    build_ok = [row for row in proxy_rows if row["build_ok"]]
    return {
        "attempts_total": len(generation_rows),
        "parse_successes": sum(1 for row in generation_rows if row["status"] == "parsed"),
        "build_successes": len(build_ok),
        "proxy_valid": len(proxy_valid),
        "strict_valid": len(strict_valid),
        "best_proxy_qps": max((row["qps"] for row in proxy_rows), default=0.0),
        "best_strict_qps": max((row["qps"] for row in strict_rows), default=0.0),
        "avg_generation_seconds": (
            sum(float(row["generation_seconds"] or 0.0) for row in generation_rows) / len(generation_rows)
            if generation_rows
            else 0.0
        ),
        "best_strict_row": max(strict_rows, key=lambda row: row["qps"], default=None),
    }


def main() -> int:
    args = parse_args()
    load_dotenv(DEFAULT_DOTENV_PATH)
    preset = PROVIDER_PRESETS[args.provider]
    if args.base_url == DEFAULT_BASE_URL:
        args.base_url = str(preset["base_url"])
    if args.model == DEFAULT_MODEL:
        args.model = str(preset["model"])
    if args.model_max_context == 8192:
        args.model_max_context = int(preset["model_max_context"])
    env_key = str(preset["env_key"])
    if args.api_key == "DUMMY" and env_key:
        args.api_key = os.environ.get(env_key, args.api_key)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.run_dir / "results.tsv"
    writer = _build_results_writer(results_path)

    try:
        _write_json(
            args.run_dir / "config.json",
            {
                **vars(args),
                "bench_repo": str(args.bench_repo),
                "prompts_path": str(args.prompts_path),
                "run_dir": str(args.run_dir),
                "base_vectors": str(args.base_vectors) if args.base_vectors else None,
                "query_vectors": str(args.query_vectors) if args.query_vectors else None,
                "ground_truth": str(args.ground_truth) if args.ground_truth else None,
            },
        )
        config = _prepare_eval_config(args)
        prompt_records = _load_prompt_records(args.prompts_path, args.max_prompts)
        incumbent_files = _bootstrap_seed_surface(config.skeleton_dir)

        rows: list[dict[str, Any]] = []
        for prompt_index, record in enumerate(prompt_records, start=1):
            prompt_dir = args.run_dir / f"prompt_{prompt_index:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            _write_json(prompt_dir / "prompt_record.json", record)
            messages = _prompt_messages(record)

            for attempt in range(1, args.attempts_per_prompt + 1):
                attempt_dir = prompt_dir / f"attempt_{attempt:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                _write_json(attempt_dir / "request_messages.json", messages)

                try:
                    raw_text, usage, generation_seconds = _query_model(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_completion_tokens=args.max_completion_tokens,
                        timeout_seconds=args.request_timeout_seconds,
                        model_max_context=args.model_max_context,
                        completion_reserve_tokens=args.completion_reserve_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    error_text = f"{type(exc).__name__}: {exc}\n"
                    response = getattr(exc, "response", None)
                    if response is not None:
                        try:
                            error_text += f"\n[status_code]={response.status_code}\n{response.text}\n"
                        except Exception:  # noqa: BLE001
                            pass
                    (attempt_dir / "request_error.txt").write_text(error_text, encoding="utf-8")
                    row = _status_row(
                        prompt_index=prompt_index,
                        attempt=attempt,
                        record=record,
                        phase="generation",
                        status="request_error",
                        candidate_sha="",
                        result=None,
                        generation_seconds=0.0,
                        usage=None,
                        change_summary="",
                        failure_type=type(exc).__name__,
                    )
                    rows.append(row)
                    writer.writerow(row)
                    _flush_results_writer(writer)
                    continue

                (attempt_dir / "response.txt").write_text(raw_text, encoding="utf-8")
                _write_json(attempt_dir / "usage.json", usage or {})

                try:
                    parsed_payload = _extract_json_object(raw_text)
                    _write_json(attempt_dir / "response.json", parsed_payload)
                    required_subset = _required_incumbent_subset(record, incumbent_files)
                    summary, candidate_subset = _parse_worker_output(
                        json.dumps(parsed_payload),
                        incumbent_files=required_subset,
                    )
                    candidate_files = dict(incumbent_files)
                    candidate_files.update(candidate_subset)
                except Exception as exc:  # noqa: BLE001
                    row = _status_row(
                        prompt_index=prompt_index,
                        attempt=attempt,
                        record=record,
                        phase="generation",
                        status="parse_error",
                        candidate_sha="",
                        result=None,
                        generation_seconds=generation_seconds,
                        usage=usage,
                        change_summary="",
                        failure_type=type(exc).__name__,
                    )
                    rows.append(row)
                    writer.writerow(row)
                    _flush_results_writer(writer)
                    continue

                candidate_sha = _surface_sha(candidate_files)
                candidate_dir = attempt_dir / "candidate"
                for relpath, content in candidate_files.items():
                    dest = candidate_dir / relpath
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")

                generation_row = _status_row(
                    prompt_index=prompt_index,
                    attempt=attempt,
                    record=record,
                    phase="generation",
                    status="parsed",
                    candidate_sha=candidate_sha,
                    result=None,
                    generation_seconds=generation_seconds,
                    usage=usage,
                    change_summary=summary,
                )
                rows.append(generation_row)
                writer.writerow(generation_row)
                _flush_results_writer(writer)

                proxy_result = _evaluate_candidate(
                    candidate_files=candidate_files,
                    workspace_dir=attempt_dir / "proxy_eval" / "workspace",
                    eval_dir=attempt_dir / "proxy_eval",
                    config=config,
                    inputs=config.proxy_inputs,
                )
                _write_json(attempt_dir / "proxy_eval.json", asdict(proxy_result))
                proxy_row = _status_row(
                    prompt_index=prompt_index,
                    attempt=attempt,
                    record=record,
                    phase="proxy",
                    status="evaluated",
                    candidate_sha=candidate_sha,
                    result=proxy_result,
                    generation_seconds=generation_seconds,
                    usage=usage,
                    change_summary=summary,
                )
                rows.append(proxy_row)
                writer.writerow(proxy_row)
                _flush_results_writer(writer)

                if args.strict_on_proxy_valid and proxy_result.valid:
                    strict_result = _evaluate_candidate(
                        candidate_files=candidate_files,
                        workspace_dir=attempt_dir / "strict_eval" / "workspace",
                        eval_dir=attempt_dir / "strict_eval",
                        config=config,
                        inputs=config.strict_inputs,
                    )
                    _write_json(attempt_dir / "strict_eval.json", asdict(strict_result))
                    strict_row = _status_row(
                        prompt_index=prompt_index,
                        attempt=attempt,
                        record=record,
                        phase="strict",
                        status="evaluated",
                        candidate_sha=candidate_sha,
                        result=strict_result,
                        generation_seconds=generation_seconds,
                        usage=usage,
                        change_summary=summary,
                    )
                    rows.append(strict_row)
                    writer.writerow(strict_row)
                    _flush_results_writer(writer)

        summary = _summarize_rows(rows)
        _write_json(args.run_dir / "summary.json", summary)
    finally:
        _close_results_writer(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
