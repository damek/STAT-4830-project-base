#!/usr/bin/env python3
"""Extract SFT training examples from vector-db-bench harness run directories.

Scans codex_cli_runs/ for worker (prompt, output) pairs where build_ok=True,
emits JSONL in HuggingFace chat format for SFT training.

Each JSONL line:
  {
    "messages": [
      {"role": "user",      "content": "<worker prompt>"},
      {"role": "assistant", "content": "<model JSON output>"}
    ],
    "metadata": {
      "run": "...", "round": "round_01", "attempt": 1,
      "build_ok": true, "valid": true, "qps": 289.0, "recall": 0.999
    }
  }

Usage:
  python scripts/vector_db_bench/extract_sft_data.py \\
      --run-root data/vector_db_bench/codex_cli_runs \\
      --output   data/vector_db_bench/sft_data.jsonl

  # Only include recall+anti-cheat valid examples with QPS >= 200:
  python scripts/vector_db_bench/extract_sft_data.py \\
      --only-valid --min-qps 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "vector_db_bench" / "codex_cli_runs"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "vector_db_bench" / "sft_data.jsonl"


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def extract_examples(run_root: Path) -> list[dict]:
    """Walk run_root looking for round_* directories containing results.tsv + worker files."""
    examples: list[dict] = []

    # Support both flat (run_root/round_01/) and nested (run_root/run_id/round_01/) layouts.
    round_dirs: list[tuple[str, Path]] = []
    for candidate in sorted(run_root.rglob("round_*")):
        if candidate.is_dir() and (candidate / "results.tsv").exists():
            run_name = candidate.parent.name if candidate.parent != run_root else ""
            round_dirs.append((run_name, candidate))

    if not round_dirs:
        print(f"[extract_sft_data] no round_* dirs with results.tsv found under {run_root}", file=sys.stderr)
        return examples

    for run_name, round_dir in round_dirs:
        results_tsv = round_dir / "results.tsv"
        try:
            with results_tsv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
        except Exception as exc:
            print(f"[extract_sft_data] skipping {results_tsv}: {exc}", file=sys.stderr)
            continue

        for row in rows:
            attempt_str = row.get("attempt", "").strip()
            if not attempt_str or attempt_str == "0":
                continue

            # Only keep rows from worker proposal/proxy phase, not baseline records.
            phase = row.get("phase", "").strip()
            if phase in ("baseline_proxy_record", "baseline_strict_record"):
                continue

            build_ok = _parse_bool(row.get("build_ok", "false"))
            if not build_ok:
                continue

            try:
                attempt_num = int(attempt_str)
            except ValueError:
                continue

            prompt_path = round_dir / f"worker_{attempt_num:02d}.prompt.txt"
            output_path = round_dir / f"worker_{attempt_num:02d}.output.json"

            if not prompt_path.exists() or not output_path.exists():
                continue

            prompt = prompt_path.read_text(encoding="utf-8").strip()
            output_text = output_path.read_text(encoding="utf-8").strip()

            if not prompt or not output_text:
                continue

            # Validate output is parseable JSON with the expected worker schema.
            try:
                parsed = json.loads(output_text)
                if not isinstance(parsed, dict) or "files" not in parsed:
                    continue
            except json.JSONDecodeError:
                continue

            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": output_text},
                    ],
                    "metadata": {
                        "run": run_name,
                        "round": round_dir.name,
                        "attempt": attempt_num,
                        "phase": phase,
                        "build_ok": True,
                        "valid": _parse_bool(row.get("valid", "false")),
                        "recall_passed": _parse_bool(row.get("recall_passed", "false")),
                        "anti_cheat_passed": _parse_bool(row.get("anti_cheat_passed", "false")),
                        "qps": _parse_float(row.get("qps")),
                        "recall": _parse_float(row.get("recall")),
                    },
                }
            )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SFT examples from harness runs")
    parser.add_argument(
        "--run-root", type=Path, default=DEFAULT_RUN_ROOT,
        help=f"Root directory containing codex_cli run subdirs (default: {DEFAULT_RUN_ROOT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--only-valid", action="store_true",
        help="Only include examples where recall_passed AND anti_cheat_passed",
    )
    parser.add_argument(
        "--min-qps", type=float, default=None,
        help="Exclude examples with QPS below this threshold (only useful with --only-valid)",
    )
    args = parser.parse_args()

    if not args.run_root.exists():
        print(f"[extract_sft_data] run-root does not exist: {args.run_root}", file=sys.stderr)
        print("  Run the harness first to generate training data.", file=sys.stderr)
        sys.exit(1)

    examples = extract_examples(args.run_root)

    if args.only_valid:
        examples = [e for e in examples if e["metadata"]["valid"]]

    if args.min_qps is not None:
        examples = [
            e for e in examples
            if (e["metadata"]["qps"] or 0.0) >= args.min_qps
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(examples)
    valid = sum(1 for e in examples if e["metadata"]["valid"])
    qps_vals = [e["metadata"]["qps"] for e in examples if e["metadata"]["qps"] is not None]
    avg_qps = sum(qps_vals) / len(qps_vals) if qps_vals else 0.0

    print(f"[extract_sft_data] wrote {total} examples → {args.output}")
    print(f"  build_ok=True (all):  {total}")
    print(f"  valid (recall+ac):    {valid} ({100*valid/total:.0f}%)" if total else "  valid: 0")
    print(f"  avg QPS (valid only): {avg_qps:.1f}")
    if total == 0:
        print(
            "\n  No examples found. Make sure the harness has been run with --run-dir\n"
            "  pointing to a persistent directory under data/vector_db_bench/codex_cli_runs/",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
