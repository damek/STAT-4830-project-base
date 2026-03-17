#!/usr/bin/env python3
"""Plot AirBench autoresearch progress for a single run directory."""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "airbench" / "autoresearch_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory under data/airbench/autoresearch_runs. Defaults to the newest run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <run-dir>/progress.png",
    )
    return parser.parse_args()


def latest_run_dir(root: Path) -> Path:
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No autoresearch runs found in {root}")
    return candidates[0]


def maybe_float(value: str) -> float | None:
    if value in ("", "None", "null", None):
        return None
    return float(value)


def maybe_int(value: str) -> int | None:
    if value in ("", "None", "null", None):
        return None
    return int(value)


def maybe_bool(value: str) -> bool | None:
    if value in ("", "None", "null", None):
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Unexpected boolean string: {value!r}")


def load_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parsed = dict(row)
            parsed["attempt"] = maybe_int(row["attempt"])
            parsed["valid"] = maybe_bool(row["valid"])
            parsed["meets_target"] = maybe_bool(row["meets_target"])
            parsed["mean_accuracy"] = maybe_float(row["mean_accuracy"])
            parsed["mean_time_seconds"] = maybe_float(row["mean_time_seconds"])
            parsed["score"] = maybe_float(row["score"])
            parsed["runtime_seconds"] = maybe_float(row["runtime_seconds"])
            parsed["remote_runtime_seconds"] = maybe_float(row["remote_runtime_seconds"])
            rows.append(parsed)
    return rows


def short_label(summary: str) -> str:
    lowered = summary.lower()
    if "matrix multiplications" in lowered or "tf32" in lowered:
        return "enable TF32 matmul"
    if "batch_size" in summary and "2500" in summary:
        return "batch_size 2000→2500"
    if "max-autotune" in summary:
        return "explicit max-autotune"
    if "whiten bias" in lowered:
        return "whiten bias steps"
    if "sgd momentum" in lowered:
        return "SGD momentum tweak"
    if "cosine learning rate" in lowered:
        return "cosine LR"
    if "training duration" in lowered or "training epochs" in lowered:
        return "train duration tweak"
    compact = " ".join(summary.replace("`", "").split())
    return textwrap.shorten(compact, width=34, placeholder="...")


def annotate_point(ax: plt.Axes, x: int, y: float, label: str, *, color: str, dy: float) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(6, dy),
        textcoords="offset points",
        fontsize=9,
        color=color,
        rotation=28,
        ha="left",
        va="bottom",
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir(DEFAULT_RUN_ROOT)
    results_path = run_dir / "results.tsv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    output_path = args.output.resolve() if args.output else (run_dir / "progress.png")
    rows = load_rows(results_path)

    proxy_rows = [r for r in rows if r["phase"] == "proxy"]
    strict_rows = [r for r in rows if r["phase"] in {"baseline_strict", "strict_confirm"}]
    baseline_strict = next((r for r in strict_rows if r["phase"] == "baseline_strict"), None)
    kept_rows = [r for r in strict_rows if r["status"] == "keep"]
    strict_discards = [r for r in strict_rows if r["status"] == "discard"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_acc, ax_time) = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )

    feasible_proxy = [r for r in proxy_rows if r["meets_target"] and r["mean_time_seconds"] is not None]
    infeasible_proxy = [r for r in proxy_rows if r["mean_accuracy"] is not None and not r["meets_target"]]

    if infeasible_proxy:
        ax_acc.scatter(
            [r["attempt"] for r in infeasible_proxy],
            [100.0 * r["mean_accuracy"] for r in infeasible_proxy],
            s=46,
            color="#d9d9d9",
            edgecolors="none",
            alpha=0.9,
            label="Discarded proxy",
            zorder=2,
        )
    if feasible_proxy:
        ax_acc.scatter(
            [r["attempt"] for r in feasible_proxy],
            [100.0 * r["mean_accuracy"] for r in feasible_proxy],
            s=46,
            color="#d9d9d9",
            edgecolors="none",
            alpha=0.9,
            zorder=2,
        )

    if baseline_strict and baseline_strict["mean_accuracy"] is not None:
        ax_acc.scatter(
            [baseline_strict["attempt"]],
            [100.0 * baseline_strict["mean_accuracy"]],
            s=84,
            color="#2e86de",
            edgecolors="#1b4f72",
            linewidths=1.0,
            label="Baseline strict",
            zorder=4,
        )
        annotate_point(
            ax_acc,
            baseline_strict["attempt"],
            100.0 * baseline_strict["mean_accuracy"],
            "baseline",
            color="#2e86de",
            dy=6,
        )

    if strict_discards:
        ax_acc.scatter(
            [r["attempt"] for r in strict_discards],
            [100.0 * r["mean_accuracy"] for r in strict_discards if r["mean_accuracy"] is not None],
            s=72,
            marker="^",
            color="#f39c12",
            edgecolors="#8a5a00",
            linewidths=0.8,
            label="Failed strict confirm",
            zorder=4,
        )

    if kept_rows:
        ax_acc.scatter(
            [r["attempt"] for r in kept_rows],
            [100.0 * r["mean_accuracy"] for r in kept_rows],
            s=84,
            color="#2ecc71",
            edgecolors="#145a32",
            linewidths=1.0,
            label="Strict-confirmed keep",
            zorder=5,
        )
        for row in kept_rows:
            if row["mean_accuracy"] is not None:
                annotate_point(
                    ax_acc,
                    row["attempt"],
                    100.0 * row["mean_accuracy"],
                    short_label(row["change_summary"]),
                    color="#1e8449",
                    dy=6,
                )

    ax_acc.axhline(94.0, color="#c0392b", linestyle="--", linewidth=1.4, label="94% target", zorder=1)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(
        f"AirBench Autoresearch Progress: {len(proxy_rows)} Experiments, {len(kept_rows)} Strict-Confirmed Improvements",
        fontsize=20,
        pad=12,
    )
    ax_acc.legend(loc="lower right", frameon=True)

    if feasible_proxy:
        ax_time.scatter(
            [r["attempt"] for r in feasible_proxy],
            [r["mean_time_seconds"] for r in feasible_proxy],
            s=48,
            color="#d9d9d9",
            edgecolors="none",
            alpha=0.95,
            label="Feasible proxy",
            zorder=2,
        )

    step_x: list[int] = []
    step_y: list[float] = []
    if baseline_strict and baseline_strict["mean_time_seconds"] is not None:
        step_x.append(baseline_strict["attempt"])
        step_y.append(baseline_strict["mean_time_seconds"])
        ax_time.scatter(
            [baseline_strict["attempt"]],
            [baseline_strict["mean_time_seconds"]],
            s=84,
            color="#2e86de",
            edgecolors="#1b4f72",
            linewidths=1.0,
            zorder=4,
        )
        annotate_point(
            ax_time,
            baseline_strict["attempt"],
            baseline_strict["mean_time_seconds"],
            "baseline strict",
            color="#2e86de",
            dy=6,
        )

    if strict_discards:
        valid_discards = [r for r in strict_discards if r["mean_time_seconds"] is not None]
        ax_time.scatter(
            [r["attempt"] for r in valid_discards],
            [r["mean_time_seconds"] for r in valid_discards],
            s=72,
            marker="^",
            color="#f39c12",
            edgecolors="#8a5a00",
            linewidths=0.8,
            zorder=4,
        )

    if kept_rows:
        ax_time.scatter(
            [r["attempt"] for r in kept_rows],
            [r["mean_time_seconds"] for r in kept_rows],
            s=84,
            color="#2ecc71",
            edgecolors="#145a32",
            linewidths=1.0,
            zorder=5,
            label="Strict-confirmed keep",
        )
        for row in kept_rows:
            if row["mean_time_seconds"] is not None:
                step_x.append(row["attempt"])
                step_y.append(row["mean_time_seconds"])
                annotate_point(
                    ax_time,
                    row["attempt"],
                    row["mean_time_seconds"],
                    short_label(row["change_summary"]),
                    color="#1e8449",
                    dy=6,
                )

    if step_x:
        paired = sorted(zip(step_x, step_y), key=lambda item: item[0])
        ax_time.step(
            [x for x, _ in paired],
            [y for _, y in paired],
            where="post",
            color="#58d68d",
            linewidth=2.4,
            label="Running best (strict)",
            zorder=3,
        )

    ax_time.set_xlabel("Experiment #")
    ax_time.set_ylabel("Benchmark Time (s, lower is better)")
    ax_time.legend(loc="upper right", frameon=True)

    for ax in (ax_acc, ax_time):
        ax.grid(True, alpha=0.28)
        ax.set_xlim(-0.5, max([0] + [r["attempt"] for r in rows if r["attempt"] is not None]) + 0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
