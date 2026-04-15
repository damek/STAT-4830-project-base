#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REFERENCE_MODELS = [
    ("anthropic/claude-sonnet-4.5", 140.9, "99.67%", "#d56ab4"),
    ("doubao-seed-2.0-code-preview-260215", 76.4, "99.93%", "#e07db7"),
    ("openai/gpt-5.2-codex", 73.8, "99.93%", "#80b73b"),
    ("doubao-seed-2.0-pro-260215", 71.4, "99.94%", "#7fbe4f"),
    ("google/gemini-3-flash-preview", 67.4, "99.92%", "#7ec96f"),
    ("MiniMax-M2.5", 57.8, "99.94%", "#e1b42a"),
    ("deepseek/deepseek-v3.2", 55.6, "99.93%", "#d7ad24"),
    ("glm-5", 54.7, "99.94%", "#d0a31f"),
    ("MiniMax-M2.7-highspeed", 51.2, "99.94%", "#c49318"),
    ("Ling-2.5-1T", 46.5, "99.94%", "#c39b5f"),
    ("x-ai/grok-4.1-fast", 43.9, "99.97%", "#b98d53"),
]

IMPROVEMENT_NOTES = {
    1: "baseline exact scan",
    5: "heap top-k",
    7: "SIMD dist tune",
    8: "flat ids/data layout",
    9: "scan/layout tuning",
    10: "heap/layout tune",
    11: "rayon parallel scan",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Plot continuous vector-db-bench progress with public reference lines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "vector_db_bench" / "qwen3_meta_snapshots" / "continuous_20260414_50rounds_fresh_results.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "vector_db_bench" / "qwen3_meta_figures" / "continuous_20260414_50rounds_fresh_progress.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    df["round"] = df["round"].astype(int)
    df["agent_best_qps"] = pd.to_numeric(df["agent_best_qps"], errors="coerce")
    df["final_eval_qps"] = pd.to_numeric(df["final_eval_qps"], errors="coerce")
    df["incumbent_qps_after"] = pd.to_numeric(df["incumbent_qps_after"], errors="coerce")
    df["final_eval_valid"] = df["final_eval_valid"].astype(str).str.lower().eq("true")
    df["promoted"] = df["promoted"].astype(str).str.lower().eq("true")

    fig, ax = plt.subplots(figsize=(13.5, 8))
    bg = "#fcfcfd"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for name, qps, recall, color in REFERENCE_MODELS:
        ax.axhline(qps, color=color, linestyle=(0, (2, 4)), linewidth=1.2, alpha=0.28, zorder=0)
        ax.text(
            df["round"].max() + 0.32,
            qps,
            f"{name}  {qps:.1f} QPS (recall {recall})",
            color=color,
            va="center",
            ha="left",
            fontsize=8.7,
            alpha=0.78,
        )

    valid = df[df["final_eval_valid"]]
    invalid = df[~df["final_eval_valid"]]
    kept = df[df["promoted"]]

    ax.scatter(
        valid["round"],
        valid["final_eval_qps"],
        s=58,
        color="#2d6cdf",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.92,
        label="Round final eval",
        zorder=3,
    )

    if not invalid.empty:
        ax.scatter(
            invalid["round"],
            [12] * len(invalid),
            s=52,
            marker="x",
            color="#d94841",
            linewidth=1.7,
            alpha=0.85,
            label="Invalid final eval",
            zorder=3,
        )

    ax.plot(
        df["round"],
        df["incumbent_qps_after"],
        color="#111827",
        linewidth=3.0,
        label="Best kept QPS so far",
        zorder=4,
    )

    ax.scatter(
        kept["round"],
        kept["incumbent_qps_after"],
        s=78,
        facecolor="#111827",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )

    kept_points = kept[["round", "incumbent_qps_before", "incumbent_qps_after"]].copy()
    kept_points["delta"] = kept_points["incumbent_qps_after"] - kept_points["incumbent_qps_before"]
    for _, row in kept_points.iterrows():
        before = float(row["incumbent_qps_before"])
        after = float(row["incumbent_qps_after"])
        round_idx = float(row["round"])
        if before <= 0:
            ax.annotate(
                f"+{row['delta']:.2f}\n{IMPROVEMENT_NOTES.get(int(round_idx), 'improvement')}",
                xy=(round_idx, after),
                xytext=(round_idx + 0.15, after + 3.0),
                fontsize=7.8,
                color="#111827",
                ha="left",
                va="bottom",
                zorder=6,
            )
            continue

        ax.plot(
            [round_idx, round_idx],
            [before, after],
            color="#111827",
            linewidth=1.1,
            alpha=0.75,
            zorder=4,
        )
        ax.annotate(
            f"+{row['delta']:.2f}\n{IMPROVEMENT_NOTES.get(int(round_idx), 'improvement')}",
            xy=(round_idx, (before * after) ** 0.5),
            xytext=(round_idx + 0.12, after + 1.8),
            fontsize=7.4,
            color="#111827",
            ha="left",
            va="bottom",
            zorder=6,
        )

    best_row = df.loc[df["incumbent_qps_after"].idxmax()]
    ax.annotate(
        f"Best kept: {best_row['incumbent_qps_after']:.2f} QPS\nround {int(best_row['round'])}",
        xy=(best_row["round"], best_row["incumbent_qps_after"]),
        xytext=(best_row["round"] + 0.5, best_row["incumbent_qps_after"] * 1.22),
        arrowprops=dict(arrowstyle="-", color="#111827", lw=1.2),
        fontsize=10,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.92),
        zorder=6,
    )

    ref_max = max(qps for _, qps, _, _ in REFERENCE_MODELS)
    y_max = max(ref_max, float(df["incumbent_qps_after"].max())) * 1.10
    ax.set_ylim(0, y_max)
    ax.set_xlim(0.6, df["round"].max() + 2.4)
    ax.set_xlabel("Round")
    ax.set_ylabel("QPS")
    ax.set_title("Continuous Qwen3-Coder-Next Run: Progress vs Comparable Reference Scores", pad=16)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.12)

    subtitle = (
        "Dots: final evaluation per completed round. Black line: incumbent best-so-far. "
        "Red x markers indicate invalid final evaluations. Jump labels show kept deltas and inferred changes."
    )
    fig.text(0.125, 0.93, subtitle, fontsize=9.5, color="#4b5563")

    legend = ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#d1d5db")
    for text in legend.get_texts():
        text.set_fontsize(9)

    fig.tight_layout(rect=(0, 0, 0.92, 0.92))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
