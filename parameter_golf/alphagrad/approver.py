"""
Human-in-the-loop approval step.

For hyperparameter proposals: shows the idea + search space, asks approve/skip.
For code_change proposals: shows what to implement, waits for confirmation that
  the human has made the change to train_gpt.py before BO proceeds.
"""
import json
from typing import Any


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2)


def approve_proposal(idea: dict, ax_params: list[dict]) -> bool:
    """
    Show the LLM's proposal and ask the user to approve it.
    Returns True if approved, False to skip this iteration.
    """
    print("\n" + "━" * 60)
    print("LLM PROPOSAL")
    print("━" * 60)
    print(f"  Type    : {idea['change_type']}")
    print(f"  Idea    : {idea['idea']}")
    print(f"  Reason  : {idea['reason']}")
    print(f"  Priority: {idea.get('priority', '?')}")
    print(f"  Params  : {idea['params']}")

    if idea["change_type"] == "code_change":
        print(f"\n  Code description:")
        print(f"  {idea.get('code_description', '(none provided)')}")
        print(f"\n  New params to add: {idea.get('new_params', {})}")
        print(f"  Params BO will tune after: {idea.get('tunable_after', [])}")

    print(f"\nAx search space ({len(ax_params)} params):")
    for p in ax_params:
        if p["type"] == "range":
            print(f"  {p['name']}: [{p['bounds'][0]}, {p['bounds'][1]}]")
        else:
            print(f"  {p['name']}: {p.get('values', '?')}")

    print("━" * 60)

    while True:
        resp = input("Approve? [y/n/q] (y=run, n=skip, q=quit): ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            print("  Skipped.")
            return False
        if resp in ("q", "quit"):
            print("  Quitting.")
            raise SystemExit(0)
        print("  Please enter y, n, or q.")


def confirm_code_change_done(idea: dict) -> bool:
    """
    For code_change proposals: ask the user to confirm they've implemented
    the change in train_gpt.py before proceeding to BO.
    """
    print("\n" + "━" * 60)
    print("CODE CHANGE REQUIRED")
    print("━" * 60)
    print(f"  Please implement the following in train_gpt.py:")
    print(f"  {idea.get('code_description', idea['idea'])}")
    if idea.get("new_params"):
        print(f"\n  Add these as new env var hyperparameters:")
        for k, v in idea["new_params"].items():
            print(f"    {k} = {v}  (default)")
    print("━" * 60)

    while True:
        resp = input("Have you implemented this? [y/n]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            print("  Skipping this iteration.")
            return False


def show_result(bpb: float, best_bpb: float, improved: bool) -> None:
    print("\n" + "━" * 60)
    if improved:
        print(f"  IMPROVED: {best_bpb:.4f} → {bpb:.4f}  (Δ = {best_bpb - bpb:.4f})")
    else:
        print(f"  No improvement: proxy {bpb:.4f} ≥ best {best_bpb:.4f}")
    print("━" * 60)
