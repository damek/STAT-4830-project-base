"""
AlphaGrad optimizer — main loop.

Each iteration:
  1. Gemini Flash proposes ONE change (hyperparameter or code_change).
  2. Human reviews and approves (or skips).
  3. If code_change: human implements it in train_gpt.py, confirms done.
  4. Ax BO runs N proxy evals on Modal A10G (~2.5 min each).
  5. If best proxy config improves on current best, save it.
  6. Print full-eval run command for the user to verify on RunPod.

Usage:
  cd parameter_golf/
  python -m alphagrad.main --iterations 10 --ax-trials 8
"""
import argparse
import copy
import json
from pathlib import Path

from .config import SOTA_CONFIG, PROXY_CONFIG, SOTA_BPB
from .llm_interface import get_llm_idea
from .search_space import build_search_space
from .ax_optimizer import run_ax_optimization
from .history import History
from .approver import approve_proposal, confirm_code_change_done, show_result
from .modal_runner import app


# ── CLI args ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaGrad hybrid optimizer")
    p.add_argument("--iterations",   type=int,   default=10,
                   help="Number of LLM → BO iterations")
    p.add_argument("--ax-trials",    type=int,   default=8,
                   help="Ax BO trials per iteration (proxy evals on Modal)")
    p.add_argument("--base-script",  default="train_gpt.py",
                   help="Base train_gpt.py for proxy eval (no FA3 needed)")
    p.add_argument("--sota-script",
                   default="records/track_10min_16mb/"
                           "2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
                   help="SOTA train_gpt.py for the printed full-eval command")
    p.add_argument("--history-file", default="alphagrad_experiments.json",
                   help="Persistent experiment log path")
    p.add_argument("--gemini-key",   default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--base-bpb",     type=float, default=SOTA_BPB,
                   help="BPB of the starting config to beat")
    p.add_argument("--auto",         action="store_true",
                   help="Auto-approve hyperparameter proposals, skip code changes")
    p.add_argument("--resume",       default=None,
                   help="Path to best_config.json to resume from")
    return p.parse_args()


# ── Proxy eval wrapper ────────────────────────────────────────────────────────

def _make_proxy_fn(base_script_path: str, proxy_base_config: dict):
    """
    Returns a function: config → bpb that runs on Modal.
    Reads the base script once and reuses the content for all trials.
    """
    from .modal_runner import proxy_eval_remote

    script_content = Path(base_script_path).read_text()

    def _proxy(config: dict) -> float:
        # Build proxy config: take optimizer params from config, rest from proxy defaults
        proxy_cfg = copy.deepcopy(proxy_base_config)
        for k in config:
            if k in proxy_base_config:
                proxy_cfg[k] = config[k]

        result = proxy_eval_remote.remote(proxy_cfg, script_content)
        bpb = result["bpb"]
        steps = result["steps"]
        print(f"      Modal: {steps} steps, BPB={bpb:.4f}")
        if bpb == float("inf"):
            print(f"      Log tail:\n{result['log_tail']}")
        return bpb

    return _proxy


# ── Full eval command printer ─────────────────────────────────────────────────

def _print_full_eval_command(sota_script: str, best_config: dict) -> None:
    print("\n  ── Full eval command (run this on 8xH100 RunPod) ──")
    env_str = " \\\n    ".join(
        f"{k}={v}" for k, v in sorted(best_config.items())
        if k not in ("RUN_ID", "VAL_LOSS_EVERY", "MAX_WALLCLOCK_SECONDS")
    )
    print(f"  {env_str} \\\n  torchrun --standalone --nproc_per_node=8 {sota_script}")
    print("  ───────────────────────────────────────────────────")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Load state
    best_sota_config = copy.deepcopy(SOTA_CONFIG)
    best_bpb = args.base_bpb
    history = History(args.history_file)

    if args.resume and Path(args.resume).exists():
        saved = json.loads(Path(args.resume).read_text())
        best_sota_config.update(saved.get("config", {}))
        best_bpb = saved.get("bpb", best_bpb)
        print(f"Resumed from {args.resume}  (BPB={best_bpb:.4f})")

    print(f"\n{'='*60}")
    print("AlphaGrad Hybrid Optimizer")
    print(f"  Target         : beat {best_bpb:.4f} BPB")
    print(f"  Iterations     : {args.iterations}")
    print(f"  Ax trials/iter : {args.ax_trials}  (proxy on Modal A10G)")
    print(f"  History        : {history.summary()}")
    print(f"{'='*60}\n")

    proxy_fn = _make_proxy_fn(args.base_script, PROXY_CONFIG)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'─'*60}")
        print(f"Iteration {iteration}/{args.iterations}  |  Best so far: {best_bpb:.4f} BPB")
        print(f"{'─'*60}")

        # ── Step 1: LLM proposes an idea ──────────────────────────────────
        print("[1/3] Asking Gemini Flash for a proposal...")
        try:
            idea = get_llm_idea(
                best_sota_config,
                best_bpb,
                history.get_recent(12),
                api_key=args.gemini_key,
            )
        except Exception as e:
            print(f"  LLM call failed: {e}")
            print("  Using fallback: tune MUON_WD and WARMDOWN_ITERS")
            idea = {
                "change_type": "hyperparameter",
                "idea": "tune weight decay and warmdown length",
                "params": ["MUON_WD", "ADAM_WD", "WARMDOWN_ITERS"],
                "reason": "fallback — core optimizer params",
                "priority": "medium",
            }

        # ── Step 2: Build search space ────────────────────────────────────
        ax_params = build_search_space(idea, best_sota_config)

        # ── Step 3: Human approval ────────────────────────────────────────
        print("[2/3] Waiting for approval...")
        if args.auto:
            if idea["change_type"] == "code_change":
                print("  [auto] Skipping code_change proposal (--auto mode).")
                continue
            print("  [auto] Auto-approving hyperparameter proposal.")
        else:
            if not approve_proposal(idea, ax_params):
                continue

            if idea["change_type"] == "code_change":
                if not confirm_code_change_done(idea):
                    continue
            # If new params were added, reload the script and update ax_params
            new_params = idea.get("new_params", {})
            if new_params:
                print(f"  Note: new params {list(new_params.keys())} will be tuned at defaults.")

        # ── Step 4: BO proxy optimization on Modal ────────────────────────
        print(f"[3/3] Running Ax BO ({args.ax_trials} proxy trials on Modal A10G)...")
        with app.run():
            candidate_config, candidate_bpb = run_ax_optimization(
                best_sota_config,
                ax_params,
                proxy_fn,
                n_trials=args.ax_trials,
            )
        print(f"\n  Best proxy BPB this round: {candidate_bpb:.4f}")

        # ── Step 5: Record and update ─────────────────────────────────────
        improved = candidate_bpb < best_bpb
        history.add(idea, candidate_config, candidate_bpb, improved)
        show_result(candidate_bpb, best_bpb, improved)

        if improved:
            best_sota_config = candidate_config
            best_bpb = candidate_bpb

            out_path = Path(args.history_file).parent / "best_config.json"
            out_path.write_text(
                json.dumps({"bpb": best_bpb, "config": best_sota_config}, indent=2)
            )
            print(f"  Saved → {out_path}")
            _print_full_eval_command(args.sota_script, best_sota_config)

    print(f"\n{'='*60}")
    print(f"Done.  Best proxy BPB: {best_bpb:.4f}")
    print(f"History: {history.summary()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
