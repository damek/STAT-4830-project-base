"""
Bayesian optimization using the Ax service API.

Calls Modal for each proxy evaluation.
Returns the best config found across all trials in a BO round.
"""
import copy
from typing import Callable

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


def run_ax_optimization(
    base_config: dict,
    ax_params: list[dict],
    eval_fn: Callable[[dict], float],
    n_trials: int = 8,
) -> tuple[dict, float]:
    """
    Run BO and return (best_config, best_bpb).

    Args:
        base_config: Current best config (non-tuned params stay fixed).
        ax_params:   Ax parameter spec list from build_search_space().
        eval_fn:     config dict → bpb float (calls Modal proxy eval).
        n_trials:    Total trials (first ~n/2 are Sobol quasi-random, rest are BO).

    Returns:
        (best_config, best_bpb) — best found in this round.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="pg_bo_round",
        parameters=ax_params,
        objectives={"bpb": ObjectiveProperties(minimize=True)},
    )

    best_bpb = float("inf")
    best_config = copy.deepcopy(base_config)

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        trial_config = copy.deepcopy(base_config)
        trial_config.update(params)

        try:
            bpb = eval_fn(trial_config)
            ax_client.complete_trial(trial_index, raw_data={"bpb": (bpb, None)})
            marker = "★" if bpb < best_bpb else " "
            print(f"  {marker} Trial {i+1:2d}/{n_trials}: BPB={bpb:.4f}  {params}")

            if bpb < best_bpb:
                best_bpb = bpb
                best_config = copy.deepcopy(trial_config)
        except Exception as e:
            print(f"    Trial {i+1:2d}/{n_trials}: FAILED — {e}")
            ax_client.log_trial_failure(trial_index)

    return best_config, best_bpb
