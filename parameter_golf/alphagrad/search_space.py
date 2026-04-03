"""
Convert an LLM proposal into an Ax parameter spec list.

For hyperparameter proposals:
  - Use the params the LLM suggested
  - Always add 2 anchor params (MATRIX_LR, MUON_MOMENTUM) so BO has baseline context
  - Narrow ranges around the current value (local search)

For code_change proposals:
  - After human implements the code, new params are added to TUNABLE_PARAMS dynamically
  - This module handles building the search space from those too
"""
import copy
from typing import Any

from .config import TUNABLE_PARAMS

# Always include these so the BO model always has optimizer context
_ANCHOR_PARAMS = ["MATRIX_LR", "MUON_MOMENTUM"]


def _narrow_range(spec: dict, current_val: float, tightness: float = 0.4) -> dict:
    """
    Narrow a range spec to ±tightness fraction around current_val,
    clamped to the original global bounds.
    tightness=0.4 means we search within 40% of the full range on each side.
    """
    spec = copy.deepcopy(spec)
    low, high = spec["bounds"]
    half = (high - low) * tightness
    spec["bounds"] = [
        max(low, current_val - half),
        min(high, current_val + half),
    ]
    if spec["bounds"][0] >= spec["bounds"][1]:
        spec["bounds"] = [low, high]  # fallback to full range if collapsed
    return spec


def build_search_space(
    idea: dict,
    current_config: dict[str, Any],
    extra_params: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Build an Ax parameter spec list from an LLM idea + current config.

    Args:
        idea:          Output from get_llm_idea().
        current_config: Current best config dict (used to narrow ranges).
        extra_params:   Additional param specs to include (e.g. from code_change
                        proposals after human implements new params).

    Returns:
        List of Ax parameter dicts for AxClient.create_experiment(parameters=...).
    """
    all_tunable = {**TUNABLE_PARAMS, **(extra_params or {})}

    # Start with the LLM's suggested params, then add anchors
    focus = list(idea.get("params", []))

    # For code_change, use tunable_after if present
    if idea.get("change_type") == "code_change":
        focus = list(idea.get("tunable_after", [])) + focus

    for p in _ANCHOR_PARAMS:
        if p not in focus:
            focus.append(p)

    focus = focus[:8]  # cap at 8 for BO tractability

    ax_params = []
    for name in focus:
        if name not in all_tunable:
            continue

        spec = copy.deepcopy(all_tunable[name])
        spec["name"] = name

        # Narrow range parameters around the current best value
        if spec["type"] == "range" and name in current_config:
            try:
                current_val = float(current_config[name])
                # Don't narrow log-scale params — they're already well-bounded
                if not spec.get("log_scale", False):
                    spec = _narrow_range(spec, current_val)
            except (TypeError, ValueError):
                pass

        ax_params.append(spec)

    return ax_params
