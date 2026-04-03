"""
LLM interface — Gemini Flash proposes ONE change per iteration.

Two proposal types:
  "hyperparameter" — tune existing numeric params (BO handles it directly)
  "code_change"    — add/modify a small piece of logic in train_gpt.py
                     (human reviews and implements before BO runs)

Output is always structured JSON so it's easy to parse and display.
"""
import json
import os
import re
from typing import Any

from google import genai
from google.genai import types as genai_types

from .config import TUNABLE_PARAMS


_KNOWN_TRICKS = """\
KNOWN TRICKS ALREADY IN THE SOTA SCRIPT (do NOT re-suggest these):
- XSA (cross-sequence attention) on all 11 layers
- BigramHash embeddings (3072 x 112)
- Partial RoPE (16/64 dims)
- LayerNorm scale 1/sqrt(layer+1)
- Value embeddings on layers 9-10
- LeakyReLU(0.5)^2 activation in MLP
- Int6 QAT with STE, late activation at LR scale < 0.15
- Full Hessian GPTQ with AR self-generated calibration
- EMA (0.997) + SWA every 50 steps
- Parallel Muon + parameter banking
- U-Net skip connections
- SmearGate position mixing
- Sliding window evaluation (stride=64)
- LZMA preset=9 compression
- Warmdown over 4000 iterations
- Flash Attention 3 (Hopper)
"""

_TUNABLE_LIST = "\n".join(
    f"  {name}: {spec}"
    for name, spec in TUNABLE_PARAMS.items()
)


def _build_prompt(
    best_config: dict[str, Any],
    best_bpb: float,
    history: list[dict],
) -> str:
    key_params = {k: best_config.get(k) for k in TUNABLE_PARAMS if k in best_config}

    tried_lines = []
    for h in history[-12:]:
        status = "IMPROVED" if h["improved"] else "no gain"
        tried_lines.append(
            f"  [{status}] {h['idea']}  bpb={h['bpb']:.4f}  params={h['params']}"
        )
    tried_str = "\n".join(tried_lines) if tried_lines else "  None yet."

    return f"""You are helping improve the #1 solution to the OpenAI Parameter Golf Challenge.
Goal: minimize val_bpb (bits per byte) on FineWeb. Constraints: ≤16MB artifact, ≤10 min on 8xH100.

CURRENT BEST BPB: {best_bpb:.4f}

CURRENT TUNABLE PARAMS (values in the best config):
{json.dumps(key_params, indent=2)}

{_KNOWN_TRICKS}

TUNABLE PARAMETER RANGES:
{_TUNABLE_LIST}

RECENT EXPERIMENT HISTORY (avoid repeating these):
{tried_str}

YOUR TASK:
Propose EXACTLY ONE change. It must be either:
  (a) A hyperparameter change — tune 1-4 existing params from the tunable list above.
  (b) A small code change — add ONE new technique not already in the SOTA script.
      Must be specific enough for a human to implement in <30 lines of code.
      Must introduce at least one new hyperparameter for BO to tune.

Rules:
- One change only. No rewrites. No vague suggestions.
- For hyperparameter proposals: pick params with the most remaining uncertainty.
- For code proposals: only suggest if you're confident it helps and isn't already there.
- Do NOT suggest anything from the "already in SOTA" list above.

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "change_type": "hyperparameter" or "code_change",
  "idea": "<one sentence describing the change>",
  "params": ["PARAM1", "PARAM2"],
  "reason": "<one sentence rationale based on known ML principles>",
  "priority": "high" | "medium" | "low",

  // only if change_type == "code_change":
  "code_description": "<what to implement and where in train_gpt.py>",
  "new_params": {{"NEW_PARAM_NAME": default_value}},
  "tunable_after": ["NEW_PARAM_NAME"]
}}"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_llm_idea(
    best_config: dict[str, Any],
    best_bpb: float,
    history: list[dict],
    api_key: str | None = None,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """
    Ask Gemini Flash for one improvement idea.

    Returns a dict with at minimum:
      change_type, idea, params, reason, priority

    For code_change proposals, also includes:
      code_description, new_params, tunable_after
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Set GEMINI_API_KEY env var or pass api_key=")

    client = genai.Client(api_key=key)
    prompt = _build_prompt(best_config, best_bpb, history)

    for attempt in range(3):
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        try:
            idea = json.loads(_strip_fences(response.text))
            break
        except json.JSONDecodeError as e:
            if attempt == 2:
                raise ValueError(f"LLM returned invalid JSON after 3 attempts: {e}\n{response.text}")
            continue

    # Validate required fields
    for field in ("change_type", "idea", "params", "reason"):
        if field not in idea:
            raise ValueError(f"LLM response missing field '{field}': {idea}")

    if idea["change_type"] not in ("hyperparameter", "code_change"):
        raise ValueError(f"Invalid change_type: {idea['change_type']}")

    # Filter params to only known tunable ones (for hyperparameter proposals)
    if idea["change_type"] == "hyperparameter":
        idea["params"] = [p for p in idea["params"] if p in TUNABLE_PARAMS]
        if not idea["params"]:
            raise ValueError(f"No valid tunable params in proposal: {idea}")

    return idea
