#!/usr/bin/env python3
"""Shared autoresearch loop logic for local and Modal-hosted runs."""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


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
        if not key:
            continue
        if os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def normalize_target_accuracy(raw_value: float) -> float:
    return raw_value / 100.0 if raw_value > 1.0 else raw_value


def ensure_auth(model: str) -> None:
    if model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is required for model {model!r}")
    if model.startswith("gemini/") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(f"GEMINI_API_KEY or GOOGLE_API_KEY is required for model {model!r}")
    if model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(f"OPENROUTER_API_KEY is required for model {model!r}")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EditableSectionSpec:
    name: str
    symbols: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class WorkerBrief:
    title: str
    section: str
    family: str
    hypothesis: str
    instructions: str

    def prompt_text(self) -> str:
        return (
            f"title: {self.title}\n"
            f"section: {self.section}\n"
            f"family: {self.family}\n"
            f"hypothesis: {self.hypothesis}\n"
            f"instructions: {self.instructions}"
        )


SECTION_SPECS = (
    EditableSectionSpec(
        name="optimizer_core",
        symbols=("zeropower_via_newtonschulz5", "Muon"),
        description="optimizer math and Muon optimizer behavior",
    ),
    EditableSectionSpec(
        name="model_core",
        symbols=("ConvGroup", "CifarNet"),
        description="model block structure and network forward/reset logic; keep whitening init unchanged",
    ),
    EditableSectionSpec(
        name="eval_core",
        symbols=("infer", "evaluate"),
        description="TTA and evaluation implementation",
    ),
    EditableSectionSpec(
        name="training_loop",
        symbols=("run_single_trial", "run_preflight"),
        description="training schedule, optimizer setup, batch size, and preflight logic",
    ),
)

SECTION_SPECS_BY_NAME = {spec.name: spec for spec in SECTION_SPECS}


def _named_top_level_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[node.name] = node
    return nodes


def _extract_method_dump(code: str, *, class_name: str, method_name: str) -> str | None:
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                return ast.dump(item, include_attributes=False)
    return None


def _section_line_span(code: str, spec: EditableSectionSpec) -> tuple[int, int]:
    tree = ast.parse(code)
    nodes = _named_top_level_nodes(tree)
    missing = [name for name in spec.symbols if name not in nodes]
    if missing:
        raise ValueError(f"missing section symbols in candidate.py: {', '.join(missing)}")
    selected = [nodes[name] for name in spec.symbols]
    start = min(node.lineno for node in selected)
    end = max(node.end_lineno or node.lineno for node in selected)
    return start, end


def _section_source(code: str, spec: EditableSectionSpec) -> str:
    start, end = _section_line_span(code, spec)
    lines = code.splitlines()
    return "\n".join(lines[start - 1 : end]).rstrip() + "\n"


def _validate_section_replacement(section_name: str, replacement_code: str, base_code: str) -> str | None:
    # The outer loop only allows one bounded rewrite at a time. This check keeps
    # the model from smuggling in wrapper edits by returning extra symbols.
    spec = SECTION_SPECS_BY_NAME.get(section_name)
    if spec is None:
        return f"unknown section: {section_name}"

    try:
        tree = ast.parse(replacement_code)
    except SyntaxError as exc:
        return f"section parse failed: {exc}"

    named_nodes = _named_top_level_nodes(tree)
    actual_names = tuple(named_nodes.keys())
    if actual_names != spec.symbols:
        expected = ", ".join(spec.symbols)
        actual = ", ".join(actual_names) or "<none>"
        return f"section {section_name} must define exactly: {expected}; got: {actual}"

    if section_name == "model_core":
        original_dump = _extract_method_dump(base_code, class_name="CifarNet", method_name="init_whiten")
        replacement_dump = _extract_method_dump(replacement_code, class_name="CifarNet", method_name="init_whiten")
        if original_dump is None or replacement_dump is None:
            return "model_core must preserve CifarNet.init_whiten"
        if original_dump != replacement_dump:
            return "model_core must preserve CifarNet.init_whiten exactly"

    return None


def apply_section_edit(base_code: str, section_name: str, replacement_code: str) -> str:
    # Rebuild the full candidate by splicing a validated section into the current
    # incumbent. Everything outside the section stays byte-for-byte identical.
    spec = SECTION_SPECS_BY_NAME[section_name]
    start, end = _section_line_span(base_code, spec)
    lines = base_code.splitlines()
    replacement_lines = replacement_code.rstrip().splitlines()
    new_lines = lines[: start - 1] + replacement_lines + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def section_inventory_text(base_code: str) -> str:
    parts = []
    for spec in SECTION_SPECS:
        parts.append(f"- {spec.name}: {spec.description}")
        parts.append("  Current code:")
        parts.append("```python")
        parts.append(_section_source(base_code, spec).rstrip())
        parts.append("```")
    return "\n".join(parts)


def extract_summary_section_and_code(raw_text: str) -> tuple[str, str, str]:
    # The proposal protocol is intentionally rigid so the loop can treat model
    # output as a structured section replacement instead of a free-form rewrite.
    summary = "model proposal"
    summary_match = re.search(r"^SUMMARY:\s*(.+)$", raw_text, flags=re.MULTILINE)
    if summary_match:
        summary = summary_match.group(1).strip()

    section_match = re.search(r"^SECTION:\s*([A-Za-z0-9_]+)\s*$", raw_text, flags=re.MULTILINE)
    if not section_match:
        raise ValueError("model response did not include SECTION")
    section_name = section_match.group(1).strip()
    if section_name not in SECTION_SPECS_BY_NAME:
        raise ValueError(f"model response used unknown section {section_name!r}")

    fence_match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        code = fence_match.group(1).strip()
    else:
        code = raw_text.strip()
        last_header_end = 0
        if summary_match:
            last_header_end = max(last_header_end, summary_match.end())
        if section_match:
            last_header_end = max(last_header_end, section_match.end())
        if last_header_end:
            code = raw_text[last_header_end:].strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else ""
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0].rstrip()
    if not code:
        raise ValueError("model response did not include candidate code")
    return summary, section_name, code


def _extract_json_payload(raw_text: str) -> Any:
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else raw_text.strip()
    return json.loads(candidate)


REQUIRED_JSON_KEYS = (
    "mean_accuracy",
    "mean_time_seconds",
    "trials",
)


def validate_candidate_contract(code: str) -> str | None:
    # Section-locked editing already preserves the wrapper. This last check only
    # verifies that the assembled file still contains the required result fields.
    missing_json_keys = [key for key in REQUIRED_JSON_KEYS if f'"{key}"' not in code and f"'{key}'" not in code]
    if missing_json_keys:
        return f"missing required JSON keys: {', '.join(missing_json_keys)}"
    return None


def build_results_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "attempt",
        "phase",
        "status",
        "candidate_sha256",
        "parent_sha256",
        "valid",
        "meets_target",
        "mean_accuracy",
        "mean_time_seconds",
        "score",
        "failure_type",
        "actual_device_name",
        "runtime_seconds",
        "remote_runtime_seconds",
        "change_summary",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer._handle = handle  # type: ignore[attr-defined]
    return writer


def close_results_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def load_results_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def eval_row(result: Any, *, attempt: int, phase: str, status: str, candidate_sha: str, parent_sha: str, change_summary: str) -> dict[str, Any]:
    side = result.as_side_info()
    return {
        "attempt": attempt,
        "phase": phase,
        "status": status,
        "candidate_sha256": candidate_sha,
        "parent_sha256": parent_sha,
        "valid": result.valid,
        "meets_target": side.get("meets_target"),
        "mean_accuracy": result.mean_accuracy,
        "mean_time_seconds": result.mean_time_seconds,
        "score": result.score,
        "failure_type": result.failure_type,
        "actual_device_name": side.get("actual_device_name"),
        "runtime_seconds": result.runtime_seconds,
        "remote_runtime_seconds": side.get("remote_runtime_seconds"),
        "change_summary": change_summary,
    }


CORE_RESULT_FIELDS = {
    "score",
    "valid",
    "failure_type",
    "message",
    "runtime_seconds",
    "mean_accuracy",
    "mean_time_seconds",
    "trials",
    "stdout_tail",
    "stderr_tail",
}


@dataclass(frozen=True)
class RecordedEvalResult:
    score: float
    valid: bool
    failure_type: str | None
    message: str
    runtime_seconds: float
    mean_accuracy: float | None
    mean_time_seconds: float | None
    trials: int | None
    stdout_tail: str
    stderr_tail: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_side_info(self) -> dict[str, Any]:
        payload = {
            "score": self.score,
            "valid": self.valid,
            "failure_type": self.failure_type,
            "message": self.message,
            "runtime_seconds": self.runtime_seconds,
            "mean_accuracy": self.mean_accuracy,
            "mean_time_seconds": self.mean_time_seconds,
            "trials": self.trials,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }
        payload.update(self.extra)
        return payload


def _recorded_result_from_payload(payload: dict[str, Any]) -> RecordedEvalResult:
    extra = {key: value for key, value in payload.items() if key not in CORE_RESULT_FIELDS}
    return RecordedEvalResult(
        score=float(payload["score"]),
        valid=bool(payload["valid"]),
        failure_type=payload.get("failure_type"),
        message=str(payload.get("message", "")),
        runtime_seconds=float(payload.get("runtime_seconds") or 0.0),
        mean_accuracy=float(payload["mean_accuracy"]) if payload.get("mean_accuracy") is not None else None,
        mean_time_seconds=float(payload["mean_time_seconds"]) if payload.get("mean_time_seconds") is not None else None,
        trials=int(payload["trials"]) if payload.get("trials") is not None else None,
        stdout_tail=str(payload.get("stdout_tail", "")),
        stderr_tail=str(payload.get("stderr_tail", "")),
        extra=extra,
    )


def serialize_incumbent_record(candidate_sha: str, proxy_result: Any, strict_result: Any) -> dict[str, Any]:
    return {
        "candidate_sha256": candidate_sha,
        "proxy": proxy_result.as_side_info(),
        "strict": strict_result.as_side_info(),
    }


def load_incumbent_record(record_path: Path, expected_sha: str) -> tuple[RecordedEvalResult, RecordedEvalResult]:
    if not record_path.exists():
        raise FileNotFoundError(
            f"incumbent record not found at {record_path}; refusing to revalidate initial incumbent"
        )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    recorded_sha = str(payload.get("candidate_sha256", "")).strip()
    if recorded_sha != expected_sha:
        raise ValueError(
            "incumbent record hash mismatch; candidate.py does not match the validated incumbent-of-record"
        )
    proxy_payload = payload.get("proxy")
    strict_payload = payload.get("strict")
    if not isinstance(proxy_payload, dict) or not isinstance(strict_payload, dict):
        raise ValueError(f"invalid incumbent record structure in {record_path}")
    return _recorded_result_from_payload(proxy_payload), _recorded_result_from_payload(strict_payload)


def is_better(new: Any, old: Any) -> bool:
    if not new.valid:
        return False
    if not old.valid:
        return True

    new_meets = bool(new.as_side_info().get("meets_target"))
    old_meets = bool(old.as_side_info().get("meets_target"))
    if new_meets != old_meets:
        return new_meets

    if new_meets and old_meets:
        if new.mean_time_seconds is None or old.mean_time_seconds is None:
            return False
        if abs(new.mean_time_seconds - old.mean_time_seconds) > 1e-9:
            return new.mean_time_seconds < old.mean_time_seconds
        if new.mean_accuracy is not None and old.mean_accuracy is not None:
            return new.mean_accuracy > old.mean_accuracy + 1e-9
        return False

    if new.mean_accuracy is not None and old.mean_accuracy is not None:
        if abs(new.mean_accuracy - old.mean_accuracy) > 1e-9:
            return new.mean_accuracy > old.mean_accuracy
    if new.mean_time_seconds is not None and old.mean_time_seconds is not None:
        if abs(new.mean_time_seconds - old.mean_time_seconds) > 1e-9:
            return new.mean_time_seconds < old.mean_time_seconds
    return False


def update_memory(memory_path: Path, incumbent_result: Any, accepted_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> None:
    side = incumbent_result.as_side_info()
    lines = ["# Memory", "", "## Current Best"]
    lines.append(f"- valid: {incumbent_result.valid}")
    lines.append(f"- meets_target: {side.get('meets_target')}")
    lines.append(f"- mean_accuracy: {incumbent_result.mean_accuracy}")
    lines.append(f"- mean_time_seconds: {incumbent_result.mean_time_seconds}")
    lines.append(f"- score: {incumbent_result.score}")
    lines.append("")
    lines.append("## Recent Accepted Changes")
    accepted_changes = [row for row in accepted_rows if row.get("attempt", 0) > 0]
    if accepted_changes:
        for row in accepted_changes[-3:]:
            lines.append(f"- attempt {row['attempt']}: {row['change_summary']}")
    else:
        lines.append("- none yet")
    lines.append("")
    lines.append("## Recent Rejections / Failures")
    if rejected_rows:
        for row in rejected_rows[-5:]:
            failure = row.get("failure_type") or row.get("status") or "discard"
            lines.append(f"- attempt {row['attempt']}: {failure} | {row['change_summary']}")
    else:
        lines.append("- none yet")
    lines.append("")
    lines.append("## Repeated Failure Modes")
    lines.append("- Preserve dtype consistency between normalized inputs and half-precision conv weights/biases.")
    lines.append("- Preserve CLI flags, especially --verbose.")
    lines.append("- Avoid CUDAGraph-unsafe repeated compiled inference patterns in TTA.")
    memory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prompt(
    *,
    program_text: str,
    strategy_text: str,
    memory_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    worker_brief: WorkerBrief | None = None,
) -> list[dict[str, str]]:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: acc={row.get('mean_accuracy')} time={row.get('mean_time_seconds')} failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in recent_rows
    ) or "- no recent attempts"
    sections_text = "\n".join(f"- {spec.name}: {spec.description}" for spec in SECTION_SPECS)
    system = (
        "You are improving one Python training script in a keep/discard experiment loop. "
        "You may edit exactly one allowed section at a time; all other code is fixed infrastructure and will be preserved mechanically. "
        "Choose one section from the allowed list and return only a replacement for that section, not the full file. "
        "The change should reflect a single clear technical hypothesis rather than random unrelated tweaks. "
        "Preserve the benchmark contract and keep the program readable. "
        "Do not return explanations after the code. Respond with exactly three parts: "
        "a first line 'SUMMARY: ...', a second line 'SECTION: <name>', and one fenced Python code block containing only the replacement code for that section. "
        f"Allowed sections:\n{sections_text}"
    )
    user = (
        f"Program instructions:\n{program_text}\n\n"
        f"Current strategy:\n{strategy_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Editable sections:\n{section_inventory_text(candidate_code)}\n\n"
        + (
            f"Assigned worker brief:\n{worker_brief.prompt_text()}\n\n"
            if worker_brief is not None
            else ""
        )
        + "Produce one section replacement only. Pick the section that best matches the experiment you want to run, and leave all other sections untouched."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def propose_candidate(
    *,
    model: str,
    program_text: str,
    strategy_text: str,
    memory_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    worker_brief: WorkerBrief | None = None,
) -> tuple[str, str, str]:
    import litellm

    messages = build_prompt(
        program_text=program_text,
        strategy_text=strategy_text,
        memory_text=memory_text,
        candidate_code=candidate_code,
        recent_rows=recent_rows,
        worker_brief=worker_brief,
    )
    response = litellm.completion(model=model, messages=messages)
    raw_text = response.choices[0].message.content or ""
    summary, section_name, replacement_code = extract_summary_section_and_code(raw_text)
    if worker_brief is not None and section_name != worker_brief.section:
        raise ValueError(
            f"worker brief required section {worker_brief.section!r}, but model returned {section_name!r}"
        )
    replacement_error = _validate_section_replacement(section_name, replacement_code, candidate_code)
    if replacement_error is not None:
        raise ValueError(replacement_error)
    updated_code = apply_section_edit(candidate_code, section_name, replacement_code)
    return f"[{section_name}] {summary}", updated_code, raw_text


def build_worker_briefs_prompt(
    *,
    program_text: str,
    strategy_text: str,
    memory_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    worker_count: int,
) -> list[dict[str, str]]:
    recent_text = "\n".join(
        f"- attempt {row['attempt']} [{row['status']}]: acc={row.get('mean_accuracy')} time={row.get('mean_time_seconds')} failure={row.get('failure_type')} summary={row.get('change_summary')}"
        for row in recent_rows[-10:]
    ) or "- no recent attempts"
    allowed_sections = "\n".join(f"- {spec.name}: {spec.description}" for spec in SECTION_SPECS)
    system = (
        "You are coordinating a batch of parallel autoresearch workers. "
        "Return a JSON array of exactly "
        f"{worker_count} worker briefs. "
        "Each brief must define one distinct experiment. "
        "Every brief must contain keys: title, section, family, hypothesis, instructions. "
        "section must be one of the allowed sections. "
        "Make the briefs meaningfully different; do not issue five variations of the same scalar tweak unless recent evidence strongly demands it. "
        "Keep each brief short, concrete, and technically coherent."
    )
    user = (
        f"Program instructions:\n{program_text}\n\n"
        f"Current strategy:\n{strategy_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Recent attempts:\n{recent_text}\n\n"
        f"Allowed sections:\n{allowed_sections}\n\n"
        f"Current editable sections:\n{section_inventory_text(candidate_code)}\n\n"
        "Return only the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def propose_worker_briefs(
    *,
    model: str,
    program_text: str,
    strategy_text: str,
    memory_text: str,
    candidate_code: str,
    recent_rows: list[dict[str, Any]],
    worker_count: int,
) -> tuple[list[WorkerBrief], str]:
    import litellm

    messages = build_worker_briefs_prompt(
        program_text=program_text,
        strategy_text=strategy_text,
        memory_text=memory_text,
        candidate_code=candidate_code,
        recent_rows=recent_rows,
        worker_count=worker_count,
    )
    response = litellm.completion(model=model, messages=messages)
    raw_text = response.choices[0].message.content or ""
    payload = _extract_json_payload(raw_text)
    if not isinstance(payload, list) or len(payload) != worker_count:
        raise ValueError(f"expected JSON array of {worker_count} worker briefs")

    briefs: list[WorkerBrief] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("worker brief entries must be objects")
        section = str(item.get("section", "")).strip()
        if section not in SECTION_SPECS_BY_NAME:
            raise ValueError(f"worker brief used unknown section {section!r}")
        brief = WorkerBrief(
            title=str(item.get("title", "")).strip(),
            section=section,
            family=str(item.get("family", "")).strip(),
            hypothesis=str(item.get("hypothesis", "")).strip(),
            instructions=str(item.get("instructions", "")).strip(),
        )
        if not all((brief.title, brief.family, brief.hypothesis, brief.instructions)):
            raise ValueError("worker brief fields must be non-empty")
        briefs.append(brief)
    return briefs, raw_text


def build_strategy_prompt(
    *,
    program_text: str,
    current_strategy_text: str,
    memory_text: str,
    round_summary: dict[str, Any],
    round_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    recent_attempts = []
    for row in round_rows[-10:]:
        recent_attempts.append(
            f"- attempt {row.get('attempt')} [{row.get('status')}]: "
            f"acc={row.get('mean_accuracy')} time={row.get('mean_time_seconds')} "
            f"failure={row.get('failure_type')} summary={row.get('change_summary')}"
        )
    recent_text = "\n".join(recent_attempts) or "- no attempts recorded"
    summary_text = json.dumps(round_summary, indent=2, sort_keys=True, default=str)
    system = (
        "You are revising strategy.md for the next batch of experiments in a two-layer keep/discard optimization loop. "
        "You are not editing candidate.py. "
        "Keep the strategy concise, practical, and focused on a few experiment families or hypotheses to try next. "
        "Assume the wrapper contract, CLI flags, JSON output, and benchmark semantics are fixed and should not be targets. "
        "Return markdown only, no code fences."
    )
    user = (
        f"Program instructions:\n{program_text}\n\n"
        f"Current strategy.md:\n{current_strategy_text}\n\n"
        f"Current memory:\n{memory_text}\n\n"
        f"Last round summary:\n{summary_text}\n\n"
        f"Last round attempts:\n{recent_text}\n\n"
        "Rewrite strategy.md for the next round. Keep it short and actionable."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def propose_strategy_update(
    *,
    model: str,
    program_text: str,
    current_strategy_text: str,
    memory_text: str,
    round_summary: dict[str, Any],
    round_rows: list[dict[str, str]],
) -> tuple[str, str]:
    import litellm

    messages = build_strategy_prompt(
        program_text=program_text,
        current_strategy_text=current_strategy_text,
        memory_text=memory_text,
        round_summary=round_summary,
        round_rows=round_rows,
    )
    response = litellm.completion(model=model, messages=messages)
    raw_text = response.choices[0].message.content or ""
    strategy_text = raw_text.strip()
    if not strategy_text:
        raise ValueError("model response did not include strategy text")
    return strategy_text, raw_text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def is_infra_failure(result: Any) -> bool:
    return result.failure_type == "gpu_mismatch"


@dataclass(frozen=True)
class AutoresearchLoopConfig:
    candidate_path: Path
    program_path: Path
    strategy_path: Path
    memory_path: Path
    incumbent_record_path: Path
    run_dir: Path
    model: str
    max_attempts: int
    final_strict_eval: bool = True
    strategy_rounds: int = 1
    strategy_model: str | None = None


def run_autoresearch_loop(
    config: AutoresearchLoopConfig,
    evaluate_proxy: Callable[[str], Any],
    evaluate_strict: Callable[[str], Any],
    *,
    logger: Callable[[str], None] = print,
) -> int:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = config.run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    results_writer = build_results_writer(config.run_dir / "results.tsv")
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    incumbent_code = config.candidate_path.read_text(encoding="utf-8")
    incumbent_sha = text_sha256(incumbent_code)
    start_time = time.time()

    def write_partial_summary(*, incumbent_proxy: Any, incumbent_strict: Any | None, infra_failures: int) -> None:
        payload = {
            "model": config.model,
            "attempts_completed": len([row for row in all_rows if row["phase"] == "proxy"]),
            "kept_attempts": len([row for row in all_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
            "discarded_attempts": len([row for row in all_rows if row["status"] == "discard"]),
            "crashed_attempts": len([row for row in all_rows if row["status"] == "crash"]),
            "infra_failures": infra_failures,
            "incumbent_sha256": incumbent_sha,
            "incumbent_mean_accuracy_proxy": incumbent_proxy.mean_accuracy,
            "incumbent_mean_time_seconds_proxy": incumbent_proxy.mean_time_seconds,
            "incumbent_valid_proxy": incumbent_proxy.valid,
            "incumbent_mean_accuracy_strict": incumbent_strict.mean_accuracy if incumbent_strict else None,
            "incumbent_mean_time_seconds_strict": incumbent_strict.mean_time_seconds if incumbent_strict else None,
            "incumbent_valid_strict": incumbent_strict.valid if incumbent_strict else None,
            "elapsed_wall_clock_seconds": time.time() - start_time,
            "run_dir": str(config.run_dir),
            "terminated_early": True,
            "termination_reason": "gpu_mismatch",
        }
        write_json(config.run_dir / "summary.json", payload)

    try:
        logger("[loop] loading incumbent-of-record")
        incumbent_proxy_result, incumbent_strict_result = load_incumbent_record(
            config.incumbent_record_path,
            incumbent_sha,
        )
        incumbent_record_payload = serialize_incumbent_record(
            incumbent_sha,
            incumbent_proxy_result,
            incumbent_strict_result,
        )
        write_json(config.run_dir / "incumbent_record.json", incumbent_record_payload)
        seed_proxy_row = eval_row(
            incumbent_proxy_result,
            attempt=0,
            phase="baseline_proxy_record",
            status="loaded",
            candidate_sha=incumbent_sha,
            parent_sha="",
            change_summary="loaded validated incumbent proxy record",
        )
        results_writer.writerow(seed_proxy_row)
        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
        all_rows.append(seed_proxy_row)
        seed_strict_row = eval_row(
            incumbent_strict_result,
            attempt=0,
            phase="baseline_strict_record",
            status="loaded",
            candidate_sha=incumbent_sha,
            parent_sha="",
            change_summary="loaded validated incumbent strict record",
        )
        results_writer.writerow(seed_strict_row)
        getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
        all_rows.append(seed_strict_row)
        accepted_rows.append(seed_strict_row)
        (config.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
        update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

        infra_failures = 0

        for attempt in range(1, config.max_attempts + 1):
            logger(f"[loop] attempt {attempt}/{config.max_attempts}: proposing edit")
            program_text = config.program_path.read_text(encoding="utf-8")
            strategy_text = config.strategy_path.read_text(encoding="utf-8")
            memory_text = config.memory_path.read_text(encoding="utf-8")
            recent_rows = all_rows[-5:]
            try:
                summary, proposed_code, raw_response = propose_candidate(
                    model=config.model,
                    program_text=program_text,
                    strategy_text=strategy_text,
                    memory_text=memory_text,
                    candidate_code=incumbent_code,
                    recent_rows=recent_rows,
                )
            except Exception as exc:
                error_row = {
                    "attempt": attempt,
                    "phase": "proposal",
                    "status": "crash",
                    "candidate_sha256": "",
                    "parent_sha256": incumbent_sha,
                    "valid": False,
                    "meets_target": False,
                    "mean_accuracy": None,
                    "mean_time_seconds": None,
                    "score": 0.0,
                    "failure_type": exc.__class__.__name__,
                    "actual_device_name": None,
                    "runtime_seconds": None,
                    "remote_runtime_seconds": None,
                    "change_summary": str(exc),
                }
                results_writer.writerow(error_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(error_row)
                all_rows.append(error_row)
                logger(f"[loop] attempt {attempt}: proposal failed: {exc}")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            proposed_sha = text_sha256(proposed_code)
            if proposed_sha == incumbent_sha:
                no_change_row = {
                    "attempt": attempt,
                    "phase": "proxy",
                    "status": "discard",
                    "candidate_sha256": proposed_sha,
                    "parent_sha256": incumbent_sha,
                    "valid": True,
                    "meets_target": None,
                    "mean_accuracy": None,
                    "mean_time_seconds": None,
                    "score": 0.0,
                    "failure_type": "no_change",
                    "actual_device_name": None,
                    "runtime_seconds": 0.0,
                    "remote_runtime_seconds": None,
                    "change_summary": f"{summary} | no code change produced",
                }
                results_writer.writerow(no_change_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(no_change_row)
                all_rows.append(no_change_row)
                logger(f"[loop] attempt {attempt}: discard before eval (no code change)")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "proposal.raw.txt").write_text(raw_response, encoding="utf-8")
            (attempt_dir / "candidate.py").write_text(proposed_code, encoding="utf-8")
            (attempt_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

            try:
                compile(proposed_code, "<candidate>", "exec")
            except SyntaxError as exc:
                syntax_row = {
                    "attempt": attempt,
                    "phase": "proxy",
                    "status": "crash",
                    "candidate_sha256": proposed_sha,
                    "parent_sha256": incumbent_sha,
                    "valid": False,
                    "meets_target": False,
                    "mean_accuracy": None,
                    "mean_time_seconds": None,
                    "score": 0.0,
                    "failure_type": "syntax_error",
                    "actual_device_name": None,
                    "runtime_seconds": 0.0,
                    "remote_runtime_seconds": None,
                    "change_summary": f"{summary} | syntax error: {exc}",
                }
                results_writer.writerow(syntax_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(syntax_row)
                all_rows.append(syntax_row)
                logger(f"[loop] attempt {attempt}: crash before eval (syntax error)")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            contract_error = validate_candidate_contract(proposed_code)
            if contract_error is not None:
                contract_row = {
                    "attempt": attempt,
                    "phase": "proxy",
                    "status": "crash",
                    "candidate_sha256": proposed_sha,
                    "parent_sha256": incumbent_sha,
                    "valid": False,
                    "meets_target": False,
                    "mean_accuracy": None,
                    "mean_time_seconds": None,
                    "score": 0.0,
                    "failure_type": "contract_error",
                    "actual_device_name": None,
                    "runtime_seconds": 0.0,
                    "remote_runtime_seconds": None,
                    "change_summary": f"{summary} | contract error: {contract_error}",
                }
                results_writer.writerow(contract_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                rejected_rows.append(contract_row)
                all_rows.append(contract_row)
                logger(f"[loop] attempt {attempt}: crash before eval (contract error)")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                continue

            config.candidate_path.write_text(proposed_code, encoding="utf-8")
            logger(f"[loop] attempt {attempt}: running proxy eval")
            proposed_result = evaluate_proxy(proposed_code)
            write_json(attempt_dir / "proxy_eval.json", proposed_result.as_side_info())
            if is_infra_failure(proposed_result):
                row = eval_row(
                    proposed_result,
                    attempt=attempt,
                    phase="proxy",
                    status="infra_fail",
                    candidate_sha=proposed_sha,
                    parent_sha=incumbent_sha,
                    change_summary=f"{summary} | infrastructure failure",
                )
                results_writer.writerow(row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                all_rows.append(row)
                rejected_rows.append(row)
                config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                infra_failures += 1
                logger(f"[loop] attempt {attempt}: gpu_mismatch after internal retries; aborting run")
                update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                write_partial_summary(
                    incumbent_proxy=incumbent_proxy_result,
                    incumbent_strict=incumbent_strict_result,
                    infra_failures=infra_failures,
                )
                return 1

            if is_better(proposed_result, incumbent_proxy_result):
                status = "candidate"
            elif proposed_result.valid:
                status = "discard"
            else:
                status = "crash"
            row = eval_row(
                proposed_result,
                attempt=attempt,
                phase="proxy",
                status=status,
                candidate_sha=proposed_sha,
                parent_sha=incumbent_sha,
                change_summary=summary,
            )
            results_writer.writerow(row)
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(row)

            if status == "candidate":
                logger(f"[loop] attempt {attempt}: proxy candidate found, running strict confirmation")
                proposed_strict_result = evaluate_strict(proposed_code)
                write_json(attempt_dir / "strict_eval.json", proposed_strict_result.as_side_info())
                if is_infra_failure(proposed_strict_result):
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status="infra_fail",
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=f"{summary} | infrastructure failure during strict confirmation",
                    )
                    results_writer.writerow(strict_row)
                    getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                    all_rows.append(strict_row)
                    rejected_rows.append(strict_row)
                    config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                    infra_failures += 1
                    logger(f"[loop] attempt {attempt}: gpu_mismatch during strict confirmation; aborting run")
                    update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)
                    write_partial_summary(
                        incumbent_proxy=incumbent_proxy_result,
                        incumbent_strict=incumbent_strict_result,
                        infra_failures=infra_failures,
                    )
                    return 1
                if is_better(proposed_strict_result, incumbent_strict_result):
                    strict_status = "keep"
                    incumbent_code = proposed_code
                    incumbent_sha = proposed_sha
                    incumbent_proxy_result = proposed_result
                    incumbent_strict_result = proposed_strict_result
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status=strict_status,
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=summary,
                    )
                    accepted_rows.append(strict_row)
                    (config.run_dir / "incumbent.py").write_text(incumbent_code, encoding="utf-8")
                    incumbent_record_payload = serialize_incumbent_record(
                        incumbent_sha,
                        incumbent_proxy_result,
                        incumbent_strict_result,
                    )
                    write_json(config.incumbent_record_path, incumbent_record_payload)
                    write_json(config.run_dir / "incumbent_record.json", incumbent_record_payload)
                    logger(
                        f"[loop] attempt {attempt}: keep after strict confirm "
                        f"acc={proposed_strict_result.mean_accuracy} time={proposed_strict_result.mean_time_seconds}"
                    )
                else:
                    strict_status = "discard" if proposed_strict_result.valid else "crash"
                    strict_row = eval_row(
                        proposed_strict_result,
                        attempt=attempt,
                        phase="strict_confirm",
                        status=strict_status,
                        candidate_sha=proposed_sha,
                        parent_sha=row["parent_sha256"],
                        change_summary=f"{summary} | failed strict confirmation",
                    )
                    rejected_rows.append(strict_row)
                    config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                    logger(
                        f"[loop] attempt {attempt}: {strict_status} after strict confirm "
                        f"acc={proposed_strict_result.mean_accuracy} time={proposed_strict_result.mean_time_seconds} "
                        f"failure={proposed_strict_result.failure_type}"
                    )
                results_writer.writerow(strict_row)
                getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
                all_rows.append(strict_row)
            else:
                rejected_rows.append(row)
                config.candidate_path.write_text(incumbent_code, encoding="utf-8")
                logger(
                    f"[loop] attempt {attempt}: {status} acc={proposed_result.mean_accuracy} "
                    f"time={proposed_result.mean_time_seconds} failure={proposed_result.failure_type}"
                )

            update_memory(config.memory_path, incumbent_strict_result, accepted_rows, rejected_rows)

        final_payload = {
            "model": config.model,
            "attempts_completed": len([row for row in all_rows if row["phase"] == "proxy"]),
            "kept_attempts": len([row for row in all_rows if row["phase"] == "strict_confirm" and row["status"] == "keep"]),
            "discarded_attempts": len([row for row in all_rows if row["status"] == "discard"]),
            "crashed_attempts": len([row for row in all_rows if row["status"] == "crash"]),
            "infra_failures": infra_failures,
            "incumbent_sha256": incumbent_sha,
            "incumbent_mean_accuracy_proxy": incumbent_proxy_result.mean_accuracy,
            "incumbent_mean_time_seconds_proxy": incumbent_proxy_result.mean_time_seconds,
            "incumbent_valid_proxy": incumbent_proxy_result.valid,
            "incumbent_mean_accuracy_strict": incumbent_strict_result.mean_accuracy,
            "incumbent_mean_time_seconds_strict": incumbent_strict_result.mean_time_seconds,
            "incumbent_valid_strict": incumbent_strict_result.valid,
            "elapsed_wall_clock_seconds": time.time() - start_time,
            "run_dir": str(config.run_dir),
        }

        if config.final_strict_eval:
            logger("[loop] running final strict evaluation on incumbent")
            strict_result = evaluate_strict(incumbent_code)
            write_json(config.run_dir / "final_strict_eval.json", strict_result.as_side_info())
            strict_row = eval_row(
                strict_result,
                attempt=config.max_attempts + 1,
                phase="final_strict",
                status="final",
                candidate_sha=incumbent_sha,
                parent_sha=incumbent_sha,
                change_summary="final incumbent verification",
            )
            results_writer.writerow(strict_row)
            getattr(results_writer, "_handle").flush()  # type: ignore[attr-defined]
            all_rows.append(strict_row)
            final_payload.update(
                {
                    "final_strict_valid": strict_result.valid,
                    "final_strict_mean_accuracy": strict_result.mean_accuracy,
                    "final_strict_mean_time_seconds": strict_result.mean_time_seconds,
                    "final_strict_score": strict_result.score,
                }
            )

        write_json(config.run_dir / "summary.json", final_payload)
        return 0
    finally:
        close_results_writer(results_writer)


def run_meta_autoresearch_loop(
    config: AutoresearchLoopConfig,
    evaluate_proxy: Callable[[str], Any],
    evaluate_strict: Callable[[str], Any],
    *,
    logger: Callable[[str], None] = print,
) -> int:
    # Outer loop: run a batch of bounded inner experiments, then rewrite
    # strategy.md for the next batch based on the round summary and ledger.
    if config.strategy_rounds <= 1:
        return run_autoresearch_loop(config, evaluate_proxy, evaluate_strict, logger=logger)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    strategy_history_dir = config.run_dir / "strategy_history"
    strategy_history_dir.mkdir(parents=True, exist_ok=True)
    (strategy_history_dir / "round_00_start.md").write_text(
        config.strategy_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    strategy_model = config.strategy_model or config.model
    campaign_rounds: list[dict[str, Any]] = []
    exit_code = 0

    for round_idx in range(1, config.strategy_rounds + 1):
        round_dir = config.run_dir / f"round_{round_idx:02d}"
        logger(
            f"[meta] round {round_idx}/{config.strategy_rounds}: "
            f"running {config.max_attempts} inner attempts"
        )
        round_cfg = AutoresearchLoopConfig(
            candidate_path=config.candidate_path,
            program_path=config.program_path,
            strategy_path=config.strategy_path,
            memory_path=config.memory_path,
            incumbent_record_path=config.incumbent_record_path,
            run_dir=round_dir,
            model=config.model,
            max_attempts=config.max_attempts,
            final_strict_eval=False,
            strategy_rounds=1,
            strategy_model=None,
        )
        round_exit_code = run_autoresearch_loop(round_cfg, evaluate_proxy, evaluate_strict, logger=logger)
        exit_code = round_exit_code

        round_summary_path = round_dir / "summary.json"
        round_summary = json.loads(round_summary_path.read_text(encoding="utf-8")) if round_summary_path.exists() else {}
        round_rows = load_results_rows(round_dir / "results.tsv")
        campaign_rounds.append(
            {
                "round": round_idx,
                "exit_code": round_exit_code,
                "run_dir": str(round_dir),
                "summary": round_summary,
            }
        )

        if round_exit_code != 0:
            logger(f"[meta] round {round_idx}: terminating campaign after non-zero exit code")
            break

        if round_idx >= config.strategy_rounds:
            break

        logger(f"[meta] round {round_idx}: revising strategy")
        program_text = config.program_path.read_text(encoding="utf-8")
        current_strategy_text = config.strategy_path.read_text(encoding="utf-8")
        memory_text = config.memory_path.read_text(encoding="utf-8")
        try:
            new_strategy_text, raw_strategy_response = propose_strategy_update(
                model=strategy_model,
                program_text=program_text,
                current_strategy_text=current_strategy_text,
                memory_text=memory_text,
                round_summary=round_summary,
                round_rows=round_rows,
            )
        except Exception as exc:
            failure_path = strategy_history_dir / f"round_{round_idx:02d}_strategy_error.txt"
            failure_path.write_text(str(exc) + "\n", encoding="utf-8")
            logger(f"[meta] round {round_idx}: strategy update failed: {exc}")
            continue

        (strategy_history_dir / f"round_{round_idx:02d}_strategy.raw.txt").write_text(
            raw_strategy_response,
            encoding="utf-8",
        )
        config.strategy_path.write_text(new_strategy_text.rstrip() + "\n", encoding="utf-8")
        (strategy_history_dir / f"round_{round_idx:02d}_strategy.md").write_text(
            new_strategy_text.rstrip() + "\n",
            encoding="utf-8",
        )

    final_record = json.loads(config.incumbent_record_path.read_text(encoding="utf-8"))
    final_summary = {
        "model": config.model,
        "strategy_model": strategy_model,
        "strategy_rounds": config.strategy_rounds,
        "attempts_per_round": config.max_attempts,
        "exit_code": exit_code,
        "rounds": campaign_rounds,
        "final_incumbent_sha256": final_record.get("candidate_sha256"),
        "final_incumbent_strict": final_record.get("strict"),
        "run_dir": str(config.run_dir),
    }
    write_json(config.run_dir / "summary.json", final_summary)
    return exit_code
