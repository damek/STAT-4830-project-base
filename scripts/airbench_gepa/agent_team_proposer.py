#!/usr/bin/env python3
"""Multi-agent candidate proposer for AirBench GEPA runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LanguageModel = Callable[[str | list[dict[str, Any]]], str]

CODE_SECTION_ORDER = (
    "setup_code",
    "data_code",
    "model_code",
    "train_eval_code",
    "cli_code",
)

SECTION_SENTINELS = (
    ("data_code", "class CifarLoader:"),
    ("model_code", "class BatchNorm(nn.BatchNorm2d):"),
    ("train_eval_code", "def infer(model, loader, tta_level=0):"),
    ("cli_code", "def parse_args() -> argparse.Namespace:"),
)


def split_solver_code(source: str) -> dict[str, str]:
    starts: dict[str, int] = {}
    for section_name, sentinel in SECTION_SENTINELS:
        try:
            starts[section_name] = source.index(sentinel)
        except ValueError as exc:
            raise ValueError(f"Could not find sentinel {sentinel!r} for section {section_name!r}") from exc

    setup_end = starts["data_code"]
    data_end = starts["model_code"]
    model_end = starts["train_eval_code"]
    train_eval_end = starts["cli_code"]
    sections = {
        "setup_code": source[:setup_end],
        "data_code": source[starts["data_code"] : data_end],
        "model_code": source[starts["model_code"] : model_end],
        "train_eval_code": source[starts["train_eval_code"] : train_eval_end],
        "cli_code": source[starts["cli_code"] :],
    }
    if assemble_solver_code(sections) != source:
        raise ValueError("Section assembly did not reproduce the input solver exactly.")
    return sections


def assemble_solver_code(sections: Mapping[str, str]) -> str:
    return "".join(sections[name] for name in CODE_SECTION_ORDER)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
            return "\n".join(lines[1:]).strip()
    return stripped


def _load_json(text: str) -> dict[str, Any]:
    candidate = _strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(candidate):
            if ch not in "[{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]"


def _section_inventory(sections: Mapping[str, str]) -> str:
    rows: list[str] = []
    for name in CODE_SECTION_ORDER:
        code = sections[name]
        lines = code.splitlines()
        header = next((line.strip() for line in lines if line.strip()), "<empty>")
        rows.append(f"- {name}: {len(lines)} lines, starts with `{header[:120]}`")
    return "\n".join(rows)


def _selected_sections_block(sections: Mapping[str, str], names: Sequence[str]) -> str:
    rendered: list[str] = []
    for name in names:
        code = sections.get(name, "")
        rendered.append(f"## {name}\n```python\n{code}\n```")
    return "\n\n".join(rendered)


def _review_candidate_block(source: str) -> str:
    try:
        sections = split_solver_code(source)
    except Exception:
        return f"```python\n{source}\n```"
    return _selected_sections_block(sections, CODE_SECTION_ORDER)


def _compact_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_call": row.get("metric_call"),
        "candidate_kind": row.get("candidate_kind"),
        "score": row.get("score"),
        "mean_accuracy": row.get("mean_accuracy"),
        "mean_time_seconds": row.get("mean_time_seconds"),
        "failure_type": row.get("failure_type"),
        "meets_target": row.get("meets_target"),
        "same_as_seed": row.get("same_as_seed"),
    }


def _derive_history_lessons(
    window_rows: Sequence[Mapping[str, Any]],
    *,
    failure_counts: Mapping[str, int],
    target_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    lessons: list[str] = []

    repeated_failures = [name for name, count in sorted(failure_counts.items()) if count >= 2]
    for failure_name in repeated_failures:
        lessons.append(
            f"Avoid repeating {failure_name}: add a local check or narrower edits before changing risky logic again."
        )

    if target_rows:
        fastest_target = min(
            (row for row in target_rows if row.get("mean_time_seconds") is not None),
            key=lambda row: float(row["mean_time_seconds"]),
            default=target_rows[0],
        )
        lessons.append(
            "Preserve invariants from target-meeting candidates, especially around CLI/JSON contract and stable "
            f"compiled evaluation paths (best target metric_call={fastest_target.get('metric_call')})."
        )
    elif window_rows and all(row.get("failure_type") in (None, "") for row in window_rows):
        lessons.append(
            "Recent candidates mostly ran successfully but missed the target, so prioritize speed/accuracy tradeoffs "
            "over broad reliability rewrites."
        )

    repeat_count = sum(1 for row in window_rows if row.get("candidate_kind") == "repeat")
    if repeat_count >= 2:
        lessons.append("Recent search is repeating itself; make narrower, section-specific edits instead of broad rewrites.")

    if not lessons:
        lessons.append("History is limited; rely on the current evaluation feedback and keep edits localized.")

    return lessons[:6]


def _build_history_summary(
    eval_history: Sequence[dict[str, Any]],
    *,
    history_window: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "history_enabled": False,
            "history_window_used": 0,
            "num_evaluations_considered": 0,
            "message": "History-aware proposer disabled. Use only the current evaluation feedback.",
            "derived_lessons": ["History disabled; do not infer trends from prior evaluations."],
        }

    window_rows = list(eval_history[-max(0, history_window) :]) if history_window > 0 else []
    failure_counts: dict[str, int] = {}
    for row in window_rows:
        failure_type = row.get("failure_type")
        if failure_type:
            failure_counts[str(failure_type)] = failure_counts.get(str(failure_type), 0) + 1

    target_rows = [row for row in window_rows if row.get("meets_target")]
    best_score_row = max(window_rows, key=lambda row: float(row.get("score", float("-inf"))), default=None)
    best_acc_row = max(
        (row for row in window_rows if row.get("mean_accuracy") is not None),
        key=lambda row: float(row["mean_accuracy"]),
        default=None,
    )

    return {
        "history_enabled": True,
        "history_window_used": min(len(eval_history), max(0, history_window)),
        "num_evaluations_considered": len(window_rows),
        "best_score_candidate": _compact_history_row(best_score_row) if best_score_row is not None else None,
        "best_accuracy_candidate": _compact_history_row(best_acc_row) if best_acc_row is not None else None,
        "target_meeting_candidates": [_compact_history_row(row) for row in target_rows[:3]],
        "failure_counts": failure_counts,
        "repeat_candidate_count": sum(1 for row in window_rows if row.get("candidate_kind") == "repeat"),
        "recent_trajectory": [_compact_history_row(row) for row in window_rows],
        "derived_lessons": _derive_history_lessons(
            window_rows,
            failure_counts=failure_counts,
            target_rows=target_rows,
        ),
    }


@dataclass
class AgentTeamSession:
    session_id: int
    run_dir: Path

    def write_json(self, name: str, payload: Any) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / name).write_text(_safe_json(payload), encoding="utf-8")

    def write_text(self, name: str, content: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / name).write_text(content, encoding="utf-8")


class AgentTeamProposer:
    def __init__(
        self,
        *,
        lm: LanguageModel,
        objective: str,
        background: str,
        eval_history_ref: Sequence[dict[str, Any]],
        run_dir: Path,
        max_rounds: int = 2,
        history_window: int = 8,
        history_enabled: bool = True,
    ) -> None:
        self.lm = lm
        self.objective = objective
        self.background = background
        self.eval_history_ref = eval_history_ref
        self.run_dir = run_dir / "agent_team"
        self.max_rounds = max_rounds
        self.history_window = history_window
        self.history_enabled = history_enabled
        self._session_counter = 0

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        del components_to_update  # Internal team decides which code sections to modify.

        solver_code = candidate.get("solver_code", "")
        if not solver_code.strip():
            return {"solver_code": solver_code}

        self._session_counter += 1
        session = AgentTeamSession(
            session_id=self._session_counter,
            run_dir=self.run_dir / f"session_{self._session_counter:03d}",
        )
        session.write_text("input_solver.py", solver_code)

        try:
            working_sections = split_solver_code(solver_code)
        except Exception as exc:
            session.write_json("split_error.json", {"error": str(exc)})
            print(f"[team {self._session_counter:03d}] split failed: {exc}")
            return {"solver_code": solver_code}

        current_feedback = reflective_dataset.get("solver_code", [])
        current_feedback_text = _safe_json(list(current_feedback))
        history_summary = _build_history_summary(
            self.eval_history_ref,
            history_window=self.history_window,
            enabled=self.history_enabled,
        )
        history_text = _safe_json(history_summary)
        reviewer_feedback = "No reviewer feedback yet."

        print(f"[team {self._session_counter:03d}] coordinator session started with {self.max_rounds} round(s)")

        for round_idx in range(1, self.max_rounds + 1):
            try:
                session.write_json(f"round_{round_idx:02d}_history.json", history_summary)
                coordinator_plan = self._run_coordinator(
                    session,
                    round_idx=round_idx,
                    sections=working_sections,
                    current_feedback_text=current_feedback_text,
                    history_text=history_text,
                    reviewer_feedback=reviewer_feedback,
                )
                perf_sections = coordinator_plan.get("performance_focus_sections") or ["train_eval_code", "model_code"]
                rel_sections = coordinator_plan.get("reliability_focus_sections") or ["data_code", "cli_code"]
                perf_sections = [name for name in perf_sections if name in CODE_SECTION_ORDER]
                rel_sections = [name for name in rel_sections if name in CODE_SECTION_ORDER]
                if not perf_sections:
                    perf_sections = ["train_eval_code"]
                if not rel_sections:
                    rel_sections = ["data_code"]

                perf_result = self._run_worker(
                    session,
                    role_name="performance",
                    round_idx=round_idx,
                    focus_sections=perf_sections,
                    sections=working_sections,
                    current_feedback_text=current_feedback_text,
                    history_text=history_text,
                    task_brief=str(coordinator_plan.get("performance_task", "")),
                    round_goal=str(coordinator_plan.get("round_goal", "")),
                )
                rel_result = self._run_worker(
                    session,
                    role_name="reliability",
                    round_idx=round_idx,
                    focus_sections=rel_sections,
                    sections=working_sections,
                    current_feedback_text=current_feedback_text,
                    history_text=history_text,
                    task_brief=str(coordinator_plan.get("reliability_task", "")),
                    round_goal=str(coordinator_plan.get("round_goal", "")),
                )

                integrated_sections = self._run_integrator(
                    session,
                    round_idx=round_idx,
                    sections=working_sections,
                    current_feedback_text=current_feedback_text,
                    coordinator_plan=coordinator_plan,
                    worker_outputs={"performance": perf_result, "reliability": rel_result},
                )

                integrated_solver = assemble_solver_code(integrated_sections)
                session.write_text(f"round_{round_idx:02d}_candidate.py", integrated_solver)

                syntax_error = self._syntax_error_message(integrated_solver)
                reviewer_result = self._run_reviewer(
                    session,
                    round_idx=round_idx,
                    solver_code=integrated_solver,
                    current_feedback_text=current_feedback_text,
                    history_text=history_text,
                    syntax_error=syntax_error,
                )

                working_sections = integrated_sections
                reviewer_feedback = str(reviewer_result.get("summary", ""))
                if syntax_error:
                    reviewer_feedback = f"Syntax error detected locally: {syntax_error}\n{reviewer_feedback}".strip()

                print(
                    f"[team {self._session_counter:03d}] round={round_idx} "
                    f"approve={bool(reviewer_result.get('approve'))} "
                    f"syntax_error={bool(syntax_error)}"
                )

                if reviewer_result.get("approve") and syntax_error is None:
                    break
            except Exception as exc:
                session.write_json(f"round_{round_idx:02d}_error.json", {"error": str(exc)})
                print(f"[team {self._session_counter:03d}] round={round_idx} failed: {exc}")
                break

        final_solver = assemble_solver_code(working_sections)
        session.write_text("final_candidate.py", final_solver)
        return {"solver_code": final_solver}

    def _chat_json(self, session: AgentTeamSession, raw_name: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raw = self.lm(messages)
        session.write_text(raw_name, raw)
        return _load_json(raw)

    def _run_coordinator(
        self,
        session: AgentTeamSession,
        *,
        round_idx: int,
        sections: Mapping[str, str],
        current_feedback_text: str,
        history_text: str,
        reviewer_feedback: str,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the coordinator of a small code-optimization team. "
                    "Your job is to decide which code sections deserve attention this round, "
                    "balancing speed improvements against runtime correctness."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Optimization objective:\n{self.objective}\n\n"
                    f"Domain background:\n{self.background}\n\n"
                    f"Persistent history side information:\n```json\n{history_text}\n```\n\n"
                    f"Current evaluation feedback:\n```json\n{current_feedback_text}\n```\n\n"
                    f"Latest reviewer feedback:\n{reviewer_feedback}\n\n"
                    f"Current code section inventory:\n{_section_inventory(sections)}\n\n"
                    "Return JSON with keys:\n"
                    "- round_goal: string\n"
                    "- performance_focus_sections: list of section names\n"
                    "- reliability_focus_sections: list of section names\n"
                    "- performance_task: string\n"
                    "- reliability_task: string\n"
                    "- integration_notes: string\n"
                    "Only use these section names: "
                    f"{list(CODE_SECTION_ORDER)}"
                ),
            },
        ]
        result = self._chat_json(session, f"round_{round_idx:02d}_coordinator.raw.txt", messages)
        session.write_json(f"round_{round_idx:02d}_coordinator.json", result)
        return result

    def _run_worker(
        self,
        session: AgentTeamSession,
        *,
        role_name: str,
        round_idx: int,
        focus_sections: Sequence[str],
        sections: Mapping[str, str],
        current_feedback_text: str,
        history_text: str,
        task_brief: str,
        round_goal: str,
    ) -> dict[str, Any]:
        role_instructions = {
            "performance": (
                "You are the performance specialist. Make targeted speed-oriented changes, "
                "but do not knowingly violate correctness or the CLI/JSON contract."
            ),
            "reliability": (
                "You are the reliability specialist. Focus on dtype safety, compile safety, "
                "contract preservation, and avoiding semantic collapse."
            ),
        }
        messages = [
            {"role": "system", "content": role_instructions[role_name]},
            {
                "role": "user",
                "content": (
                    f"Objective:\n{self.objective}\n\n"
                    f"Round goal:\n{round_goal}\n\n"
                    f"Task brief:\n{task_brief}\n\n"
                    f"Persistent history side information:\n```json\n{history_text}\n```\n\n"
                    f"Current evaluation feedback:\n```json\n{current_feedback_text}\n```\n\n"
                    f"Focus sections:\n{', '.join(focus_sections)}\n\n"
                    f"{_selected_sections_block(sections, focus_sections)}\n\n"
                    "Return JSON with keys:\n"
                    "- notes: short string\n"
                    "- edits: list of objects, each with:\n"
                    "  - section: one of the focus sections\n"
                    "  - reason: short string\n"
                    "  - new_content: complete replacement text for that section\n"
                    "If no good edit is available, return edits as an empty list."
                ),
            },
        ]
        result = self._chat_json(session, f"round_{round_idx:02d}_{role_name}.raw.txt", messages)
        session.write_json(f"round_{round_idx:02d}_{role_name}.json", result)
        return result

    def _run_integrator(
        self,
        session: AgentTeamSession,
        *,
        round_idx: int,
        sections: Mapping[str, str],
        current_feedback_text: str,
        coordinator_plan: Mapping[str, Any],
        worker_outputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, str]:
        section_blocks = []
        for name in CODE_SECTION_ORDER:
            section_blocks.append(f"## {name}\n```python\n{sections[name]}\n```")
        rendered_sections = "\n\n".join(section_blocks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the integrator. Produce a clean full-program update by selectively "
                    "replacing only the sections that need to change. Keep unaffected sections intact."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective:\n{self.objective}\n\n"
                    f"Domain background:\n{self.background}\n\n"
                    f"Current evaluation feedback:\n```json\n{current_feedback_text}\n```\n\n"
                    f"Coordinator plan:\n```json\n{_safe_json(coordinator_plan)}\n```\n\n"
                    f"Worker outputs:\n```json\n{_safe_json(worker_outputs)}\n```\n\n"
                    f"Current sections:\n\n{rendered_sections}\n\n"
                    "Return JSON with keys:\n"
                    "- updated_sections: object mapping section name to its full replacement text\n"
                    "- notes: short string\n"
                    "Only include sections that actually change. "
                    f"Valid section names: {list(CODE_SECTION_ORDER)}"
                ),
            },
        ]
        result = self._chat_json(session, f"round_{round_idx:02d}_integrator.raw.txt", messages)
        session.write_json(f"round_{round_idx:02d}_integrator.json", result)
        merged = dict(sections)
        updated_sections = result.get("updated_sections", {})
        if isinstance(updated_sections, dict):
            for name, value in updated_sections.items():
                if name in merged and isinstance(value, str) and value.strip():
                    merged[name] = value
        return merged

    def _run_reviewer(
        self,
        session: AgentTeamSession,
        *,
        round_idx: int,
        solver_code: str,
        current_feedback_text: str,
        history_text: str,
        syntax_error: str | None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the reviewer. Your role is to look for breakage, likely regressions, "
                    "contract violations, and repeated failure modes before this candidate is evaluated."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective:\n{self.objective}\n\n"
                    f"Persistent history side information:\n```json\n{history_text}\n```\n\n"
                    f"Current evaluation feedback from parent candidate:\n```json\n{current_feedback_text}\n```\n\n"
                    f"Local syntax check result:\n{syntax_error or 'No syntax error detected.'}\n\n"
                    "Do not infer truncation from prompt formatting alone; if you see full section blocks "
                    "and local syntax says OK, do not claim the file is truncated unless code is actually missing.\n\n"
                    f"Candidate to review by section:\n{_review_candidate_block(solver_code)}\n\n"
                    "Return JSON with keys:\n"
                    "- approve: boolean\n"
                    "- summary: short string\n"
                    "- issues: list of short strings\n"
                ),
            },
        ]
        result = self._chat_json(session, f"round_{round_idx:02d}_reviewer.raw.txt", messages)
        session.write_json(f"round_{round_idx:02d}_reviewer.json", result)
        return result

    @staticmethod
    def _syntax_error_message(source: str) -> str | None:
        try:
            compile(source, "<agent-team-candidate>", "exec")
        except SyntaxError as exc:
            return f"{exc.__class__.__name__}: {exc}"
        return None


def build_agent_team_candidate_proposer(
    *,
    lm: LanguageModel,
    objective: str,
    background: str,
    eval_history_ref: Sequence[dict[str, Any]],
    run_dir: Path,
    max_rounds: int = 2,
    history_window: int = 8,
    history_enabled: bool = True,
) -> Callable[[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]], dict[str, str]]:
    proposer = AgentTeamProposer(
        lm=lm,
        objective=objective,
        background=background,
        eval_history_ref=eval_history_ref,
        run_dir=run_dir,
        max_rounds=max_rounds,
        history_window=history_window,
        history_enabled=history_enabled,
    )
    return proposer
