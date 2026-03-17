#!/usr/bin/env python3
"""Evaluator for AirBench candidate scripts executed on Modal."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Protocol


def _normalize_accuracy_value(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


class RemoteRunner(Protocol):
    def remote(
        self,
        solver_code: str,
        script_args: list[str],
        timeout_seconds: int,
        print_subprocess_logs: bool = False,
    ) -> dict[str, Any]:
        """Run a candidate remotely and return a structured result."""


@dataclasses.dataclass(frozen=True)
class AirbenchEvalConfig:
    target_accuracy: float = 0.94
    trials: int = 1
    warmup_trials: int = 1
    timeout_seconds: int = 60 * 15
    remote_data_dir: str = "/vol/cifar10"
    preflight: bool = True
    candidate_verbose: bool = False
    stream_subprocess_logs: bool = False
    expected_device_marker: str | None = "40GB"
    gpu_mismatch_retries: int = 2

    def normalized_target_accuracy(self) -> float:
        return _normalize_accuracy_value(float(self.target_accuracy))


@dataclasses.dataclass(frozen=True)
class AirbenchEvalResult:
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
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

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


def build_script_args(config: AirbenchEvalConfig) -> list[str]:
    args = [
        "--data-dir",
        config.remote_data_dir,
        "--trials",
        str(config.trials),
        "--warmup-trials",
        str(config.warmup_trials),
        "--target-accuracy",
        str(config.normalized_target_accuracy()),
        "--json-only",
    ]
    if config.candidate_verbose:
        args.append("--verbose")
    if config.preflight:
        args.append("--preflight")
    return args


def _build_result(
    *,
    score: float,
    valid: bool,
    failure_type: str | None,
    message: str,
    runtime_seconds: float,
    mean_accuracy: float | None,
    mean_time_seconds: float | None,
    trials: int | None,
    stdout_tail: str,
    stderr_tail: str,
    extra: dict[str, Any] | None = None,
) -> AirbenchEvalResult:
    return AirbenchEvalResult(
        score=score,
        valid=valid,
        failure_type=failure_type,
        message=message,
        runtime_seconds=runtime_seconds,
        mean_accuracy=mean_accuracy,
        mean_time_seconds=mean_time_seconds,
        trials=trials,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        extra=extra or {},
    )


def eval_result_from_remote_response(
    remote_result: dict[str, Any] | None,
    *,
    config: AirbenchEvalConfig,
    elapsed: float,
    last_remote_exception: Exception | None = None,
    mismatch_attempts: int = 0,
) -> AirbenchEvalResult:
    if remote_result is None:
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="modal_error",
            message=f"Modal execution failed: {last_remote_exception}",
            runtime_seconds=elapsed,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail="",
            stderr_tail="",
        )

    stdout_tail = str(remote_result.get("stdout_tail", ""))
    stderr_tail = str(remote_result.get("stderr_tail", ""))

    if not remote_result.get("ok", False):
        return _build_result(
            score=0.0,
            valid=False,
            failure_type=str(remote_result.get("failure_type", "remote_failure")),
            message=str(remote_result.get("message", "candidate failed on Modal")),
            runtime_seconds=elapsed,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            extra={
                "remote_runtime_seconds": remote_result.get("runtime_seconds"),
                "requested_gpu": remote_result.get("requested_gpu"),
                "actual_device_name": remote_result.get("actual_device_name"),
                "gpu_mismatch_attempts": mismatch_attempts,
            },
        )

    payload = remote_result.get("result")
    if not isinstance(payload, dict):
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="invalid_payload",
            message="remote result did not contain a JSON payload",
            runtime_seconds=elapsed,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    try:
        mean_accuracy = _normalize_accuracy_value(float(payload["mean_accuracy"]))
        mean_time_seconds = float(payload["mean_time_seconds"])
        trials = int(payload["trials"])
    except (KeyError, TypeError, ValueError) as exc:
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="invalid_payload",
            message=f"remote payload missing required numeric fields: {exc}",
            runtime_seconds=elapsed,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            extra={"payload": payload},
        )

    actual_device_name = str(payload.get("_modal", {}).get("actual_device_name") or payload.get("device_name") or "")
    expected_device_marker = (config.expected_device_marker or "").upper()
    if expected_device_marker and expected_device_marker not in actual_device_name.upper():
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="gpu_mismatch",
            message=(
                f"expected device containing {config.expected_device_marker!r}, "
                f"but payload reported {actual_device_name or '<unknown>'}"
            ),
            runtime_seconds=elapsed,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=trials,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            extra={
                "payload": payload,
                "requested_gpu": payload.get("_modal", {}).get("requested_gpu"),
                "actual_device_name": actual_device_name,
                "gpu_mismatch_attempts": mismatch_attempts,
            },
        )

    if mean_time_seconds <= 0.0:
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="invalid_time",
            message="mean_time_seconds must be positive",
            runtime_seconds=elapsed,
            mean_accuracy=mean_accuracy,
            mean_time_seconds=mean_time_seconds,
            trials=trials,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            extra={"payload": payload},
        )

    target_accuracy = config.normalized_target_accuracy()
    accuracy_margin = mean_accuracy - target_accuracy
    meets_target = mean_accuracy >= target_accuracy
    inverse_time = 1.0 / mean_time_seconds
    score = 1000.0 + inverse_time if meets_target else mean_accuracy
    message = (
        f"mean_accuracy={mean_accuracy:.4f} "
        f"mean_time_seconds={mean_time_seconds:.4f} "
        f"meets_target={meets_target}"
    )

    return _build_result(
        score=score,
        valid=True,
        failure_type=None,
        message=message,
        runtime_seconds=elapsed,
        mean_accuracy=mean_accuracy,
        mean_time_seconds=mean_time_seconds,
        trials=trials,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        extra={
            "payload": payload,
            "target_accuracy": target_accuracy,
            "accuracy_margin": accuracy_margin,
            "meets_target": meets_target,
            "requested_gpu": payload.get("_modal", {}).get("requested_gpu"),
            "actual_device_name": actual_device_name,
            "gpu_mismatch_attempts": mismatch_attempts,
            "scores": {
                "accuracy_margin": accuracy_margin,
                "inverse_time": inverse_time,
                "meets_target": 1.0 if meets_target else 0.0,
            },
            "remote_runtime_seconds": remote_result.get("runtime_seconds"),
        },
    )


def evaluate_solver_code(
    solver_code: str,
    config: AirbenchEvalConfig,
    remote_runner: RemoteRunner,
) -> AirbenchEvalResult:
    started_at = time.time()

    if not solver_code or not solver_code.strip():
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="empty_solver",
            message="empty solver code",
            runtime_seconds=0.0,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail="",
            stderr_tail="",
        )

    try:
        compile(solver_code, "<candidate>", "exec")
    except SyntaxError as exc:
        return _build_result(
            score=0.0,
            valid=False,
            failure_type="syntax_error",
            message=f"syntax error: {exc}",
            runtime_seconds=0.0,
            mean_accuracy=None,
            mean_time_seconds=None,
            trials=None,
            stdout_tail="",
            stderr_tail="",
        )

    remote_result: dict[str, Any] | None = None
    last_remote_exception: Exception | None = None
    mismatch_attempts = 0
    max_attempts = max(1, config.gpu_mismatch_retries + 1)
    for attempt_idx in range(1, max_attempts + 1):
        try:
            remote_result = remote_runner.remote(
                solver_code=solver_code,
                script_args=build_script_args(config),
                timeout_seconds=config.timeout_seconds,
                print_subprocess_logs=config.stream_subprocess_logs,
            )
        except Exception as exc:
            last_remote_exception = exc
            remote_result = None
            break

        if remote_result.get("failure_type") != "gpu_mismatch":
            break

        mismatch_attempts = attempt_idx
        if attempt_idx >= max_attempts:
            break

    elapsed = time.time() - started_at
    return eval_result_from_remote_response(
        remote_result,
        config=config,
        elapsed=elapsed,
        last_remote_exception=last_remote_exception,
        mismatch_attempts=mismatch_attempts,
    )
