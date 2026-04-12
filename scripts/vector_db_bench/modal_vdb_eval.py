#!/usr/bin/env python3
"""Modal remote build + vector-db-benchmark (CPU).

Use this to run evaluation in the cloud while keeping Codex (or another proposer) local.

Prerequisites:
  - `modal` CLI authenticated (`modal token set ...`)
  - Local clone of KCORES/vector-db-bench
  - Benchmark JSON inputs under that repo (e.g. data/query_vectors.json) or a separate data dir

Example:
  export VECTOR_DB_BENCH_ROOT=/path/to/vector-db-bench
  modal run scripts/vector_db_bench/modal_vdb_eval.py \\
    --candidate-json /path/to/candidate_files.json \\
    --proxy-max-queries 500

`candidate_files.json` is a JSON object: { "relative/path": "file contents", ... }
matching the harness editable surface (see CONTRACT.md).

Environment:
  VECTOR_DB_BENCH_ROOT  (required) absolute path to vector-db-bench repo; mounted read-only at /opt/vector-db-bench
  VECTOR_DB_BENCH_DATA  (optional) absolute path to dir containing base_vectors.json, query_vectors.json,
                        ground_truth.json — mounted at /opt/vdb-data. If unset, uses $VECTOR_DB_BENCH_ROOT/data
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "alpha-vector-db-bench-eval"
REMOTE_BENCH = "/opt/vector-db-bench"
REMOTE_DATA = "/opt/vdb-data"
SERVER_PORT = 8080

_bench_root = os.environ.get("VECTOR_DB_BENCH_ROOT", "").strip()
if not _bench_root:
    raise RuntimeError(
        "Set VECTOR_DB_BENCH_ROOT to an absolute path of your local vector-db-bench clone before `modal run`."
    )
_bench_path = Path(_bench_root).resolve()
if not _bench_path.is_dir():
    raise RuntimeError(f"VECTOR_DB_BENCH_ROOT is not a directory: {_bench_path}")

_data_root = os.environ.get("VECTOR_DB_BENCH_DATA", "").strip()
_data_path = Path(_data_root).resolve() if _data_root else _bench_path / "data"
if not _data_path.is_dir():
    raise RuntimeError(
        f"Data directory missing: {_data_path}. Set VECTOR_DB_BENCH_DATA or prepare { _bench_path / 'data' }."
    )

def _bench_mount_filter(local_path: str) -> bool:
    parts = Path(local_path).parts
    skip = {".git", "target", ".idea", ".vscode", "data"}
    return not any(part in skip for part in parts)


bench_mount = modal.Mount.from_local_dir(
    str(_bench_path),
    remote_path=REMOTE_BENCH,
    condition=_bench_mount_filter,
)

data_mount = modal.Mount.from_local_dir(
    str(_data_path),
    remote_path=REMOTE_DATA,
)

rust_image = (
    modal.Image.debian_bookworm_slim(python_version="3.11")
    .apt_install(
        "curl",
        "ca-certificates",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "clang",
        "git",
    )
    .run_commands(
        'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable',
        "bash -lc 'source /root/.cargo/env && rustc --version && cargo --version'",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "RUST_BACKTRACE": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("benchmark produced empty stdout")
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("benchmark stdout did not contain JSON")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("benchmark JSON root must be an object")
    return payload


@app.function(
    image=rust_image,
    mounts=[bench_mount, data_mount],
    timeout=60 * 45,
    cpu=4.0,
    memory=8192,
)
def remote_evaluate(
    candidate_files_json: str,
    *,
    concurrency: int = 4,
    warmup: int = 100,
    recall_threshold: float = 0.95,
    seed: int = 42,
    max_queries: int = 0,
    batch_size: int = 1000,
    base_vectors_name: str = "base_vectors.json",
    query_vectors_name: str = "query_vectors.json",
    ground_truth_name: str = "ground_truth.json",
) -> dict[str, Any]:
    """Build skeleton workspace with candidate files, run benchmark client, return parsed JSON stdout."""
    bench = Path(REMOTE_BENCH)
    skeleton = bench / "skeleton"
    benchmark_dir = bench / "benchmark"
    if not skeleton.is_dir() or not benchmark_dir.is_dir():
        return {"error": "missing skeleton or benchmark in mount", "bench": str(bench)}

    data_dir = Path(REMOTE_DATA)
    base_v = data_dir / base_vectors_name
    query_v = data_dir / query_vectors_name
    gt = data_dir / ground_truth_name
    for p in (base_v, query_v, gt):
        if not p.is_file():
            return {"error": f"missing data file in {REMOTE_DATA}: {p.name}"}

    try:
        files: dict[str, str] = json.loads(candidate_files_json)
    except json.JSONDecodeError as exc:
        return {"error": "invalid candidate_files_json", "detail": str(exc)}
    if not isinstance(files, dict) or not files:
        return {"error": "candidate_files_json must be a non-empty object"}

    workspace = Path(tempfile.mkdtemp(prefix="vdb_ws_"))
    server_proc: subprocess.Popen[str] | None = None
    try:
        shutil.copytree(skeleton, workspace, dirs_exist_ok=True)
        for rel, content in files.items():
            rel = rel.replace("\\", "/").lstrip("/")
            if ".." in rel.split("/"):
                return {"error": f"invalid path: {rel!r}"}
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

        br = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60 * 25,
        )
        if br.returncode != 0:
            return {
                "error": "skeleton_build_failed",
                "stderr": (br.stderr or br.stdout)[-8000:],
            }

        bb = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            timeout=60 * 25,
        )
        if bb.returncode != 0:
            return {
                "error": "benchmark_build_failed",
                "stderr": (bb.stderr or bb.stdout)[-8000:],
            }

        server_bin = workspace / "target" / "release" / "vector-db-skeleton"
        if not server_bin.is_file():
            return {"error": "server binary missing after build", "expected": str(server_bin)}

        bench_bin = benchmark_dir / "target" / "release" / "vector-db-benchmark"
        if not bench_bin.is_file():
            return {"error": "benchmark binary missing after build", "expected": str(bench_bin)}

        server_url = f"http://127.0.0.1:{SERVER_PORT}"
        log_out = open("/tmp/vdb_server.stdout", "w", encoding="utf-8")
        log_err = open("/tmp/vdb_server.stderr", "w", encoding="utf-8")
        server_proc = subprocess.Popen(
            [str(server_bin)],
            cwd=workspace,
            stdout=log_out,
            stderr=log_err,
            text=True,
        )
        deadline = time.time() + 60
        ok = False
        while time.time() < deadline:
            try:
                import socket

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", SERVER_PORT)) == 0:
                    ok = True
                    s.close()
                    break
                s.close()
            except OSError:
                pass
            time.sleep(0.2)
            if server_proc.poll() is not None:
                log_out.close()
                log_err.close()
                so = Path("/tmp/vdb_server.stdout").read_text(encoding="utf-8", errors="replace")
                se = Path("/tmp/vdb_server.stderr").read_text(encoding="utf-8", errors="replace")
                return {"error": "server_exited_early", "stdout": so[-4000:], "stderr": se[-4000:]}

        if not ok:
            return {"error": "server_start_timeout"}

        mq_arg: list[str] = []
        if max_queries > 0:
            mq_arg = ["--max-queries", str(max_queries)]

        be = subprocess.run(
            [
                str(bench_bin),
                "--server-url",
                server_url,
                "--concurrency",
                str(concurrency),
                "--warmup",
                str(warmup),
                "--base-vectors",
                str(base_v),
                "--query-vectors",
                str(query_v),
                "--ground-truth",
                str(gt),
                "--recall-threshold",
                str(recall_threshold),
                "--seed",
                str(seed),
                "--batch-size",
                str(batch_size),
                *mq_arg,
            ],
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
        out = be.stdout or ""
        err = be.stderr or ""
        if be.returncode != 0:
            return {
                "error": "benchmark_failed",
                "returncode": be.returncode,
                "stderr": err[-8000:],
                "stdout_tail": out[-4000:],
            }
        try:
            payload = _extract_json_payload(out)
        except Exception as exc:
            return {
                "error": "benchmark_json_parse_failed",
                "detail": str(exc),
                "stdout_tail": out[-8000:],
            }
        return {"ok": True, "payload": payload}
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        shutil.rmtree(workspace, ignore_errors=True)


@app.local_entrypoint()
def main(
    candidate_json: str,
    concurrency: int = 4,
    warmup: int = 100,
    max_queries: int = 0,
) -> None:
    """`modal run ...::main --candidate-json /path/to.json`"""
    path = Path(candidate_json)
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    data = path.read_text(encoding="utf-8")
    result = remote_evaluate.remote(
        data,
        concurrency=concurrency,
        warmup=warmup,
        max_queries=max_queries,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "Run with: modal run scripts/vector_db_bench/modal_vdb_eval.py::main --candidate-json ...\n"
        "Set VECTOR_DB_BENCH_ROOT and optionally VECTOR_DB_BENCH_DATA in the environment first."
    )
