#!/usr/bin/env python3
"""
Run Parameter Golf training on a remote GPU machine over SSH.

train_gpt_exploration.py (and train_gpt.py) always execute PyTorch on the host
that runs Python. This script does not move CUDA to your laptop; it starts the
same command on a cloud VM (Prime Intellect pod, RunPod, etc.) where you already
have CUDA.

One-time on the remote host
---------------------------
  - Clone this repo (or sync it).
  - Python venv + pip install -r requirements.txt
  - Download data: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
  - SSH key: your **public** key must be on the server (Prime: upload in dashboard).
    Optional file ``GPU_key.txt`` next to ``remote_train.py`` (gitignored): use a
    **private** SSH key (``-----BEGIN ... PRIVATE KEY-----``) for ``ssh -i``, *or*
    a single-line ``pit_...`` Prime API token (API use only; not valid for ``ssh -i``).

Typical usage (from your laptop)
--------------------------------
  Option A — env vars:
  set REMOTE_TRAIN_HOST=ubuntu@203.0.113.50
  set REMOTE_TRAIN_DIR=/home/ubuntu/STAT-4830-AlphaGrad-project/parameter_golf

  Option B — copy ``remote_target.example.txt`` to ``remote_target.txt`` (gitignored);
  put ``user@host`` on line 1 and remote ``parameter_golf`` path on line 2.

  python remote_train.py --export DATA_PATH --export TOKENIZER_PATH -- \\
    torchrun --standalone --nproc_per_node=1 train_gpt_exploration.py

Forward local env vars with repeated --export KEY (only if set locally).

Prime Intellect: after the pod is ACTIVE, copy the SSH command from the UI;
the part before @ is the user, after @ is the host — pass as -H user@host.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Default credentials file (gitignored): SSH private key *or* a single-line Prime API token.
_GPU_KEY_FILE = Path(__file__).resolve().parent / "GPU_key.txt"
# Optional host + remote dir (gitignored): see remote_target.example.txt
_REMOTE_TARGET_FILE = Path(__file__).resolve().parent / "remote_target.txt"


def _load_remote_target_file() -> tuple[str | None, str | None]:
    """Return (host, remote_dir) from remote_target.txt if present."""
    if not _REMOTE_TARGET_FILE.is_file():
        return None, None
    lines: list[str] = []
    with _REMOTE_TARGET_FILE.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    host = lines[0] if len(lines) >= 1 and "@" in lines[0] else None
    remote_dir = lines[1] if len(lines) >= 2 else None
    return host, remote_dir


def _first_nonblank_line(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    return ""


def _resolve_ssh_identity(cli_identity: str | None) -> str | None:
    """
    Prefer CLI / SSH_IDENTITY. Else read ``GPU_key.txt``:
    - PEM / OpenSSH private key -> use as ``ssh -i`` path.
    - ``ak-...`` Modal token -> set MODAL_TOKEN_ID; use modal_exploration.py instead of SSH.
    - ``pit_...`` Prime API token -> set PRIME_INTELLECT_API_KEY; SSH needs a real key elsewhere.
    """
    if cli_identity:
        return cli_identity
    env_i = os.environ.get("SSH_IDENTITY")
    if env_i:
        return env_i
    if not _GPU_KEY_FILE.is_file():
        return None
    first = _first_nonblank_line(_GPU_KEY_FILE)
    if first.startswith("-----BEGIN"):
        return str(_GPU_KEY_FILE)
    if first.startswith("pit_"):
        os.environ.setdefault("PRIME_INTELLECT_API_KEY", first)
        print(
            "GPU_key.txt: Prime API token loaded (PRIME_INTELLECT_API_KEY). "
            "SSH still needs a private key — put an OpenSSH/PEM key in GPU_key.txt, "
            "or set SSH_IDENTITY / pass -i.",
            file=sys.stderr,
        )
        return None
    # Unknown format: try as key file (ssh will fail if invalid)
    return str(_GPU_KEY_FILE)


def _build_remote_shell(
    remote_dir: str,
    exports: list[tuple[str, str]],
    remote_argv: list[str],
) -> str:
    parts: list[str] = [f"cd {shlex.quote(remote_dir)}"]
    for k, v in exports:
        parts.append(f"export {k}={shlex.quote(v)}")
    parts.append(shlex.join(remote_argv))
    return " && ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("---", 1)[0].strip())
    p.add_argument(
        "-H",
        "--host",
        default=os.environ.get("REMOTE_TRAIN_HOST"),
        help="SSH destination, e.g. ubuntu@1.2.3.4 (or set REMOTE_TRAIN_HOST)",
    )
    p.add_argument(
        "-d",
        "--remote-dir",
        default=os.environ.get("REMOTE_TRAIN_DIR"),
        help="Directory on the remote machine containing train_gpt_exploration.py "
        "(or set REMOTE_TRAIN_DIR)",
    )
    p.add_argument(
        "-i",
        "--identity",
        default=None,
        help="Path to SSH private key (overrides SSH_IDENTITY / GPU_key.txt)",
    )
    p.add_argument(
        "--export",
        action="append",
        default=[],
        metavar="ENV_VAR",
        help="Copy this environment variable from the local machine to the remote shell (if set)",
    )
    p.add_argument(
        "remote_argv",
        nargs=argparse.REMAINDER,
        help="Remote command, e.g. torchrun ... train_gpt_exploration.py (use -- before it if needed)",
    )
    args = p.parse_args()

    file_host, file_dir = _load_remote_target_file()
    host = args.host or file_host
    remote_dir = args.remote_dir or file_dir

    identity = _resolve_ssh_identity(args.identity)

    if not host:
        p.error(
            "Missing SSH target. Pass -H user@host, set REMOTE_TRAIN_HOST, or create "
            "remote_target.txt (see remote_target.example.txt in this folder)."
        )
    if not remote_dir:
        p.error(
            "Missing remote working directory. Pass -d /path/to/parameter_golf, set "
            "REMOTE_TRAIN_DIR, or add line 2 to remote_target.txt."
        )

    remote_argv = args.remote_argv
    if remote_argv and remote_argv[0] == "--":
        remote_argv = remote_argv[1:]
    if not remote_argv:
        p.error(
            "Provide the remote command, e.g.\n"
            "  python remote_train.py -H user@host -d ~/pg -- "
            "torchrun --standalone --nproc_per_node=1 train_gpt_exploration.py"
        )

    exports: list[tuple[str, str]] = []
    for name in args.export:
        if name in os.environ and os.environ[name] != "":
            exports.append((name, os.environ[name]))

    remote_shell = _build_remote_shell(remote_dir, exports, remote_argv)
    ssh_cmd = ["ssh", "-t"]
    if identity:
        ssh_cmd.extend(["-i", identity])
    ssh_cmd.append(host)
    ssh_cmd.append(remote_shell)

    print("Running on remote:", remote_shell[:200] + ("..." if len(remote_shell) > 200 else ""))
    return subprocess.call(ssh_cmd)


if __name__ == "__main__":
    sys.exit(main() or 0)
