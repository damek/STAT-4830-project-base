"""
Persistent experiment log. Saved as JSON.

Each entry stores: timestamp, idea text, params changed, BPB achieved,
whether it improved the best, and the exact config delta.
"""
import json
import time
from pathlib import Path
from typing import Any


class History:
    def __init__(self, path: str = "alphagrad_experiments.json"):
        self.path = Path(path)
        self.entries: list[dict] = []
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self.entries = []

    def add(
        self,
        idea: dict,
        config: dict[str, Any],
        bpb: float,
        improved: bool,
    ) -> dict:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "change_type": idea.get("change_type", "hyperparameter"),
            "idea": idea.get("idea", ""),
            "params": idea.get("params", []),
            "reason": idea.get("reason", ""),
            "priority": idea.get("priority", "medium"),
            "bpb": bpb,
            "improved": improved,
            # Only record the values of params that were actually tuned
            "config_delta": {
                k: config[k]
                for k in idea.get("params", [])
                if k in config
            },
        }
        self.entries.append(entry)
        self._save()
        return entry

    def get_recent(self, n: int = 12) -> list[dict]:
        return self.entries[-n:]

    def best_bpb(self) -> float:
        if not self.entries:
            return float("inf")
        return min(e["bpb"] for e in self.entries)

    def n_improvements(self) -> int:
        return sum(1 for e in self.entries if e["improved"])

    def summary(self) -> str:
        n = len(self.entries)
        if n == 0:
            return "No experiments yet."
        return (
            f"{n} experiments | {self.n_improvements()} improved | "
            f"best proxy BPB = {self.best_bpb():.4f}"
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))
