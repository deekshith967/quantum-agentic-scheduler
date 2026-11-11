"""Layer 8: contextual bandit (UCB) that selects the solver arm and QAOA depth.

Context bucket = (qubit band, coupling density, deadline). Each (bucket, arm) keeps its
own value Q and count n. Selection uses UCB (Equation 7.1); learning uses a constant
step size alpha (Equation 11.2). The table is persisted to data/runtime/bandit.json.

The starting table for the scenario's bucket is the ILLUSTRATIVE prior from Table 7.1
of the report (Q = 1.30, 1.20, 0.85 with n = 6, 2, 2). It is synthetic, not learned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import config

QUANTUM_ARMS = ["qaoa_p1", "qaoa_p2", "qaoa_p3"]
PRODUCTION_ARMS = QUANTUM_ARMS + ["cpsat"]
PRIOR = {"q<=4|dense|no-deadline": {"qaoa_p1": [1.30, 6], "qaoa_p2": [1.20, 2], "qaoa_p3": [0.85, 2]}}


def bucket(n_qubits: int, density: float, deadline: bool = False) -> str:
    q = "q<=4" if n_qubits <= 4 else ("q5-8" if n_qubits <= 8 else ("q9-12" if n_qubits <= 12 else "q>12"))
    return f"{q}|{'dense' if density >= 0.5 else 'sparse'}|{'deadline' if deadline else 'no-deadline'}"


def reward(feasible: bool, gap: float, latency_s: float, rejected: bool) -> float:
    """r = 1*[feasible] + (1 - gap) - 0.1*latency_s - 0.5*[rejected]   (Equation 11.1)"""
    return round((1.0 if feasible else 0.0) + (1.0 - gap) - 0.1 * latency_s - (0.5 if rejected else 0.0), 6)


class ContextualBandit:
    def __init__(self, path: Path | None = None, kappa: float = config.UCB_KAPPA, alpha: float = config.BANDIT_ALPHA):
        self.path = Path(path or config.RUNTIME / "bandit.json")
        self.kappa, self.alpha = kappa, alpha
        if self.path.exists():
            self.table = json.loads(self.path.read_text())
        else:
            self.table = {"_note": "Q[bucket][arm] = [value, count]. Initial values for q<=4|dense|no-deadline are the illustrative prior of Table 7.1.",
                          **json.loads(json.dumps(PRIOR))}
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.table, indent=1))

    def select(self, ctx: str, mode: str = "quantum_eval") -> dict:
        arms = QUANTUM_ARMS if mode == "quantum_eval" else PRODUCTION_ARMS
        row = self.table.setdefault(ctx, {})
        for a in arms:
            row.setdefault(a, [0.0, 0])
        untried = [a for a in arms if row[a][1] == 0]
        N = sum(row[a][1] for a in arms)
        scores = {}
        for a in arms:
            q, n = row[a]
            scores[a] = None if n == 0 else round(q + self.kappa * math.sqrt(math.log(max(N, 1)) / n), 4)
        arm = untried[0] if untried else max(arms, key=lambda a: scores[a])
        return {"context": ctx, "mode": mode, "arm": arm, "scores": scores, "N": N,
                "table": {a: {"Q": row[a][0], "n": row[a][1]} for a in arms}, "rule": "try-untried" if untried else "UCB"}

    def update(self, ctx: str, arm: str, r: float) -> dict:
        row = self.table.setdefault(ctx, {})
        q, n = row.get(arm, [0.0, 0])
        new_q = q + (r - q) if n == 0 else q + self.alpha * (r - q)
        row[arm] = [round(new_q, 6), n + 1]
        self.save()
        return {"context": ctx, "arm": arm, "reward": r, "Q_before": q, "Q_after": round(new_q, 6), "n": n + 1}
