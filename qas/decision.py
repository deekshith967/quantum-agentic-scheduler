"""Layer 11: decode QAOA samples, verify feasibility independently, apply the acceptance rule (Algorithm 9.1)."""
from __future__ import annotations

from . import config
from .qubo import QuboModel
from .schemas import ProjectState
from .solvers.classical import true_cost, violations


def candidates_from_counts(st: ProjectState, model: QuboModel, counts: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Returns (feasible candidates sorted by count, rejected samples with reasons)."""
    feas, rej = [], []
    for bits, n in counts.items():
        x = [int(b) for b in bits]
        a = model.decode(x)
        v = violations(st, a) if model.base_load == {} or not any(model.base_load.values()) else _chunk_violations(st, model, a)
        row = {"bits": bits, "count": n, "assignment": a, "qubo_cost": round(model.cost(x), 6)}
        if v:
            rej.append({**row, "violations": v})
        else:
            feas.append({**row, "cost": round(model.cost(x), 6)})
    feas.sort(key=lambda r: (-r["count"], r["cost"]))
    rej.sort(key=lambda r: -r["count"])
    return feas, rej


def _chunk_violations(st: ProjectState, model: QuboModel, a: dict) -> list[str]:
    """Feasibility of a decomposed sub-problem: capacity is reduced by the load already placed."""
    sub = st.model_copy(deep=True)
    sub.tasks = [t for t in st.tasks if t.id in model.tasks]
    for p in sub.people:
        if p.avail_h is not None:
            p.avail_h = p.avail_h - model.base_load.get(p.id, 0.0)
    return violations(sub, a)


def accept(candidate_cost: float, baseline_cost: float, tau: float = config.ACCEPT_TAU) -> dict:
    thr = (1 + tau) * baseline_cost
    gap = (candidate_cost - baseline_cost) / baseline_cost if baseline_cost else 0.0
    return {"accept": candidate_cost <= thr + 1e-9, "threshold": round(thr, 6), "gap": round(max(gap, 0.0), 6), "tau": tau}


def hitl_triggers(st: ProjectState, assignment: dict[str, str], fallback_reason: str | None) -> dict:
    reasons, level = [], "standard"
    confs = st.decision_confidences()
    lo = min(confs.values()) if confs else 1.0
    if lo < config.CONF_THRESHOLD:
        worst = min(confs, key=confs.get)
        reasons.append(f"{worst} has confidence {lo:.2f} (< {config.CONF_THRESHOLD})")
    if fallback_reason:
        reasons.append(f"classical fallback used: {fallback_reason}")
    for f in st.flags:
        if f.startswith(("conflict", "no effort history", "no calendar", "unknown person", "constraint for")):
            reasons.append(f)
    for t, r in assignment.items():
        p = st.person(r)
        if p and p.avail_h == 0:
            reasons.append(f"{t} assigned to {r} who has zero availability")
    if reasons:
        level = "flagged"
    return {"required": True, "level": level, "reasons": reasons, "min_confidence": round(lo, 3)}


def summarise(st: ProjectState, assignment: dict[str, str]) -> dict:
    c = true_cost(st, assignment)
    return {"cost": {"total": round(c["total"], 4), "effort_h": round(c["effort_h"], 4), "imbalance": round(c["imbalance"], 4), "penalty": 0.0},
            "load_h": {k: round(v, 4) for k, v in c["loads"].items()},
            "capacity_h": {p.id: p.avail_h for p in st.people}}
