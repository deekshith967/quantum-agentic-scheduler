"""Layer 13: ground truth at three levels and evaluation metrics (Chapter 10)."""
from __future__ import annotations

import math
import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "pred": len(pred), "gold": len(gold)}


def extraction_items(st) -> set:
    items = {("task", _norm(t.title)) for t in st.tasks}
    items |= {("avail", p.id, float(p.avail_h)) for p in st.people if p.avail_h is not None and "voice" in p.src}
    items |= {("rule", c.who, tuple(sorted(_norm(st.task(t).title) for t in c.tasks))) for c in st.constraints if c.type == "at_most_one"}
    return items


def gold_items(case: dict) -> set:
    g = {("task", _norm(t)) for t in case["gold"]["tasks"]}
    g |= {("avail", w, float(h)) for w, h in case["gold"].get("voice_availability", {}).items()}
    g |= {("rule", r["who"], tuple(sorted(_norm(x) for x in r["titles"]))) for r in case["gold"].get("rules", [])}
    return g


def optimality_gap(cost: float, optimum: float) -> float:
    return max(0.0, (cost - optimum) / optimum) if optimum else 0.0


def approximation_ratio(F: float, cmax: float, cmin: float) -> float:
    return (cmax - F) / (cmax - cmin) if cmax != cmin else 1.0


def mae(pairs: list[tuple[float, float]]) -> float:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else math.nan


def ground_truth_record(decision: dict, exact: dict, outcomes: list[dict] | None = None) -> dict:
    """Level 1 exact optimum, Level 2 PM label, Level 3 actual hours."""
    rec = {"level1_exact": {"assignment": exact.get("assignment"), "cost": exact.get("cost"), "status": exact.get("status")},
           "proposed": {"assignment": decision["assignment"], "cost": decision["cost"]["total"]},
           "gap": optimality_gap(decision["cost"]["total"], exact["cost"]) if exact.get("cost") else None,
           "level2_label": decision.get("review", {}).get("verdict"),
           "level3_outcomes": outcomes or []}
    if outcomes:
        rec["effort_mae_h"] = round(mae([(o["actual_h"], o["estimated_h"]) for o in outcomes]), 4)
    return rec
