"""Layer 14: promotion rule, bandit learning, effort-history updates and fine-tuning datasets (Algorithm 11.1).

Implemented: promotion rule, Q-table update, appending validated actual hours to the
effort history (which immediately changes future RAG estimates), and export of SFT /
DPO JSONL datasets. Not run here: the QLoRA/DPO training itself (scripts/finetune_slm.py
is provided as a template and needs a GPU plus a downloaded model).
"""
from __future__ import annotations

import csv
import datetime as dt
import json

from . import config, store
from .bandit import ContextualBandit, reward


def bandit_feedback(bandit: ContextualBandit, session: dict, verdict: str) -> dict | None:
    sel = session.get("bandit")
    if not sel or not sel.get("arm", "").startswith("qaoa"):
        return None
    q = session.get("qaoa") or {}
    dec = session["decision"]
    feasible = bool(session.get("qaoa_feasible"))
    gap = dec["solver"].get("gap", 1.0) if dec["solver"].get("used") == "qaoa" else 1.0
    latency = q.get("time_opt_s", 0.0) / max(q.get("starts", 1), 1)   # mean time of one optimisation run
    r = reward(feasible, gap, latency, verdict == "rejected")
    upd = bandit.update(sel["context"], sel["arm"], r)
    upd["latency_s"] = round(latency, 3)
    store.audit("feedback", "bandit:update", dec["decision_id"], **upd)
    return upd


def record_outcomes(session: dict, outcomes: list[dict], kb=None) -> list[dict]:
    """Promote actual hours of an APPROVED decision into the effort history."""
    dec = session["decision"]
    if dec["status"] != "approved":
        raise ValueError("outcomes can only be recorded for approved decisions")
    st = session["state"]
    tasks = {t["id"]: t for t in st["tasks"]}
    rows, out = [], []
    for o in outcomes:
        t = tasks[o["task_id"]]
        person = dec["assignment"][o["task_id"]]
        est = t["est_h"][person]
        rows.append([f"OUT-{dec['decision_id']}-{o['task_id']}", t["title"], t.get("skill") or "", person, float(o["actual_h"]), st.get("date") or dt.date.today().isoformat()])
        out.append({"task_id": o["task_id"], "person": person, "estimated_h": est, "actual_h": float(o["actual_h"])})
    path = config.RUNTIME / "effort_outcomes.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "title", "skill", "person", "actual_h", "date"])
        w.writerows(rows)
    if kb is not None:
        kb.reload_history()
    store.audit("feedback", "effort:append", dec["decision_id"], rows=len(rows))
    return out


def export_finetune(session: dict) -> dict:
    """SFT pairs from approved extractions; DPO pairs where the PM edited the plan or a constraint."""
    dec = session["decision"]
    verdict = dec.get("review", {}).get("verdict")
    if verdict is None:
        return {"sft": 0, "dpo": 0, "reason": "unreviewed output never becomes training data"}
    st = session["state"]
    target = {"tasks": [t["title"] for t in st["tasks"]], "people": [{"id": p["id"], "avail_h": p["avail_h"]} for p in st["people"]],
              "constraints": [c for c in st["constraints"] if c["type"] == "at_most_one"]}
    raw = {"brief": session["inputs"].get("brief_text"), "transcript": session["inputs"].get("transcript_text"),
           "ocr": [i["text"] for i in session.get("perception", []) if i["kind"] == "task"]}
    n_sft = n_dpo = 0
    if verdict in ("approved", "edited"):
        store.append_jsonl("finetune_sft", {"input": raw, "output": target})
        n_sft = 1
    if verdict == "edited" or dec.get("review", {}).get("edited"):
        rv = dec["review"]
        store.append_jsonl("finetune_dpo", {"prompt": raw, "chosen": rv.get("edited_to"), "rejected": rv.get("previous")})
        n_dpo = 1
    return {"sft": n_sft, "dpo": n_dpo}
