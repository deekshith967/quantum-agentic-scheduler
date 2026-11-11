"""Layer 12: human-in-the-loop review state machine, approval tokens and guarded writes (Figure 9.1).

States: draft -> pending | flagged -> approved | rejected; an edit is re-verified by
Layer 11 before it can be approved. Only `approved` issues a single-use approval token
that the tool gateway accepts for the write tools.
"""
from __future__ import annotations

import hashlib
import secrets

from . import store
from .schemas import Decision, ProjectState, ToolCall
from .solvers.classical import schedule_slots, violations


def _h(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


class ApprovalRegistry:
    def __init__(self):
        self.data = store.load_json("hitl", "approvals") or {}

    def issue(self, decision_id: str) -> str:
        tok = secrets.token_urlsafe(16)
        self.data[decision_id] = {"hash": _h(tok), "used": False}
        store.save_json("hitl", "approvals", self.data)
        return tok

    def check(self, decision_id: str, token: str) -> bool:
        rec = self.data.get(decision_id)
        return bool(rec and token and rec["hash"] == _h(token))


class ReviewService:
    def __init__(self, gateway, approvals: ApprovalRegistry):
        self.gw, self.approvals = gateway, approvals

    def submit(self, dec: Decision) -> Decision:
        dec.status = "flagged" if dec.hitl.get("level") == "flagged" else "pending"
        store.audit("system", "hitl:submit", dec.decision_id, level=dec.status, reasons=dec.hitl.get("reasons", []))
        return dec

    def approve(self, dec: Decision, st: ProjectState, acknowledged: list[str] | None = None, reviewer: str = "PM") -> tuple[Decision, dict]:
        if dec.status not in ("pending", "flagged"):
            raise ValueError(f"cannot approve a decision in state {dec.status}")
        if dec.violations:
            raise ValueError("decision has violations; edit it first")
        if dec.status == "flagged" and len(acknowledged or []) < len(dec.hitl.get("reasons", [])):
            raise ValueError("flagged review: acknowledge every flag before approving")
        token = self.approvals.issue(dec.decision_id)
        dec.status = "approved"
        dec.review = {"verdict": "approved", "reviewer": reviewer, "acknowledged": acknowledged or []}
        names = {t.id: t.title for t in st.tasks}
        tickets = [{"task_id": t, "title": names.get(t, t), "assignee": r, "sprint": st.sprint_id} for t, r in dec.assignment.items()]
        events = [{"task_id": t, "person": r, "date": st.date, "start": dec.timetable.get(t, ["", ""])[0], "end": dec.timetable.get(t, ["", ""])[1]}
                  for t, r in dec.assignment.items()]
        r1 = self.gw.execute(ToolCall(tool="create_tickets", args={"decision_id": dec.decision_id, "approval_token": token, "tickets": tickets}), actor=reviewer)
        r2 = self.gw.execute(ToolCall(tool="create_events", args={"decision_id": dec.decision_id, "approval_token": token, "events": events}), actor=reviewer)
        store.audit(reviewer, "hitl:approve", dec.decision_id, acknowledged=acknowledged or [])
        return dec, {"tickets": r1.data.get("created", []), "events": r2.data.get("created", []), "ok": r1.ok and r2.ok}

    def edit(self, dec: Decision, st: ProjectState, new_assignment: dict[str, str], reviewer: str = "PM") -> Decision:
        from .decision import summarise
        v = violations(st, new_assignment)
        dec.review = {"verdict": "edited", "reviewer": reviewer, "previous": dict(dec.assignment), "edited_to": new_assignment}
        dec.assignment = new_assignment
        dec.violations = v
        s = summarise(st, new_assignment)
        dec.cost, dec.load_h = s["cost"], s["load_h"]
        if v:
            dec.status = "flagged"
            dec.hitl = {**dec.hitl, "level": "blocked", "reasons": [f"edit blocked: {x}" for x in v]}
        else:
            dec.timetable = schedule_slots(st, new_assignment)["timetable"]
            dec.status = "pending"
            dec.solver = {**dec.solver, "used": "human_edit"}
        store.audit(reviewer, "hitl:edit", dec.decision_id, violations=v)
        return dec

    def reject(self, dec: Decision, comment: str, reviewer: str = "PM") -> Decision:
        dec.status = "rejected"
        dec.review = {"verdict": "rejected", "reviewer": reviewer, "comment": comment}
        store.audit(reviewer, "hitl:reject", dec.decision_id, comment=comment)
        return dec
