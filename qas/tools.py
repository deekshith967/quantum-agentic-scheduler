"""Layer 6: typed tools behind one gateway.

Read tools run freely. Write tools (`create_tickets`, `create_events`) are registered
but refuse to run without an approval token issued by Layer 12 for that decision.
Every call is written to the audit log. External systems (calendar, Jira) are mocks
backed by JSON files with production-like request/response shapes.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from . import config, store
from .rag import KnowledgeBase
from .schemas import ToolCall, ToolResult


@dataclass
class Tool:
    name: str
    fn: Callable[..., dict]
    risk: str               # "read" | "compute" | "write"
    description: str
    params: dict[str, str]


class ApprovalError(PermissionError):
    pass


class ToolGateway:
    def __init__(self, kb: KnowledgeBase | None = None, approvals: Callable[[str, str], bool] | None = None):
        self.kb = kb or KnowledgeBase()
        self._approval_check = approvals or (lambda decision_id, token: False)
        self.tools: dict[str, Tool] = {}
        self._register_defaults()

    # ------------------------------------------------------------------ registry
    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [{"name": t.name, "risk": t.risk, "description": t.description, "params": t.params} for t in self.tools.values()]

    def execute(self, call: ToolCall, actor: str = "agent") -> ToolResult:
        cid = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()
        tool = self.tools.get(call.tool)
        if tool is None:
            res = ToolResult(call_id=cid, tool=call.tool, ok=False, error="unknown tool")
        else:
            try:
                if tool.risk == "write":
                    did, tok = call.args.get("decision_id", ""), call.args.get("approval_token", "")
                    if not self._approval_check(did, tok):
                        raise ApprovalError(f"{call.tool} requires a valid approval token for decision {did or '?'}")
                data = tool.fn(**call.args)
                res = ToolResult(call_id=cid, tool=call.tool, ok=True, data=data)
            except ApprovalError as exc:
                res = ToolResult(call_id=cid, tool=call.tool, ok=False, error=str(exc))
            except Exception as exc:  # tool errors are returned, never raised to the agent
                res = ToolResult(call_id=cid, tool=call.tool, ok=False, error=f"{type(exc).__name__}: {exc}")
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        args = {k: v for k, v in call.args.items() if k != "approval_token"}
        ev = store.audit(actor, f"tool:{call.tool}", cid, ok=res.ok, args=args, error=res.error)
        res.audit_id = ev.ts + "/" + cid
        return res

    # ------------------------------------------------------------------ tools
    def _register_defaults(self) -> None:
        org = config.ORG_DIR
        people = {p["id"]: p for p in json.loads((org / "people.json").read_text())["people"]}
        cal = json.loads((org / "calendar.json").read_text())
        deps = json.loads((org / "dependencies.json").read_text())

        def skill_lookup(person_ids: list[str]) -> dict:
            return {"people": {pid: {"role": people[pid]["role"], "skills": people[pid]["skills"]} for pid in person_ids if pid in people},
                    "unknown": [p for p in person_ids if p not in people]}

        def calendar_availability(person_ids: list[str], date: str) -> dict:
            day = cal["availability"].get(date)
            if day is None:
                raise ValueError(f"no calendar data for {date}")
            return {"date": date, "start": cal["workday"]["start"], "hours": {p: day.get(p) for p in person_ids}}

        def effort_history(pairs: list[dict]) -> dict:
            out = []
            for pr in pairs:
                e = self.kb.estimate_effort(pr["title"], pr.get("skill"), pr["person_id"])
                out.append({"task_id": pr["task_id"], "person_id": pr["person_id"], **e})
            return {"estimates": out}

        def dependency_check(task_titles: dict[str, str]) -> dict:
            inv = {v.lower(): k for k, v in task_titles.items()}
            sched = [[inv[a], inv[b]] for a, b in deps["scheduling_precedence"] if a in inv and b in inv]
            assign = [[inv[a], inv[b]] for a, b in deps["assignment_precedence"] if a in inv and b in inv]
            return {"assignment_precedence": assign, "scheduling_precedence": sched}

        def policy_search(query: str, k: int = 3) -> dict:
            return {"passages": self.kb.search_policy(query, k)}

        def create_tickets(decision_id: str, approval_token: str, tickets: list[dict]) -> dict:
            ids = []
            for t in tickets:
                tid = f"REL-{uuid.uuid4().hex[:4].upper()}"
                store.append_jsonl("mock_jira", {"id": tid, "decision_id": decision_id, **t})
                ids.append(tid)
            return {"created": ids}

        def create_events(decision_id: str, approval_token: str, events: list[dict]) -> dict:
            ids = []
            for e in events:
                eid = f"EV-{uuid.uuid4().hex[:6]}"
                store.append_jsonl("mock_calendar_events", {"id": eid, "decision_id": decision_id, **e})
                ids.append(eid)
            return {"created": ids}

        R = self.register
        R(Tool("skill_lookup", skill_lookup, "read", "Roles and skills for people.", {"person_ids": "list[str]"}))
        R(Tool("calendar_availability", calendar_availability, "read", "Available hours on a date.", {"person_ids": "list[str]", "date": "YYYY-MM-DD"}))
        R(Tool("effort_history", effort_history, "read", "Estimate hours per (task, person) from similar past tasks (RAG).",
               {"pairs": "list[{task_id,title,skill,person_id}]"}))
        R(Tool("dependency_check", dependency_check, "read", "Precedence between tasks.", {"task_titles": "dict[task_id,title]"}))
        R(Tool("policy_search", policy_search, "read", "Search approved policy documents (hybrid BM25 + dense, RRF).", {"query": "str", "k": "int"}))
        R(Tool("create_tickets", create_tickets, "write", "Create tickets in the issue tracker (mock). Needs approval token.",
               {"decision_id": "str", "approval_token": "str", "tickets": "list"}))
        R(Tool("create_events", create_events, "write", "Create calendar events (mock). Needs approval token.",
               {"decision_id": "str", "approval_token": "str", "events": "list"}))
