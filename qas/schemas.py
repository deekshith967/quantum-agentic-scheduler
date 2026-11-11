"""Layer 3 data contract and the other typed objects shared between layers.

Every field that comes from an input or a tool carries `src` (provenance) and
`conf` (confidence in [0, 1]) so that later layers can decide how far to trust it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Task(BaseModel):
    id: str
    title: str
    skill: Optional[str] = None
    column: str = "To Do"
    est_h: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    src: str = ""
    conf: float = 1.0


class Person(BaseModel):
    id: str
    role: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    avail_h: Optional[float] = None
    src: str = ""
    conf: float = 1.0


class Constraint(BaseModel):
    type: Literal["at_most_one", "capacity", "precedence"]
    who: str = "*"
    tasks: list[str] = Field(default_factory=list)
    src: str = ""
    conf: float = 1.0
    policy_ok: Optional[bool] = None


class Objective(BaseModel):
    minimise: list[str] = Field(default_factory=lambda: ["effort", "imbalance"])
    mu: float = 0.1


class Provenance(BaseModel):
    field: str
    source: str
    detail: str = ""
    ts: str = Field(default_factory=now_iso)


class ProjectState(BaseModel):
    sprint_id: str = "SPR-14"
    session_id: str
    date: Optional[str] = None
    day_name: Optional[str] = None
    tasks: list[Task] = Field(default_factory=list)
    excluded_tasks: list[Task] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    dependencies: list[list[str]] = Field(default_factory=list)
    objective: Objective = Field(default_factory=Objective)
    missing: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)

    def person(self, pid: str) -> Optional[Person]:
        return next((p for p in self.people if p.id == pid), None)

    def task(self, tid: str) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == tid), None)

    def decision_confidences(self) -> dict[str, float]:
        out = {f"task:{t.id}": t.conf for t in self.tasks}
        out.update({f"avail:{p.id}": p.conf for p in self.people if p.avail_h is not None})
        out.update({f"constraint:{c.type}:{c.who}:{'+'.join(c.tasks)}": c.conf for c in self.constraints})
        return out


class Extracted(BaseModel):
    """One item produced by Layer 2 perception."""
    kind: Literal["task", "person", "availability", "constraint", "objective", "date"]
    value: dict[str, Any]
    src: str
    conf: float
    text: str = ""


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    intent: str = ""


class ToolResult(BaseModel):
    call_id: str
    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0
    audit_id: str = ""


class HopRecord(BaseModel):
    hop: int
    intent: str
    tool: str
    args: dict[str, Any]
    observation: str
    effect: str


class Decision(BaseModel):
    decision_id: str
    session_id: str
    status: Literal["draft", "pending", "flagged", "approved", "rejected"] = "draft"
    assignment: dict[str, str]
    timetable: dict[str, list[str]] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)
    load_h: dict[str, float] = Field(default_factory=dict)
    capacity_h: dict[str, float] = Field(default_factory=dict)
    solver: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    min_confidence: float = 1.0
    hitl: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""
    review: dict[str, Any] = Field(default_factory=dict)
    created: str = Field(default_factory=now_iso)


class AuditEvent(BaseModel):
    ts: str = Field(default_factory=now_iso)
    actor: str
    action: str
    ref: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
