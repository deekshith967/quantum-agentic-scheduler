"""Layer 3: merge perception output into a validated ProjectState (Algorithm 5.1)."""
from __future__ import annotations

import datetime as dt
import json

from . import config
from .schemas import Constraint, Extracted, Objective, Person, ProjectState, Provenance, Task


def resolve_day(day_name: str | None) -> str | None:
    """Map a weekday name to the first matching date in the (mock) calendar."""
    if not day_name:
        return None
    cal = json.loads((config.ORG_DIR / "calendar.json").read_text())
    for d in sorted(cal["availability"]):
        if dt.date.fromisoformat(d).strftime("%A").lower() == day_name.lower():
            return d
    return None


def build_state(session_id: str, items: list[Extracted]) -> ProjectState:
    st = ProjectState(session_id=session_id)
    k = 0
    for e in items:
        if e.kind == "task":
            t = Task(id="", title=e.value["title"], column=e.value.get("column", "To Do"), src=e.src, conf=e.conf)
            if t.column == "Done":
                t.id = f"X{len(st.excluded_tasks) + 1}"
                st.excluded_tasks.append(t)
            else:
                k += 1
                t.id = f"T{k}"
                st.tasks.append(t)
            st.provenance.append(Provenance(field=f"task:{t.id}", source=e.src, detail=f"column={t.column}"))
    for e in items:
        if e.kind == "person":
            for pid in e.value["ids"]:
                if not st.person(pid):
                    st.people.append(Person(id=pid, src=e.src, conf=e.conf))
        elif e.kind == "date":
            st.day_name = e.value["day_name"]
            st.date = resolve_day(st.day_name)
        elif e.kind == "objective":
            st.objective = Objective(minimise=e.value["minimise"], mu=e.value["mu"])
    for e in items:
        if e.kind == "availability":
            p = st.person(e.value["who"])
            if p is None:
                p = Person(id=e.value["who"], src=e.src, conf=e.conf)
                st.people.append(p)
            if p.avail_h is None or e.value["hours"] < p.avail_h:  # keep the more restrictive value
                p.avail_h, p.src, p.conf = e.value["hours"], e.src, e.conf
            st.provenance.append(Provenance(field=f"avail:{p.id}", source=e.src, detail=e.text))
        elif e.kind == "constraint":
            ids = [t.id for title in e.value["titles"] for t in st.tasks if t.title == title]
            if len(ids) >= 2:
                st.constraints.append(Constraint(type=e.value["type"], who=e.value["who"], tasks=ids, src=e.src, conf=e.conf))
                st.provenance.append(Provenance(field="constraint", source=e.src, detail=e.text))
    st.constraints.append(Constraint(type="capacity", who="*", src="policy:working-hours", conf=1.0))
    st.missing = compute_missing(st)
    return st


def compute_missing(st: ProjectState) -> list[str]:
    miss: list[str] = []
    if any(p.role is None for p in st.people):
        miss.append("skills")
    if any("calendar" not in p.src for p in st.people):
        miss.append("availability")
    if any(len(t.est_h) < len(st.people) for t in st.tasks):
        miss.append("durations")
    if not any(pr.field == "dependencies" for pr in st.provenance):
        miss.append("dependencies")
    if any(c.type == "at_most_one" and c.policy_ok is None for c in st.constraints):
        miss.append("policy")
    return miss
