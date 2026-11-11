"""Layer 7: bounded ReAct loop with multi-hop evidence gathering (Algorithm 6.1)."""
from __future__ import annotations

import json

from . import config
from .schemas import HopRecord, ProjectState, Provenance, ToolResult
from .state import compute_missing
from .tools import ToolGateway


def _skill_for(title: str) -> str | None:
    t = title.lower()
    best, hits = None, 0
    for skill, kws in config.SKILL_KEYWORDS.items():
        n = sum(1 for k in kws if k in t)
        if n > hits:
            best, hits = skill, n
    return best


def apply_observation(st: ProjectState, tool: str, res: ToolResult) -> str:
    """Update the state from a tool result and return a short description of the effect."""
    d = res.data
    if tool == "skill_lookup":
        for pid, info in d["people"].items():
            p = st.person(pid)
            p.role, p.skills = info["role"], info["skills"]
        for t in st.tasks:
            t.skill = t.skill or _skill_for(t.title)
        for pid in d.get("unknown", []):
            st.flags.append(f"unknown person {pid}")
        return "skills filled; tasks tagged " + ", ".join(f"{t.id}:{t.skill}" for t in st.tasks)
    if tool == "calendar_availability":
        eff = []
        for pid, h in d["hours"].items():
            p = st.person(pid)
            if h is None:
                st.flags.append(f"no calendar entry for {pid}")
                continue
            if p.avail_h is None:
                p.avail_h, p.conf = float(h), 0.99
                p.src = "calendar"
            elif abs(p.avail_h - h) < 1e-9:
                old = p.conf
                p.conf = max(p.conf, 0.95)
                p.src += "+calendar"
                eff.append(f"{pid} {h} h confirmed ({old} → {p.conf})")
                continue
            else:
                st.flags.append(f"conflict: {pid} calendar {h} h vs {p.src} {p.avail_h} h; kept {min(h, p.avail_h)} h")
                p.avail_h = min(h, p.avail_h)
                p.src += "+calendar(conflict)"
            eff.append(f"{pid} {p.avail_h} h")
            st.provenance.append(Provenance(field=f"avail:{pid}", source="calendar_availability", detail=str(h)))
        return "; ".join(eff)
    if tool == "effort_history":
        n = 0
        for e in d["estimates"]:
            t = st.task(e["task_id"])
            if e["hours"] is None:
                st.flags.append(f"no effort history for {e['task_id']} / {e['person_id']}")
                t.est_h[e["person_id"]] = float("nan")
            else:
                t.est_h[e["person_id"]] = e["hours"]
                t.evidence[e["person_id"]] = e["evidence"]
                n += 1
        return f"{n} costs filled"
    if tool == "dependency_check":
        st.dependencies = d["scheduling_precedence"]
        st.provenance.append(Provenance(field="dependencies", source="dependency_check", detail=json.dumps(d)))
        return (f"assignment precedence: {d['assignment_precedence'] or 'none'}; "
                f"scheduling precedence: {d['scheduling_precedence'] or 'none'}")
    if tool == "policy_search":
        top = d["passages"][0] if d["passages"] else None
        ok = bool(top and any(w in top["text"].lower() for w in ["cap", "at most one"]))
        for c in st.constraints:
            if c.type == "at_most_one" and c.policy_ok is None:
                c.policy_ok = ok
                if not ok:
                    st.flags.append(f"constraint for {c.who} not supported by policy")
        return f"top passage '{top['section'] if top else None}' → constraint {'allowed' if ok else 'not confirmed'} (confidence unchanged)"
    return "no effect"


def run_react(st: ProjectState, planner, gateway: ToolGateway, max_hops: int = config.MAX_HOPS):
    trace: list[HopRecord] = []
    seen: set[str] = set()
    hops = 0
    status = "COMPLETE"
    while st.missing and hops < max_hops:
        call = planner.next_action(st)
        if call is None:
            break
        key = call.tool + json.dumps(call.args, sort_keys=True, default=str)
        if key in seen:
            status = "ESCALATE"
            st.flags.append(f"repeated call to {call.tool}")
            break
        before = st.model_dump_json()
        res = gateway.execute(call)
        hops += 1
        seen.add(key)
        if not res.ok:
            st.flags.append(f"tool error {call.tool}: {res.error}")
            trace.append(HopRecord(hop=hops, intent=call.intent, tool=call.tool, args=call.args, observation=f"ERROR {res.error}", effect="none"))
            status = "ESCALATE"
            break
        effect = apply_observation(st, call.tool, res)
        st.missing = compute_missing(st)
        obs = json.dumps(res.data, default=str)
        trace.append(HopRecord(hop=hops, intent=call.intent, tool=call.tool, args=call.args,
                               observation=obs[:400] + ("…" if len(obs) > 400 else ""), effect=effect))
        if st.model_dump_json() == before:
            status = "ESCALATE"
            st.flags.append(f"no new information from {call.tool}")
            break
    if st.missing:
        status = "ESCALATE"
    trace.append(HopRecord(hop=hops + 1, intent="Is anything still missing?", tool="stop check", args={},
                           observation=f"missing = {st.missing}", effect=f"exit loop with {status}"))
    return st, trace, status
