"""Layer 10, classical: exact brute force, CP-SAT assignment and CP-SAT time-slot scheduling."""
from __future__ import annotations

import datetime as dt
import itertools
import json
import time

from ortools.sat.python import cp_model

from .. import config
from ..schemas import ProjectState


def true_cost(st: ProjectState, assignment: dict[str, str]) -> dict:
    """Effort + mu * sum_{r<s} (L_r - L_s)^2, evaluated directly (no QUBO, no penalties)."""
    people = [p.id for p in st.people]
    loads = {r: 0.0 for r in people}
    for t, r in assignment.items():
        loads[r] += st.task(t).est_h[r]
    effort = sum(loads.values())
    imb = sum((loads[a] - loads[b]) ** 2 for a, b in itertools.combinations(people, 2))
    return {"total": round(effort + st.objective.mu * imb, 6), "effort_h": round(effort, 6), "imbalance": round(st.objective.mu * imb, 6), "loads": loads}


def violations(st: ProjectState, assignment: dict[str, str | None]) -> list[str]:
    """Independent feasibility check (re-reads the rules from the state, not from the QUBO)."""
    v = []
    for t in st.tasks:
        if assignment.get(t.id) in (None, "CONFLICT"):
            v.append(f"{t.id} not assigned to exactly one person")
    if v:
        return v
    loads: dict[str, float] = {p.id: 0.0 for p in st.people}
    for t, r in assignment.items():
        loads[r] += st.task(t).est_h[r]
    for p in st.people:
        if p.avail_h is not None and loads[p.id] > p.avail_h + 1e-9:
            v.append(f"capacity: {p.id} {loads[p.id]:g} h > {p.avail_h:g} h")
    for c in st.constraints:
        if c.type == "at_most_one":
            mine = [t for t in c.tasks if assignment.get(t) == c.who]
            if len(mine) > 1:
                v.append(f"rule: {c.who} has more than one of {c.tasks}")
    return v


def brute_force(st: ProjectState, max_states: int = 2 ** 20) -> dict:
    people = [p.id for p in st.people]
    tasks = [t.id for t in st.tasks]
    if len(people) ** len(tasks) > max_states:
        return {"status": "skipped", "reason": "too many states"}
    t0 = time.perf_counter()
    best, feasible = None, []
    for combo in itertools.product(people, repeat=len(tasks)):
        a = dict(zip(tasks, combo))
        if not violations(st, a):
            c = true_cost(st, a)["total"]
            feasible.append((c, a))
            if best is None or c < best[0] - 1e-12:
                best = (c, a)
    dt_ = time.perf_counter() - t0
    if best is None:
        return {"status": "infeasible", "time_s": dt_}
    return {"status": "optimal", "assignment": best[1], "cost": round(best[0], 6), "n_feasible": len(feasible),
            "feasible_set": [a for _, a in feasible], "time_s": dt_}


def cpsat_assign(st: ProjectState, time_limit_s: float = 10.0) -> dict:
    """Exact assignment with capacity and rules as hard constraints. Durations scaled x10 to integers."""
    people = [p.id for p in st.people]
    tasks = [t.id for t in st.tasks]
    S = 10
    c = {(t, r): int(round(st.task(t).est_h[r] * S)) for t in tasks for r in people}
    m = cp_model.CpModel()
    x = {(t, r): m.NewBoolVar(f"x_{t}_{r}") for t in tasks for r in people}
    for t in tasks:
        m.AddExactlyOne(x[(t, r)] for r in people)
    L = {r: sum(c[(t, r)] * x[(t, r)] for t in tasks) for r in people}
    ub = sum(max(c[(t, r)] for r in people) for t in tasks) + 1
    for p in st.people:
        if p.avail_h is not None:
            m.Add(L[p.id] <= int(round(p.avail_h * S)))
    for con in st.constraints:
        if con.type == "at_most_one" and con.who in people:
            m.Add(sum(x[(t, con.who)] for t in con.tasks if t in tasks) <= 1)
    sq = []
    for a, b in itertools.combinations(people, 2):
        d = m.NewIntVar(-ub, ub, f"d_{a}_{b}")
        m.Add(d == L[a] - L[b])
        s = m.NewIntVar(0, ub * ub, f"s_{a}_{b}")
        m.AddMultiplicationEquality(s, [d, d])
        sq.append(s)
    E = sum(L.values())
    w = int(round(st.objective.mu * 100))
    m.Minimize(1000 * E + w * sum(sq))          # = 10^4 * (E + mu * sum D^2) in hour units
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = time_limit_s
    t0 = time.perf_counter()
    status = solver.Solve(m)
    dt_ = time.perf_counter() - t0
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": name.lower(), "time_s": dt_}
    a = {t: r for t in tasks for r in people if solver.Value(x[(t, r)])}
    return {"status": name.lower(), "assignment": a, "cost": true_cost(st, a)["total"], "time_s": dt_}


def _hhmm(start: str, units: int) -> str:
    h, mnt = map(int, start.split(":"))
    t = dt.datetime(2000, 1, 1, h, mnt) + dt.timedelta(minutes=30 * units)
    return t.strftime("%H:%M")


def schedule_slots(st: ProjectState, assignment: dict[str, str], day_start: str | None = None) -> dict:
    """Place assigned tasks in 30-minute units inside each person's window (Section 8.6)."""
    if day_start is None:
        cal = json.loads((config.ORG_DIR / "calendar.json").read_text())
        day_start = cal["workday"]["start"]
    m = cp_model.CpModel()
    U = 2
    horizon = {p.id: int(round((p.avail_h if p.avail_h is not None else 8) * U)) for p in st.people}
    s, e, iv = {}, {}, {}
    for t, r in assignment.items():
        d = int(round(st.task(t).est_h[r] * U))
        s[t] = m.NewIntVar(0, horizon[r], f"s_{t}")
        e[t] = m.NewIntVar(0, horizon[r], f"e_{t}")
        iv[t] = m.NewIntervalVar(s[t], d, e[t], f"i_{t}")
    for r in horizon:
        m.AddNoOverlap([iv[t] for t, rr in assignment.items() if rr == r])
    for a, b in st.dependencies:
        if a in s and b in s:
            m.Add(s[b] >= e[a])
    mk = m.NewIntVar(0, max(horizon.values()), "makespan")
    m.AddMaxEquality(mk, list(e.values()))
    m.Minimize(mk)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    st_ = solver.Solve(m)
    if st_ not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "timetable": {}}
    tt = {t: [_hhmm(day_start, solver.Value(s[t])), _hhmm(day_start, solver.Value(e[t]))] for t in assignment}
    return {"status": solver.StatusName(st_).lower(), "timetable": tt, "makespan_h": solver.Value(mk) / U, "day_start": day_start}
