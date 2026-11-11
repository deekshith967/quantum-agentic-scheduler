"""Orchestrator: runs one planning request through the layers and keeps the session record.

The report proposes LangGraph for orchestration. The prototype uses an explicit Python
state machine with the same nodes (perceive -> react -> select -> formulate -> solve ->
decide -> review -> write) so it runs without extra services.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import config, store
from .bandit import ContextualBandit, bucket
from .decision import accept, candidates_from_counts, hitl_triggers, summarise
from .evaluation import ground_truth_record
from .feedback import bandit_feedback, export_finetune, record_outcomes
from .hitl import ApprovalRegistry, ReviewService
from .llm import explain, get_planner
from .perception.audio import transcribe
from .perception.text import extract_brief, extract_transcript
from .perception.vision import read_board
from .qubo import build_qubo, coupling_density, decompose, qubits_needed
from .rag import KnowledgeBase
from .react import run_react
from .schemas import Decision, ProjectState
from .solvers.classical import brute_force, cpsat_assign, schedule_slots, true_cost
from .solvers.qaoa import QAOASolver
from .state import build_state
from .tools import ToolGateway


def roster() -> list[str]:
    return [p["id"] for p in json.loads((config.ORG_DIR / "people.json").read_text())["people"]]


class Orchestrator:
    def __init__(self, today: str | None = None):
        self.kb = KnowledgeBase(today=today)
        self.approvals = ApprovalRegistry()
        self.gw = ToolGateway(self.kb, approvals=self.approvals.check)
        self.review = ReviewService(self.gw, self.approvals)
        self.bandit = ContextualBandit()
        self.planner = get_planner()

    # ------------------------------------------------------------- sessions
    def new_session(self) -> dict:
        sid = "S-" + uuid.uuid4().hex[:6].upper()
        s = {"session_id": sid, "inputs": {}, "created": time.time()}
        store.save_json("sessions", sid, s)
        store.audit("PM", "session:new", sid)
        return s

    def get(self, sid: str) -> dict:
        s = store.load_json("sessions", sid)
        if s is None:
            raise KeyError(sid)
        return s

    def save(self, s: dict) -> None:
        store.save_json("sessions", s["session_id"], s)

    def add_input(self, sid: str, kind: str, text: str | None = None, path: str | None = None) -> dict:
        s = self.get(sid)
        if kind == "text":
            s["inputs"]["brief_text"] = text
        elif kind == "image":
            s["inputs"]["image_path"] = path
        elif kind == "audio":
            s["inputs"]["audio_path"] = path
        store.audit("PM", f"input:{kind}", sid)
        self.save(s)
        return s

    def load_demo(self, sid: str) -> dict:
        d = config.DEMO_DIR
        self.add_input(sid, "text", text=(d / "brief.txt").read_text().strip())
        self.add_input(sid, "image", path=str(d / "whiteboard.png"))
        return self.add_input(sid, "audio", path=str(d / "voice_note.txt"))

    # ------------------------------------------------------------- tier A
    def perceive(self, sid: str) -> dict:
        s = self.get(sid)
        inp, items, timing = s["inputs"], [], {}
        names = roster()
        t0 = time.perf_counter()
        if inp.get("image_path"):
            items += read_board(inp["image_path"])
        timing["vision_s"] = round(time.perf_counter() - t0, 3)
        titles = [i.value["title"] for i in items if i.kind == "task" and i.value.get("column") != "Done"]
        if inp.get("brief_text"):
            items += extract_brief(inp["brief_text"], names)
        if inp.get("audio_path"):
            t1 = time.perf_counter()
            text, conf, method = transcribe(inp["audio_path"])
            timing["stt_s"] = round(time.perf_counter() - t1, 3)
            s["inputs"].update({"transcript_text": text, "stt_conf": conf, "stt_method": method})
            items += extract_transcript(text, names, titles, stt_conf=conf)
        st = build_state(sid, items)
        s["perception"] = [i.model_dump() for i in items]
        s["perception_timing"] = timing
        s["state_draft"] = st.model_dump()
        s["state"] = st.model_dump()
        self.save(s)
        store.audit("system", "perceive", sid, items=len(items))
        return s

    # ------------------------------------------------------------- tiers B to D
    def plan(self, sid: str, mode: str = "quantum_eval", restarts: int | None = None, noisy: bool = False,
             force_arm: str | None = None, optimiser_sim: str | None = None) -> dict:
        s = self.get(sid)
        if "state" not in s:
            s = self.perceive(sid)
        st = ProjectState(**s["state_draft"])
        t_all = time.perf_counter()
        st, trace, status = run_react(st, self.planner, self.gw)
        s["trace"] = [h.model_dump() for h in trace]
        s["react_status"] = status
        s["planner"] = self.planner.name
        if status != "COMPLETE" or not st.tasks or len(st.people) < 2:
            s["state"] = st.model_dump()
            s["decision"] = None
            s["error"] = f"evidence incomplete ({status}); missing={st.missing}; flags={st.flags}"
            self.save(s)
            return s
        # L8 strategy
        nq = qubits_needed(st)
        chunks = decompose(st) if nq > config.QUBIT_BUDGET else [[t.id for t in st.tasks]]
        probe = build_qubo(st, chunks[0])
        ctx = bucket(probe.n, coupling_density(probe), deadline=False)
        sel = self.bandit.select(ctx, mode)
        if force_arm:
            sel = {**sel, "arm": force_arm, "rule": "forced by user"}
        s["bandit"] = sel
        # L10 classical baselines (always run)
        exact = brute_force(st)
        base = cpsat_assign(st)
        s["exact"] = {k: v for k, v in exact.items() if k != "feasible_set"}
        s["baseline"] = base
        # L9 + L10 quantum
        qres, fallback, assignment, solver_info = None, None, None, {}
        if sel["arm"].startswith("qaoa"):
            p = int(sel["arm"][-1])
            parts, combined, base_load, feasible_all = [], {}, {r.id: 0.0 for r in st.people}, True
            for ch in chunks:
                model = build_qubo(st, ch, base_load=base_load)
                solver = QAOASolver(model, optimiser_sim=optimiser_sim or config.OPTIMISER_SIM)
                r = solver.solve(p, restarts=restarts or config.QAOA_RESTARTS, noisy_too=noisy)
                feas, rej = candidates_from_counts(st, model, r["counts"])
                r["qubo"] = {"encoding": model.encoding, "n_qubits": model.n, "variables": [list(v) for v in model.variables],
                             "coefficients": model.coefficient_table(), "penalty_P": model.penalty_P, "mu": model.mu,
                             "unencoded": model.unencoded, "tasks": ch, "base_load": dict(base_load)}
                h, J, c0 = model.to_ising()
                r["ising"] = {"h": {f"z{i+1}": v for i, v in h.items()}, "J": {f"z{i+1}z{j+1}": v for (i, j), v in J.items()}, "constant": c0}
                r["candidates"] = feas[:8]
                r["rejected"] = [{"bits": x["bits"], "count": x["count"], "violations": x["violations"]} for x in rej[:8]]
                r["P_feasible"] = round(sum(r["probs"].get(x["bits"], 0) for x in feas), 4)
                parts.append(r)
                if not feas:
                    feasible_all = False
                    break
                combined.update(feas[0]["assignment"])
                for t, who in feas[0]["assignment"].items():
                    base_load[who] += st.task(t).est_h[who]
            qres = parts[0] if len(parts) == 1 else {**parts[0], "chunks": parts}
            s["qaoa_feasible"] = feasible_all
            if feasible_all and base.get("cost") is not None:
                qc = true_cost(st, combined)["total"]
                acc = accept(qc, base["cost"])
                if acc["accept"]:
                    assignment = combined
                    mode_count = parts[0]["candidates"][0]["count"]
                    solver_info = {"used": "qaoa", "p": p, "backend": parts[0]["backend"], "shots": parts[0]["shots"], "mode_count": mode_count,
                                   "chunks": len(parts), "baseline": "cp-sat", "baseline_cost": base["cost"], "gap": acc["gap"],
                                   "gap_pct": round(acc["gap"] * 100, 2), "threshold": acc["threshold"], "qaoa_cost": qc}
                else:
                    fallback = f"quantum underperformed (cost {qc} > {acc['threshold']})"
            elif not feasible_all:
                fallback = "no feasible QAOA sample"
        if assignment is None:
            if base.get("assignment") is None:
                s["state"] = st.model_dump()
                s["decision"] = None
                s["error"] = f"no feasible assignment ({base.get('status')})"
                self.save(s)
                return s
            assignment = base["assignment"]
            solver_info = {"used": "cp-sat", "reason": fallback or "bandit chose classical arm", "baseline": "cp-sat",
                           "baseline_cost": base["cost"], "gap": 0.0, "gap_pct": 0.0}
        s["qaoa"] = qres
        # L10 time slots + L11 decision
        slots = schedule_slots(st, assignment)
        summ = summarise(st, assignment)
        trig = hitl_triggers(st, assignment, fallback)
        dec = Decision(decision_id=f"D-{sid[2:]}-{len(store.list_json('decisions')) + 1}", session_id=sid, assignment=assignment,
                       timetable=slots["timetable"], solver=solver_info, violations=[], min_confidence=trig["min_confidence"], hitl=trig,
                       evidence=[t.src for t in st.tasks] + sorted({c.src for c in st.constraints}) +
                                [f"eff:{','.join(e['doc_id'] for e in t.evidence.get(assignment[t.id], []))}" for t in st.tasks], **summ)
        dec.explanation = explain(dec, st)
        dec = self.review.submit(dec)
        s["state"] = st.model_dump()
        s["decision"] = dec.model_dump()
        s["timing"] = {"plan_total_s": round(time.perf_counter() - t_all, 3)}
        s["ground_truth"] = ground_truth_record(s["decision"], exact)
        store.save_json("decisions", dec.decision_id, s["decision"])
        self.save(s)
        return s

    # ------------------------------------------------------------- tier D/E: review and feedback
    def _dec(self, s: dict) -> tuple[Decision, ProjectState]:
        if not s.get("decision"):
            raise ValueError("no decision for this session")
        return Decision(**s["decision"]), ProjectState(**s["state"])

    def review_action(self, sid: str, verdict: str, edits: dict | None = None, comment: str = "", acknowledged: list[str] | None = None) -> dict:
        s = self.get(sid)
        dec, st = self._dec(s)
        s.setdefault("review_log", [])
        result = {}
        if verdict == "approve":
            history = dec.review if dec.review.get("verdict") == "edited" else None
            dec, result = self.review.approve(dec, st, acknowledged)
            if history:
                dec.review.update({"edited": True, "previous": history["previous"], "edited_to": history["edited_to"]})
        elif verdict == "edit":
            dec = self.review.edit(dec, st, edits or {})
        elif verdict == "reject":
            dec = self.review.reject(dec, comment)
        else:
            raise ValueError("verdict must be approve, edit or reject")
        s["review_log"].append({"verdict": verdict, "edits": edits, "comment": comment, "status_after": dec.status, "ts": time.time()})
        s["decision"] = dec.model_dump()
        if dec.status in ("approved", "rejected"):
            s["bandit_update"] = bandit_feedback(self.bandit, s, "rejected" if dec.status == "rejected" else "approved")
            s["finetune_export"] = export_finetune(s)
            s["ground_truth"] = ground_truth_record(s["decision"], s["exact"])
        s["write_result"] = result or s.get("write_result")
        store.save_json("decisions", dec.decision_id, s["decision"])
        self.save(s)
        return s

    def record_outcomes(self, sid: str, outcomes: list[dict]) -> dict:
        s = self.get(sid)
        out = record_outcomes(s, outcomes, kb=self.kb)
        s["outcomes"] = out
        s["ground_truth"] = ground_truth_record(s["decision"], s["exact"], out)
        self.save(s)
        return s
