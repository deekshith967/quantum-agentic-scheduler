"""Layer 13 benchmark: synthetic sprint cases run end to end (Algorithm 10.1).

Each case has a rendered task-board image, a typed brief, a voice-note transcript and a
mock calendar. The script runs perception -> ReAct -> formulation -> QAOA (p = 1, 2, 3,
forced) and CP-SAT, then compares everything with gold labels and the exact optimum.

All data are synthetic and generated with a fixed seed. Results are written to
benchmarks/results/ and summarised in benchmarks/RESULTS.md. Nothing here is hand-edited.

    python scripts/run_benchmark.py --cases 30 --restarts 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as stats
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QAS_RUNTIME", tempfile.mkdtemp(prefix="qas_bench_"))

from qas import config  # noqa: E402
from qas.evaluation import extraction_items, gold_items, prf  # noqa: E402
from qas.perception.board_render import render_board  # noqa: E402

POOL = [("Payment API integration", "backend"), ("Orders API endpoint", "backend"), ("Refund service integration", "backend"),
        ("UI testing", "qa"), ("Checkout regression testing", "qa"), ("Search UI testing", "qa"),
        ("Release notes", "docs"), ("API changelog notes", "docs"), ("Cart layout component", "frontend")]
DONE_POOL = ["Login bug fix", "Cache warmup script", "Logo update"]
PEOPLE = ["Anika", "Ravi", "Kiran"]


def make_case(k: int, rng: random.Random, out: Path) -> dict:
    n_people = 2 if k < 20 else 3
    people = rng.sample(PEOPLE, n_people)
    n_tasks = rng.randint(3, 4) if n_people == 2 else rng.randint(3, 4)
    tasks = rng.sample(POOL, n_tasks)
    done = rng.choice(DONE_POOL)
    avail = {p: rng.choice([6, 8, 8]) for p in people}
    voice, voice_avail, rules = [], {}, []
    conflict = False
    if rng.random() < 0.6:
        who = rng.choice(people)
        voice.append(f"{who}'s on half-day leave Thursday.")
        voice_avail[who] = 4.0
        if rng.random() < 0.25:
            conflict = True          # calendar still says a full day
        else:
            avail[who] = 4
    if rng.random() < 0.6:
        who = rng.choice(people)
        a, b = rng.sample([t for t, _ in tasks], 2)
        voice.append(f"{who} can take at most one of the {a.lower()} and {b.lower()}.")
        rules.append({"who": who, "titles": [a, b]})
    cdir = out / f"case_{k:02d}"
    org = cdir / "org"
    org.mkdir(parents=True, exist_ok=True)
    (org / "people.json").write_text((config.ORG_DIR / "people.json").read_text())
    (org / "dependencies.json").write_text((config.ORG_DIR / "dependencies.json").read_text())
    (org / "calendar.json").write_text(json.dumps({"workday": {"start": "09:00", "hours": 8}, "availability": {"2026-11-12": avail}}))
    notes = [("To Do", t) for t, _ in tasks] + [("Done", done)]
    render_board(notes, cdir / "board.png", seed=k)
    names = " and ".join([", ".join(people[:-1]), people[-1]]) if len(people) > 2 else " and ".join(people)
    (cdir / "brief.txt").write_text(f"Plan Thursday's release tasks for {names}, keep the load fair.")
    (cdir / "voice.txt").write_text(" ".join(voice) or "No changes from me.")
    return {"case_id": f"BM-{k:02d}", "dir": str(cdir), "people": people, "conflict": conflict,
            "gold": {"tasks": [t for t, _ in tasks], "voice_availability": voice_avail, "rules": rules},
            "hitl_expected": "flagged" if (rules or conflict) else "standard"}


def run(n_cases: int, restarts: int, seed: int) -> dict:
    from qas.pipeline import Orchestrator
    rng = random.Random(seed)
    out = Path(tempfile.mkdtemp(prefix="qas_cases_"))
    cases = [make_case(k, rng, out) for k in range(n_cases)]
    orig_org = config.ORG_DIR
    rows = []
    for c in cases:
        config.ORG_DIR = Path(c["dir"]) / "org"
        o = Orchestrator(today="2026-11-11")
        row = {"case_id": c["case_id"], "people": len(c["people"]), "expected_hitl": c["hitl_expected"]}
        sid = o.new_session()["session_id"]
        d = Path(c["dir"])
        o.add_input(sid, "text", text=(d / "brief.txt").read_text())
        o.add_input(sid, "image", path=str(d / "board.png"))
        o.add_input(sid, "audio", path=str(d / "voice.txt"))
        t0 = time.perf_counter()
        s = o.perceive(sid)
        row["perception_s"] = round(time.perf_counter() - t0, 3)
        from qas.schemas import ProjectState
        row["extraction"] = prf(extraction_items(ProjectState(**s["state_draft"])), gold_items(c))
        row["depths"] = {}
        for p in (1, 2, 3):
            t1 = time.perf_counter()
            s = o.plan(sid, force_arm=f"qaoa_p{p}", restarts=restarts)
            lat = time.perf_counter() - t1
            if s.get("decision") is None:
                row["depths"][p] = {"status": "no decision", "error": s.get("error")}
                continue
            q = s["qaoa"] or {}
            ex = s["exact"]
            opt_bits = None
            for cand in q.get("candidates", []):
                if cand["assignment"] == ex.get("assignment"):
                    opt_bits = cand["bits"]
            dec = s["decision"]
            row["depths"][p] = {
                "status": "ok", "n_qubits": q.get("qubo", {}).get("n_qubits"), "encoding": q.get("qubo", {}).get("encoding"),
                "chunks": len(q.get("chunks", [])) or 1, "solver_used": dec["solver"]["used"],
                "fallback_reason": dec["solver"].get("reason") if dec["solver"]["used"] != "qaoa" else None,
                "P_feasible": q.get("P_feasible"), "P_opt": q.get("probs", {}).get(opt_bits, 0.0) if opt_bits else 0.0,
                "gap_final": round((dec["cost"]["total"] - ex["cost"]) / ex["cost"], 6) if ex.get("cost") else None,
                "cpsat_gap": round((s["baseline"]["cost"] - ex["cost"]) / ex["cost"], 6) if ex.get("cost") else None,
                "hitl": dec["hitl"]["level"], "latency_s": round(lat, 3), "violations_in_proposal": len(dec["violations"])}
        rows.append(row)
        print(c["case_id"], row["extraction"]["f1"], {p: (v.get("solver_used"), v.get("P_opt")) for p, v in row["depths"].items()}, flush=True)
    config.ORG_DIR = orig_org
    return {"config": {"cases": n_cases, "restarts": restarts, "seed": seed, "shots": config.SHOTS, "optimiser_sim": config.OPTIMISER_SIM,
                       "backend": "Qiskit Aer (classical simulation)"}, "rows": rows}


def summarise(res: dict) -> str:
    rows = res["rows"]
    f1 = [r["extraction"]["f1"] for r in rows]
    lines = ["# Benchmark results (synthetic cases, generated by scripts/run_benchmark.py)", "",
             f"Configuration: {json.dumps(res['config'])}", "",
             "All quantum results are classical simulations with Qiskit Aer. These numbers describe this prototype on synthetic data only.", "",
             "## Perception (Layers 2 and 3)", "",
             f"* Extraction F1 over {len(rows)} cases: mean {stats.mean(f1):.3f}, min {min(f1):.3f}",
             f"* Perception time per case: mean {stats.mean(r['perception_s'] for r in rows):.2f} s", "",
             "## Optimisation and decision (Layers 8 to 12)", "",
             "| Depth | Cases with decision | QAOA accepted | Classical fallback | Mean P(feasible) | Mean P(optimal) | Mean final gap | Proposals with violations | HITL level correct | Mean latency (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in ("1", "2", "3"):
        ds = [(r, r["depths"].get(p) or r["depths"].get(int(p))) for r in rows]
        ok = [(r, d) for r, d in ds if d and d.get("status") == "ok"]
        if not ok:
            continue
        acc = sum(1 for _, d in ok if d["solver_used"] == "qaoa")
        fb = len(ok) - acc
        hitl_ok = sum(1 for r, d in ok if d["hitl"] == r["expected_hitl"])
        lines.append(f"| {p} | {len(ok)}/{len(rows)} | {acc} | {fb} | {stats.mean(d['P_feasible'] or 0 for _, d in ok):.3f} | "
                     f"{stats.mean(d['P_opt'] for _, d in ok):.3f} | {stats.mean(d['gap_final'] for _, d in ok) * 100:.2f} % | "
                     f"{sum(d['violations_in_proposal'] > 0 for _, d in ok)} | {hitl_ok}/{len(ok)} | {stats.mean(d['latency_s'] for _, d in ok):.2f} |")
    two = [d for r in rows for p, d in r["depths"].items() if d.get("status") == "ok" and d["encoding"] == "binary"]
    oh = [d for r in rows for p, d in r["depths"].items() if d.get("status") == "ok" and d["encoding"] == "one_hot"]
    lines += ["", "## By encoding (all depths pooled)", "",
              f"* Binary encoding (2 people): {len(two)} runs, QAOA accepted in {sum(d['solver_used'] == 'qaoa' for d in two)}, mean P(optimal) {stats.mean([d['P_opt'] for d in two] or [0]):.3f}",
              f"* One-hot encoding (3 people): {len(oh)} runs, QAOA accepted in {sum(d['solver_used'] == 'qaoa' for d in oh)}, mean P(optimal) {stats.mean([d['P_opt'] for d in oh] or [0]):.3f}",
              f"* CP-SAT gap to the exact optimum: max {max((d['cpsat_gap'] or 0) for r in rows for d in r['depths'].values() if d.get('status') == 'ok') * 100:.2f} %",
              "", "## Cases without a decision", ""]
    none = [(r["case_id"], p, d.get("error")) for r in rows for p, d in r["depths"].items() if d.get("status") != "ok"]
    lines += [f"* {c} p={p}: {e}" for c, p, e in none] or ["* none"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    res = run(a.cases, a.restarts, a.seed)
    outd = ROOT / "benchmarks" / "results"
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "benchmark.json").write_text(json.dumps(res, indent=1, default=str))
    md = summarise(res)
    (ROOT / "benchmarks" / "RESULTS.md").write_text(md)
    print(md)
