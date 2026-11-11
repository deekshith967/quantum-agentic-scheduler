"""Command-line run of the report's scenario through all layers (no UI needed).

    python scripts/run_demo.py            # QAOA arm chosen by the bandit, auto-approve after acknowledging flags
    python scripts/run_demo.py --arm cpsat --no-approve
"""
import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QAS_RUNTIME", tempfile.mkdtemp(prefix="qas_demo_"))

from qas.pipeline import Orchestrator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="qaoa_p1 | qaoa_p2 | qaoa_p3 | cpsat (default: bandit)")
    ap.add_argument("--noisy", action="store_true", help="also sample with the illustrative noise model")
    ap.add_argument("--optimiser", default=None, help="numpy | qiskit")
    ap.add_argument("--no-approve", action="store_true")
    a = ap.parse_args()
    o = Orchestrator(today="2026-11-11")
    sid = o.new_session()["session_id"]
    o.load_demo(sid)
    s = o.perceive(sid)
    print("== Perception (L2, L3)")
    for i in s["perception"]:
        print(f"  {i['kind']:<12} {i['src']:<8} conf {i['conf']:.2f}  {json.dumps(i['value'])[:90]}")
    s = o.plan(sid, mode="production" if a.arm == "cpsat" else "quantum_eval", force_arm=a.arm, noisy=a.noisy, optimiser_sim=a.optimiser)
    print("== ReAct multi-hop trace (L4 to L7)")
    for h in s["trace"]:
        print(f"  {h['hop']}  {h['tool']:<22} {h['effect']}")
    print("== Bandit (L8):", s["bandit"]["arm"], s["bandit"]["scores"])
    q = s.get("qaoa")
    if q:
        print("== QUBO (L9):", q["qubo"]["coefficients"])
        print("== Ising:", q["ising"])
        print(f"== QAOA (L10) p={q['p']} F={q['F']} gammas={q['gammas']} betas={q['betas']} [{q['executed_on']}]")
        print("   counts:", q["counts"])
        if q.get("counts_noisy"):
            print("   noisy counts (illustrative model):", q["counts_noisy"])
    d = s["decision"]
    print("== Baselines: brute force", s["exact"]["assignment"], s["exact"]["cost"], "| CP-SAT", s["baseline"]["assignment"], s["baseline"]["cost"])
    print("== Decision (L11):", d["decision_id"], d["assignment"], d["cost"], d["timetable"])
    print("   status:", d["status"], d["hitl"]["reasons"])
    print("   explanation:", d["explanation"])
    if not a.no_approve:
        s = o.review_action(sid, "approve", acknowledged=d["hitl"]["reasons"])
        print("== HITL (L12): approved; writes", s["write_result"])
        print("== Feedback (L14): bandit", s.get("bandit_update"), "| fine-tune export", s["finetune_export"])
        s = o.record_outcomes(sid, [{"task_id": "T1", "actual_h": 3.6}, {"task_id": "T2", "actual_h": 2.3}, {"task_id": "T3", "actual_h": 1.8}])
        print("== Ground truth (L13):", json.dumps(s["ground_truth"]))


if __name__ == "__main__":
    main()
