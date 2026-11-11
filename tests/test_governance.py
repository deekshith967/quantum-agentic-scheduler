"""Tiers C to E: bandit, decision, HITL, feedback, full pipeline and API."""
import pytest
from fastapi.testclient import TestClient

from qas.bandit import ContextualBandit, reward
from qas.decision import accept


def test_ucb_selects_p2_from_illustrative_prior(tmp_path):
    b = ContextualBandit(path=tmp_path / "b.json")
    sel = b.select("q<=4|dense|no-deadline")
    assert sel["arm"] == "qaoa_p2"
    assert sel["scores"] == {"qaoa_p1": 1.6097, "qaoa_p2": 1.7365, "qaoa_p3": 1.3865}


def test_bandit_update_equation_11_2(tmp_path):
    b = ContextualBandit(path=tmp_path / "b.json")
    r = reward(True, 0.0, 4.8, False)
    assert r == 1.52
    u = b.update("q<=4|dense|no-deadline", "qaoa_p2", r)
    assert u["Q_after"] == 1.264 and u["n"] == 3


def test_acceptance_rule():
    assert accept(7.9, 7.9)["accept"] and accept(7.9, 7.9)["threshold"] == 8.295
    assert not accept(9.0, 7.9)["accept"]


@pytest.fixture(scope="module")
def planned():
    from qas.pipeline import Orchestrator
    o = Orchestrator(today="2026-11-11")
    s = o.new_session()
    o.load_demo(s["session_id"])
    return o, o.plan(s["session_id"], force_arm="qaoa_p2")


def test_pipeline_decision(planned):
    _, s = planned
    d = s["decision"]
    assert d["assignment"] == {"T1": "Anika", "T2": "Ravi", "T3": "Anika"}
    assert d["cost"]["total"] == 7.9 and d["solver"]["used"] == "qaoa" and d["solver"]["gap"] == 0.0
    assert d["status"] == "flagged" and "0.81" in d["hitl"]["reasons"][0]


def test_hitl_requires_acknowledgement_then_writes(planned):
    o, s = planned
    sid = s["session_id"]
    with pytest.raises(ValueError):
        o.review_action(sid, "approve")
    s2 = o.review_action(sid, "approve", acknowledged=s["decision"]["hitl"]["reasons"])
    assert s2["decision"]["status"] == "approved" and len(s2["write_result"]["tickets"]) == 3
    assert s2["finetune_export"]["sft"] == 1
    s3 = o.record_outcomes(sid, [{"task_id": "T1", "actual_h": 3.6}, {"task_id": "T2", "actual_h": 2.3}, {"task_id": "T3", "actual_h": 1.8}])
    assert s3["ground_truth"]["effort_mae_h"] == pytest.approx(0.3667, abs=1e-4)


def test_edit_that_breaks_a_rule_is_blocked():
    from qas.pipeline import Orchestrator
    o = Orchestrator(today="2026-11-11")
    sid = o.new_session()["session_id"]
    o.load_demo(sid)
    o.plan(sid, force_arm="qaoa_p1")
    s = o.review_action(sid, "edit", edits={"T1": "Anika", "T2": "Ravi", "T3": "Ravi"})
    assert s["decision"]["status"] == "flagged" and s["decision"]["violations"]
    s = o.review_action(sid, "edit", edits={"T1": "Anika", "T2": "Anika", "T3": "Ravi"})
    assert s["decision"]["status"] == "pending" and not s["decision"]["violations"]


def test_api_end_to_end():
    from qas.api import app
    c = TestClient(app)
    assert c.get("/health").json()["ok"]
    sid = c.post("/v1/sessions").json()["session_id"]
    c.post(f"/v1/sessions/{sid}/demo")
    assert c.post(f"/v1/sessions/{sid}/perceive").status_code == 200
    s = c.post(f"/v1/sessions/{sid}/plan", json={"force_arm": "cpsat", "mode": "production"}).json()
    assert s["decision"]["solver"]["used"] == "cp-sat"
    r = c.post(f"/v1/sessions/{sid}/review", json={"verdict": "approve"})
    assert r.status_code == 409          # flagged review needs acknowledgement
    r = c.post(f"/v1/sessions/{sid}/review", json={"verdict": "approve", "acknowledged": s["decision"]["hitl"]["reasons"]})
    assert r.json()["decision"]["status"] == "approved"
    assert len(c.get("/v1/mock/jira").json()) >= 3
