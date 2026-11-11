"""Tiers A and B: perception, state, RAG, tools, ReAct."""
from qas import config
from qas.llm import RuleBasedPlanner
from qas.perception.text import extract_brief, extract_transcript
from qas.perception.vision import read_board
from qas.rag import KnowledgeBase
from qas.react import run_react
from qas.schemas import ToolCall
from qas.state import build_state
from qas.tools import ToolGateway

ROSTER = ["Anika", "Ravi", "Kiran", "Meera"]


def _demo_state():
    items = read_board(config.DEMO_DIR / "whiteboard.png")
    titles = [i.value["title"] for i in items if i.value["column"] != "Done"]
    items += extract_brief((config.DEMO_DIR / "brief.txt").read_text(), ROSTER)
    items += extract_transcript((config.DEMO_DIR / "voice_note.txt").read_text(), ROSTER, titles)
    return build_state("S-T", items)


def test_ocr_reads_demo_board():
    items = read_board(config.DEMO_DIR / "whiteboard.png")
    got = {(i.value["title"], i.value["column"]) for i in items}
    assert got == {("Payment API integration", "To Do"), ("UI testing", "To Do"), ("Release notes", "To Do"), ("Login bug fix", "Done")}
    assert all(i.conf > 0.85 for i in items)


def test_transcript_constraint_and_availability():
    out = extract_transcript("Ravi's on half-day leave Thursday, so he can take at most one of the UI testing and release notes.",
                             ROSTER, ["Payment API integration", "UI testing", "Release notes"])
    kinds = {e.kind: e for e in out}
    assert kinds["availability"].value == {"who": "Ravi", "hours": 4.0}
    assert kinds["constraint"].value["titles"] == ["UI testing", "Release notes"]
    assert kinds["constraint"].conf == 0.81


def test_state_excludes_done_column():
    st = _demo_state()
    assert [t.title for t in st.tasks] == ["Payment API integration", "UI testing", "Release notes"]
    assert [t.title for t in st.excluded_tasks] == ["Login bug fix"]
    assert st.date == "2026-11-12" and st.objective.mu == 0.1


def test_rag_estimates_reproduce_cost_table():
    kb = KnowledgeBase(today="2026-11-11")
    want = {("Payment API integration", "backend"): (3, 5), ("UI testing", "qa"): (4, 2), ("Release notes", "docs"): (2, 1)}
    for (title, skill), (a, r) in want.items():
        assert kb.estimate_effort(title, skill, "Anika")["hours"] == a
        assert kb.estimate_effort(title, skill, "Ravi")["hours"] == r
    top = kb.search_policy("partial leave cap tasks at most one of UI testing Release notes", k=2)
    assert "Task caps during leave" in [p["section"] for p in top]


def test_react_trace_matches_table_6_5():
    st, trace, status = run_react(_demo_state(), RuleBasedPlanner(), ToolGateway(KnowledgeBase(today="2026-11-11")))
    assert status == "COMPLETE"
    assert [h.tool for h in trace] == ["skill_lookup", "calendar_availability", "effort_history", "dependency_check", "policy_search", "stop check"]
    assert st.person("Ravi").conf == 0.95 and st.dependencies == [["T1", "T3"]]


def test_write_tool_refuses_without_approval():
    gw = ToolGateway(KnowledgeBase(today="2026-11-11"))
    r = gw.execute(ToolCall(tool="create_tickets", args={"decision_id": "D-X", "approval_token": "forged", "tickets": []}))
    assert not r.ok and "approval token" in r.error


def test_conflicting_sources_keep_restrictive_value():
    items = extract_brief("Plan Thursday for Anika and Ravi.", ROSTER)
    items += extract_transcript("Anika is on half-day leave Thursday.", ROSTER, [])
    st = build_state("S-C", items)
    st.tasks = []
    st, _, _ = run_react(st, RuleBasedPlanner(), ToolGateway(KnowledgeBase(today="2026-11-11")))
    assert st.person("Anika").avail_h == 4.0
    assert any(f.startswith("conflict") for f in st.flags)
