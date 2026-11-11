"""Layer 10: QAOA on Aer and the classical baselines."""
import numpy as np

from qas.qubo import build_qubo
from qas.solvers.classical import brute_force, cpsat_assign, schedule_slots, violations
from qas.solvers.qaoa import QAOASolver


def test_numpy_statevector_equals_qiskit(scenario):
    m = build_qubo(scenario)
    a, b = QAOASolver(m, optimiser_sim="numpy"), QAOASolver(m, optimiser_sim="qiskit")
    rng = np.random.default_rng(0)
    for p in (1, 2, 3):
        v = rng.uniform(0, np.pi, 2 * p)
        assert abs(a.expectation(p, v) - b.expectation(p, v)) < 1e-9
        assert np.allclose(a.probabilities(p, v), b.probabilities(p, v), atol=1e-10)


def test_report_p1_optimum_reproduces(scenario):
    """Table 8.2, p = 1: gamma = 0.9935, beta = 2.5373 gives F = 10.19 and P(010) = 0.385."""
    s = QAOASolver(build_qubo(scenario), optimiser_sim="qiskit")
    v = np.array([0.9935, 2.5373])
    assert abs(s.expectation(1, v) - 10.19) < 0.01
    probs = s.probabilities(1, v)
    assert abs(probs[0b010] - 0.3848) < 0.001      # index bit i = x_{i+1}; 010 -> x2 = 1 -> index 2


def test_qaoa_solve_samples_on_aer(scenario):
    r = QAOASolver(build_qubo(scenario), optimiser_sim="numpy").solve(2, restarts=6)
    assert sum(r["counts"].values()) == 1024
    assert max(r["counts"], key=r["counts"].get) == "010"
    assert r["circuit"]["cx"] == 12
    assert "simulation" in r["executed_on"]


def test_bit_order_decode(scenario):
    m = build_qubo(scenario)
    assert m.decode([0, 1, 0]) == {"T1": "Anika", "T2": "Ravi", "T3": "Anika"}
    assert m.decode([0, 0, 1]) == {"T1": "Anika", "T2": "Anika", "T3": "Ravi"}


def test_brute_force_and_cpsat_agree(scenario):
    bf = brute_force(scenario)
    assert bf["assignment"] == {"T1": "Anika", "T2": "Ravi", "T3": "Anika"} and bf["cost"] == 7.9
    assert bf["n_feasible"] == 2           # only 001 and 010 respect capacity and the rule
    cp = cpsat_assign(scenario)
    assert cp["status"] == "optimal" and cp["assignment"] == bf["assignment"] and cp["cost"] == 7.9


def test_feasibility_check_catches_capacity(scenario):
    v = violations(scenario, {"T1": "Ravi", "T2": "Anika", "T3": "Anika"})
    assert any("capacity: Ravi 5 h > 4 h" in x for x in v)


def test_time_slots_match_figure_8_5(scenario):
    s = schedule_slots(scenario, {"T1": "Anika", "T2": "Ravi", "T3": "Anika"})
    assert s["timetable"] == {"T1": ["09:00", "12:00"], "T2": ["09:00", "11:00"], "T3": ["12:00", "14:00"]}
    assert s["makespan_h"] == 5.0
