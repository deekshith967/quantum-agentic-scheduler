"""Layer 9: the QUBO and Ising coefficients must match Calculations 7.3 and 7.4 of the report."""
import numpy as np

from qas.qubo import build_qubo, Poly

REPORT_C = {"000": 17.1, "001": 11.6, "010": 7.9, "011": 16.0, "100": 11.1, "101": 10.4, "110": 11.5, "111": 24.4}


def test_qubo_coefficients(scenario):
    t = build_qubo(scenario).coefficient_table()
    assert t["constant"] == 17.1
    assert t["linear"] == {"x1": -6.0, "x2": -9.2, "x3": -5.5}
    assert t["quadratic"] == {"x1x2": 9.6, "x1x3": 4.8, "x2x3": 13.6}


def test_enumeration_matches_table_7_3(scenario):
    m = build_qubo(scenario)
    for bits, c in REPORT_C.items():
        assert abs(m.cost([int(b) for b in bits]) - c) < 1e-9
    assert abs(m.cost_vector().mean() - 13.75) < 1e-9


def test_ising_matches_equation_7_6(scenario):
    h, J, const = build_qubo(scenario).to_ising()
    assert h == {0: -0.6, 1: -1.2, 2: -1.85}
    assert J == {(0, 1): 2.4, (0, 2): 1.2, (1, 2): 3.4}
    assert const == 13.75


def test_sparse_pauli_diagonal_and_qiskit_optimization(scenario):
    m = build_qubo(scenario)
    op, const = m.sparse_pauli()
    diag = np.real(np.diag(op.to_matrix())) + const
    assert np.allclose(diag, m.cost_vector())
    qop, off = m.quadratic_program().to_ising()
    assert np.allclose(np.real(np.diag(qop.to_matrix())) + off, m.cost_vector())


def test_one_hot_encoding_penalises_unassigned(scenario):
    scenario.people.append(__import__("qas.schemas", fromlist=["Person"]).Person(id="Kiran", avail_h=8))
    for t in scenario.tasks:
        t.est_h["Kiran"] = 3
    m = build_qubo(scenario)
    assert m.encoding == "one_hot" and m.n == 9
    valid = m.encode({"T1": "Anika", "T2": "Ravi", "T3": "Kiran"})
    empty = [0] * 9
    assert m.cost(empty) > m.cost(valid)


def test_poly_square_uses_x_squared_equals_x():
    p = Poly(1.0, {0: 2.0})
    sq = p.mul_affine(p)
    assert sq.c0 == 1.0 and sq.lin[0] == 2 * 2.0 + 4.0
