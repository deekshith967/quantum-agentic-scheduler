"""Layer 9: problem formulation (QUBO and Ising) for task-to-person assignment.

Objective (Section 7.3 of the report):
    C(x) = E(x) + mu * sum_{r<s} (L_r - L_s)^2 + penalties
For two people this is exactly E + mu * D^2 with D = L_A - L_B (Equation 7.5).

Encodings:
* binary   (2 people): x_t = 0 -> first person, x_t = 1 -> second person.
* one_hot  (M people): x_{t,r} = 1 if task t goes to r, plus lambda * (1 - sum_r x_{t,r})^2.

Constraints encoded as penalties: "at most one of" rules (P * pairwise products).
Constraints NOT encoded: capacity (would need slack qubits). They are returned in
`unencoded` and enforced by Layer 11 and by the classical solvers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import config
from .schemas import ProjectState


class Poly:
    """Quadratic pseudo-Boolean polynomial c0 + sum a_i x_i + sum_{i<j} b_ij x_i x_j."""

    def __init__(self, c0: float = 0.0, lin: dict | None = None, quad: dict | None = None):
        self.c0 = float(c0)
        self.lin: dict[int, float] = dict(lin or {})
        self.quad: dict[tuple[int, int], float] = dict(quad or {})

    @staticmethod
    def var(i: int, coef: float = 1.0) -> "Poly":
        return Poly(0.0, {i: coef})

    def __add__(self, o: "Poly | float") -> "Poly":
        if not isinstance(o, Poly):
            return Poly(self.c0 + o, self.lin, self.quad)
        out = Poly(self.c0 + o.c0, self.lin, self.quad)
        for i, a in o.lin.items():
            out.lin[i] = out.lin.get(i, 0.0) + a
        for k, b in o.quad.items():
            out.quad[k] = out.quad.get(k, 0.0) + b
        return out

    __radd__ = __add__

    def scale(self, s: float) -> "Poly":
        return Poly(self.c0 * s, {i: a * s for i, a in self.lin.items()}, {k: b * s for k, b in self.quad.items()})

    def __sub__(self, o: "Poly") -> "Poly":
        return self + o.scale(-1.0)

    def mul_affine(self, o: "Poly") -> "Poly":
        assert not self.quad and not o.quad, "only affine x affine products are supported"
        out = Poly(self.c0 * o.c0)
        for i, a in self.lin.items():
            out.lin[i] = out.lin.get(i, 0.0) + a * o.c0
        for j, b in o.lin.items():
            out.lin[j] = out.lin.get(j, 0.0) + b * self.c0
        for (i, a), (j, b) in itertools.product(self.lin.items(), o.lin.items()):
            if i == j:
                out.lin[i] = out.lin.get(i, 0.0) + a * b        # x_i^2 = x_i
            else:
                k = (min(i, j), max(i, j))
                out.quad[k] = out.quad.get(k, 0.0) + a * b
        return out

    def clean(self, eps: float = 1e-12) -> "Poly":
        return Poly(round(self.c0, 10), {i: round(a, 10) for i, a in self.lin.items() if abs(a) > eps},
                    {k: round(b, 10) for k, b in self.quad.items() if abs(b) > eps})

    def evaluate(self, x) -> float:
        return self.c0 + sum(a * x[i] for i, a in self.lin.items()) + sum(b * x[i] * x[j] for (i, j), b in self.quad.items())


@dataclass
class QuboModel:
    encoding: str
    tasks: list[str]
    people: list[str]
    variables: list[tuple[str, str | None]]     # binary: (task, None); one-hot: (task, person)
    poly: Poly
    effort: Poly
    imbalance: Poly
    loads: dict[str, Poly]
    penalty_P: float
    mu: float
    unencoded: list[str] = field(default_factory=list)
    base_load: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.variables)

    def cost(self, x) -> float:
        return self.poly.evaluate(x)

    def decode(self, x) -> dict[str, str | None]:
        if self.encoding == "binary":
            return {t: self.people[int(x[i])] for i, (t, _) in enumerate(self.variables)}
        out: dict[str, str | None] = {t: None for t in self.tasks}
        for i, (t, r) in enumerate(self.variables):
            if x[i]:
                out[t] = r if out[t] is None else "CONFLICT"
        return out

    def encode(self, assignment: dict[str, str]) -> list[int]:
        if self.encoding == "binary":
            return [1 if assignment[t] == self.people[1] else 0 for t, _ in self.variables]
        return [1 if assignment.get(t) == r else 0 for t, r in self.variables]

    def coefficient_table(self) -> dict:
        p = self.poly.clean()
        return {"constant": p.c0, "linear": {f"x{i+1}": a for i, a in sorted(p.lin.items())},
                "quadratic": {f"x{i+1}x{j+1}": b for (i, j), b in sorted(p.quad.items())}}

    # ----------------------------------------------------------- Ising
    def to_ising(self) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
        p = self.poly.clean()
        h = {i: 0.0 for i in range(self.n)}
        J: dict[tuple[int, int], float] = {}
        const = p.c0
        for i, a in p.lin.items():
            h[i] -= a / 2
            const += a / 2
        for (i, j), b in p.quad.items():
            h[i] -= b / 4
            h[j] -= b / 4
            J[(i, j)] = b / 4
            const += b / 4
        return {i: round(v, 10) for i, v in h.items()}, {k: round(v, 10) for k, v in J.items()}, round(const, 10)

    def sparse_pauli(self):
        from qiskit.quantum_info import SparsePauliOp
        h, J, const = self.to_ising()
        n = self.n
        terms = []
        for i, v in h.items():
            if abs(v) > 1e-12:
                lab = ["I"] * n
                lab[n - 1 - i] = "Z"         # Qiskit labels are little-endian
                terms.append(("".join(lab), v))
        for (i, j), v in J.items():
            lab = ["I"] * n
            lab[n - 1 - i] = "Z"
            lab[n - 1 - j] = "Z"
            terms.append(("".join(lab), v))
        return SparsePauliOp.from_list(terms or [("I" * n, 0.0)]), const

    def cost_vector(self) -> np.ndarray:
        """C(x) for every basis index k, where bit i of k is variable i (qubit i)."""
        n = self.n
        idx = np.arange(2 ** n)
        X = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
        v = np.full(2 ** n, self.poly.c0)
        for i, a in self.poly.lin.items():
            v += a * X[:, i]
        for (i, j), bq in self.poly.quad.items():
            v += bq * X[:, i] * X[:, j]
        return v

    def quadratic_program(self):
        """Same model as a qiskit-optimization QuadraticProgram (used to cross-check the Ising map)."""
        from qiskit_optimization import QuadraticProgram
        qp = QuadraticProgram("assignment")
        for i in range(self.n):
            qp.binary_var(f"x{i+1}")
        p = self.poly.clean()
        qp.minimize(constant=p.c0, linear={f"x{i+1}": a for i, a in p.lin.items()},
                    quadratic={(f"x{i+1}", f"x{j+1}"): b for (i, j), b in p.quad.items()})
        return qp


def build_qubo(st: ProjectState, task_ids: list[str] | None = None, penalty_scale: float = 1.0,
               base_load: dict[str, float] | None = None) -> QuboModel:
    tasks = task_ids or [t.id for t in st.tasks]
    people = [p.id for p in st.people]
    base = {r: float((base_load or {}).get(r, 0.0)) for r in people}
    c = {(t, r): st.task(t).est_h[r] for t in tasks for r in people}
    P = config.BASE_PENALTY * penalty_scale
    mu = st.objective.mu
    if len(people) == 2:
        enc = "binary"
        variables = [(t, None) for t in tasks]
        A, B = people
        loads = {A: Poly(base[A]), B: Poly(base[B])}
        ind: dict[tuple[str, str], Poly] = {}
        for i, t in enumerate(tasks):
            loads[A] = loads[A] + Poly(c[(t, A)], {i: -c[(t, A)]})
            loads[B] = loads[B] + Poly.var(i, c[(t, B)])
            ind[(t, A)] = Poly(1.0, {i: -1.0})
            ind[(t, B)] = Poly.var(i)
        hard = Poly()
    else:
        enc = "one_hot"
        variables = [(t, r) for t in tasks for r in people]
        idx = {v: i for i, v in enumerate(variables)}
        loads = {r: Poly(base[r]) for r in people}
        ind = {}
        for (t, r), i in idx.items():
            loads[r] = loads[r] + Poly.var(i, c[(t, r)])
            ind[(t, r)] = Poly.var(i)
        hard = Poly()
        for t in tasks:
            s = Poly(1.0) - sum((Poly.var(idx[(t, r)]) for r in people), Poly())
            hard = hard + s.mul_affine(s).scale(config.ONE_HOT_LAMBDA)
    effort = sum((loads[r] for r in people), Poly()) + (-sum(base.values()))
    imb = Poly()
    for r, s in itertools.combinations(people, 2):
        d = loads[r] - loads[s]
        imb = imb + d.mul_affine(d)
    pen = Poly()
    for con in st.constraints:
        if con.type == "at_most_one" and con.who in people:
            ts = [t for t in con.tasks if t in tasks]
            for a, b in itertools.combinations(ts, 2):
                pen = pen + ind[(a, con.who)].mul_affine(ind[(b, con.who)]).scale(P)
    poly = (effort + imb.scale(mu) + pen + hard).clean()
    unencoded = [f"capacity:{p.id}<={p.avail_h}h" for p in st.people]
    return QuboModel(enc, tasks, people, variables, poly, effort.clean(), imb.clean(), loads, P, mu, unencoded, base)


def qubits_needed(st: ProjectState, n_tasks: int | None = None) -> int:
    n = n_tasks if n_tasks is not None else len(st.tasks)
    return n if len(st.people) == 2 else n * len(st.people)


def coupling_density(model: QuboModel) -> float:
    n = model.n
    return len(model.poly.quad) / (n * (n - 1) / 2) if n > 1 else 0.0


def decompose(st: ProjectState, budget: int = config.QUBIT_BUDGET) -> list[list[str]]:
    """Split tasks into consecutive chunks that each fit the qubit budget (Algorithm 7.2, simplified)."""
    per = 1 if len(st.people) == 2 else len(st.people)
    size = max(1, budget // per)
    ids = [t.id for t in st.tasks]
    return [ids[i:i + size] for i in range(0, len(ids), size)]
