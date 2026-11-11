"""Layer 10, quantum: QAOA with Qiskit, sampled on Qiskit Aer.

Honesty note: with backend "aer" or "aer_noisy" the quantum circuit is SIMULATED on a
classical CPU. No quantum hardware is used unless backend "ibm_runtime" is selected
and configured (that path is provided but untested in this prototype).

Parameter optimisation evaluates <H_C> exactly from the statevector:
* optimiser_sim="qiskit": qiskit.quantum_info.Statevector of the QAOAAnsatz circuit.
* optimiser_sim="numpy": an equivalent NumPy statevector of the same ansatz (much faster;
  tests/test_qaoa.py checks it agrees with Qiskit to 1e-9). Used by the benchmark.
The final circuit is always built with QAOAAnsatz and sampled on Aer.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .. import config
from ..qubo import QuboModel


def _numpy_state(costs: np.ndarray, gammas, betas, n: int) -> np.ndarray:
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * costs)
        c, s = np.cos(b), -1j * np.sin(b)
        t = psi.reshape([2] * n)  # axis k corresponds to qubit n-1-k
        for ax in range(n):
            a0 = np.take(t, 0, axis=ax)
            a1 = np.take(t, 1, axis=ax)
            t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = t.reshape(-1)
    return psi


def interp(v: np.ndarray, p: int) -> np.ndarray:
    """INTERP initialisation from depth p to p+1 (Zhou et al., 2020)."""
    out = np.zeros(p + 1)
    for i in range(p + 1):
        out[i] = ((i / p) * v[i - 1] if i >= 1 else 0) + (((p - i) / p) * v[i] if i < p else 0)
    return out


class QAOASolver:
    def __init__(self, model: QuboModel, backend: str = config.QUANTUM_BACKEND, optimiser_sim: str = config.OPTIMISER_SIM,
                 shots: int = config.SHOTS, seed: int = config.SEED):
        self.m, self.backend, self.sim_mode, self.shots, self.seed = model, backend, optimiser_sim, shots, seed
        self.H, self.const = model.sparse_pauli()
        self.costs = model.cost_vector()
        self._ansatz = {}

    def ansatz(self, p: int):
        if p not in self._ansatz:
            from qiskit.circuit.library import QAOAAnsatz
            a = QAOAAnsatz(self.H, reps=p)
            gam = [t for t in a.parameters if "γ" in t.name]
            bet = [t for t in a.parameters if "β" in t.name]
            self._ansatz[p] = (a, gam, bet)
        return self._ansatz[p]

    def bind(self, p: int, v):
        a, gam, bet = self.ansatz(p)
        return a.assign_parameters({**dict(zip(gam, v[:p])), **dict(zip(bet, v[p:]))})

    def probabilities(self, p: int, v) -> np.ndarray:
        if self.sim_mode == "numpy":
            return np.abs(_numpy_state(self.costs, v[:p], v[p:], self.m.n)) ** 2
        from qiskit.quantum_info import Statevector
        return np.abs(Statevector(self.bind(p, v)).data) ** 2

    def expectation(self, p: int, v) -> float:
        if self.sim_mode == "numpy":
            return float(self.probabilities(p, v) @ self.costs)
        from qiskit.quantum_info import Statevector
        return float(np.real(Statevector(self.bind(p, v)).expectation_value(self.H))) + self.const

    def optimise(self, p: int, restarts: int = config.QAOA_RESTARTS, warm: np.ndarray | None = None) -> dict:
        rng = np.random.default_rng(self.seed + p)
        best, nfev, t0 = None, 0, time.perf_counter()
        starts = [warm] if warm is not None else []
        starts += [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)]) for _ in range(restarts)]
        for x0 in starts:
            r = minimize(lambda v: self.expectation(p, v), x0, method="COBYLA", options={"maxiter": config.QAOA_MAXITER})
            nfev += r.nfev
            if best is None or r.fun < best.fun:
                best = r
        return {"x": best.x, "F": float(best.fun), "nfev": int(nfev), "time_s": time.perf_counter() - t0, "starts": len(starts)}

    def sample(self, p: int, v, noisy: bool = False) -> dict[str, int]:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        qc = self.bind(p, v)
        qc.measure_all()
        if self.backend == "ibm_runtime":  # pragma: no cover - needs IBM account
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            svc = QiskitRuntimeService()
            dev = svc.least_busy(operational=True, simulator=False)
            tq = transpile(qc, dev, optimization_level=1)
            res = SamplerV2(mode=dev).run([tq], shots=self.shots).result()
            raw = res[0].data.meas.get_counts()
        elif noisy or self.backend == "aer_noisy":
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            nm = NoiseModel()  # illustrative depolarising model, NOT a real device calibration
            nm.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rz", "sx", "x"])
            nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
            tq = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=7)
            raw = AerSimulator(noise_model=nm, seed_simulator=self.seed).run(tq, shots=self.shots).result().get_counts()
        else:
            sim = AerSimulator(seed_simulator=self.seed)
            raw = sim.run(transpile(qc, sim, seed_transpiler=7), shots=self.shots).result().get_counts()
        return {k[::-1]: int(c) for k, c in raw.items()}   # Qiskit keys are q_{n-1}..q_0 -> reverse to x1..xn

    def circuit_stats(self, p: int, v) -> dict:
        from qiskit import transpile
        qc = self.bind(p, v)
        qc.measure_all()
        tq = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=7)
        ops = tq.count_ops()
        return {"depth": tq.depth(), "cx": int(ops.get("cx", 0)), "rz": int(ops.get("rz", 0)), "sx": int(ops.get("sx", 0))}

    def solve(self, p: int, restarts: int = config.QAOA_RESTARTS, noisy_too: bool = False) -> dict:
        opt = self.optimise(p, restarts)
        v = opt["x"]
        probs = self.probabilities(p, v)
        n = self.m.n
        prob_by_bits = {format(k, f"0{n}b")[::-1]: float(pk) for k, pk in enumerate(probs) if pk > 1e-6}
        t0 = time.perf_counter()
        counts = self.sample(p, v)
        out = {"p": p, "F": round(opt["F"], 4), "gammas": [round(float(a), 4) for a in v[:p]], "betas": [round(float(a), 4) for a in v[p:]],
               "probs": {k: round(pv, 4) for k, pv in sorted(prob_by_bits.items())}, "counts": dict(sorted(counts.items())),
               "shots": self.shots, "backend": self.backend if self.backend != "aer" else "aer_simulator", "optimiser_sim": self.sim_mode,
               "nfev": opt["nfev"], "starts": opt["starts"], "time_opt_s": round(opt["time_s"], 3),
               "time_sample_s": round(time.perf_counter() - t0, 3), "circuit": self.circuit_stats(p, v),
               "executed_on": "classical CPU simulation (Qiskit Aer)" if self.backend != "ibm_runtime" else "IBM Quantum hardware"}
        if noisy_too and self.backend == "aer":
            out["counts_noisy"] = dict(sorted(self.sample(p, v, noisy=True).items()))
        return out
