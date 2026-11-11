"""Central configuration. Values can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QAS_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
ORG_DIR = DATA / "org"
HISTORY_CSV = DATA / "history" / "effort_history.csv"   # synthetic seed history (read-only)
DEMO_DIR = DATA / "demo"
RUNTIME = Path(os.environ.get("QAS_RUNTIME", DATA / "runtime"))
OUTCOMES_CSV = RUNTIME / "effort_outcomes.csv"            # validated actual hours appended by Layer 14

# Layer 5 / 7
MAX_HOPS = 8
EFFORT_TOP_K = 3
EFFORT_RECENCY_DAYS = 365
PLANNING_UNIT_H = 1.0          # durations are rounded to this unit for the QUBO

# Layer 8
UCB_KAPPA = 0.5
BANDIT_ALPHA = 0.2

# Layer 9
QUBIT_BUDGET = 12
DEFAULT_MU = 0.1
BASE_PENALTY = 10.0            # P = BASE_PENALTY * penalty_scale
ONE_HOT_LAMBDA = 20.0

# Layer 10
SHOTS = 1024
SEED = 42
QAOA_RESTARTS = int(os.environ.get("QAS_QAOA_RESTARTS", "10"))
QAOA_MAXITER = 400
QUANTUM_BACKEND = os.environ.get("QUANTUM_BACKEND", "aer")   # aer | aer_noisy | ibm_runtime
OPTIMISER_SIM = os.environ.get("QAS_OPTIMISER_SIM", "numpy")   # numpy (fast, validated) | qiskit

# Layer 11 / 12
ACCEPT_TAU = 0.05
CONF_THRESHOLD = 0.85

# Optional LLM (OpenAI-compatible endpoint such as Ollama). Empty -> rule-based planner.
LLM_BASE_URL = os.environ.get("QAS_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("QAS_LLM_MODEL", "")
LLM_API_KEY = os.environ.get("QAS_LLM_API_KEY", "")

SKILL_KEYWORDS = {
    "backend": ["api", "integration", "backend", "database", "endpoint", "payment", "service"],
    "qa": ["test", "testing", "qa", "regression", "ui test"],
    "docs": ["notes", "documentation", "docs", "readme", "changelog", "release notes"],
    "frontend": ["frontend", "css", "layout", "component"],
}

RUNTIME.mkdir(parents=True, exist_ok=True)
