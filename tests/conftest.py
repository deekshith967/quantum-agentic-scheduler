import os
import sys
import tempfile
from pathlib import Path

# Isolate all runtime writes (sessions, audit log, bandit table, outcomes) in a temp dir.
_RT = tempfile.mkdtemp(prefix="qas_test_")
os.environ["QAS_RUNTIME"] = _RT
os.environ.setdefault("QAS_TODAY", "2026-11-11")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from qas.schemas import Constraint, Person, ProjectState, Task  # noqa: E402


@pytest.fixture
def scenario() -> ProjectState:
    """The report's running example after Tier B (Table 6.3)."""
    return ProjectState(
        session_id="test", date="2026-11-12",
        tasks=[Task(id="T1", title="Payment API integration", skill="backend", est_h={"Anika": 3, "Ravi": 5}),
               Task(id="T2", title="UI testing", skill="qa", est_h={"Anika": 4, "Ravi": 2}),
               Task(id="T3", title="Release notes", skill="docs", est_h={"Anika": 2, "Ravi": 1})],
        people=[Person(id="Anika", avail_h=8), Person(id="Ravi", avail_h=4)],
        constraints=[Constraint(type="at_most_one", who="Ravi", tasks=["T2", "T3"], conf=0.81), Constraint(type="capacity")],
        dependencies=[["T1", "T3"]])
