"""FastAPI backend (Layer 1 endpoints plus review, feedback and inspection routes).

Run:  uvicorn qas.api:app --reload      then open http://localhost:8000
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, config, store
from .perception.audio import STTUnavailable
from .pipeline import Orchestrator

warnings.filterwarnings("ignore")
app = FastAPI(title="Quantum-Enhanced Agentic Scheduler", version=__version__,
              description="Prototype of the 15-layer agentic architecture with QAOA (Qiskit Aer) task allocation.")
ORCH: Orchestrator | None = None


def orch() -> Orchestrator:
    global ORCH
    if ORCH is None:
        ORCH = Orchestrator(today=os.environ.get("QAS_TODAY"))
    return ORCH


def _get(sid: str) -> dict:
    try:
        return orch().get(sid)
    except KeyError:
        raise HTTPException(404, f"unknown session {sid}")


class PlanReq(BaseModel):
    mode: str = "quantum_eval"
    restarts: Optional[int] = None
    noisy: bool = False
    force_arm: Optional[str] = None
    optimiser_sim: Optional[str] = None


class ReviewReq(BaseModel):
    verdict: str
    edits: Optional[dict[str, str]] = None
    comment: str = ""
    acknowledged: list[str] = []


class OutcomeReq(BaseModel):
    outcomes: list[dict[str, Any]]


class SearchReq(BaseModel):
    query: str
    k: int = 3


@app.get("/health")
def health():
    import qiskit, qiskit_aer
    return {"ok": True, "version": __version__, "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
            "planner": orch().planner.name, "quantum_backend": config.QUANTUM_BACKEND, "optimiser_sim": config.OPTIMISER_SIM,
            "note": "Quantum circuits are simulated on a classical CPU with Qiskit Aer unless QUANTUM_BACKEND=ibm_runtime."}


@app.post("/v1/sessions")
def new_session():
    return orch().new_session()


@app.get("/v1/sessions/{sid}")
def get_session(sid: str):
    return _get(sid)


@app.post("/v1/sessions/{sid}/demo")
def load_demo(sid: str):
    _get(sid)
    return orch().load_demo(sid)


@app.post("/v1/sessions/{sid}/inputs")
async def add_input(sid: str, kind: str = Form(...), text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    _get(sid)
    if kind == "text":
        if not text:
            raise HTTPException(400, "text required")
        return orch().add_input(sid, "text", text=text)
    if kind not in ("image", "audio") or file is None:
        raise HTTPException(400, "kind must be text, image or audio (with a file)")
    up = config.RUNTIME / "uploads" / sid
    up.mkdir(parents=True, exist_ok=True)
    dest = up / f"{kind}-{uuid.uuid4().hex[:6]}{Path(file.filename or '').suffix or '.bin'}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return orch().add_input(sid, kind, path=str(dest))


@app.post("/v1/sessions/{sid}/perceive")
def perceive(sid: str):
    _get(sid)
    try:
        return orch().perceive(sid)
    except STTUnavailable as exc:
        raise HTTPException(422, str(exc))


@app.post("/v1/sessions/{sid}/plan")
def plan(sid: str, req: PlanReq):
    _get(sid)
    return orch().plan(sid, mode=req.mode, restarts=req.restarts, noisy=req.noisy, force_arm=req.force_arm, optimiser_sim=req.optimiser_sim)


@app.post("/v1/sessions/{sid}/review")
def review(sid: str, req: ReviewReq):
    _get(sid)
    try:
        return orch().review_action(sid, req.verdict, req.edits, req.comment, req.acknowledged)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/v1/sessions/{sid}/outcomes")
def outcomes(sid: str, req: OutcomeReq):
    _get(sid)
    try:
        return orch().record_outcomes(sid, req.outcomes)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/v1/sessions/{sid}/image")
def session_image(sid: str):
    p = _get(sid)["inputs"].get("image_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "no image")
    return FileResponse(p)


@app.get("/v1/decisions/{did}")
def get_decision(did: str):
    d = store.load_json("decisions", did)
    if d is None:
        raise HTTPException(404, "unknown decision")
    return d


@app.get("/v1/bandit")
def bandit():
    return orch().bandit.table


@app.get("/v1/tools")
def tools():
    return orch().gw.schemas()


@app.post("/v1/rag/search")
def rag_search(req: SearchReq):
    return {"passages": orch().kb.search_policy(req.query, req.k)}


@app.get("/v1/audit")
def audit(limit: int = 50):
    return store.read_jsonl("audit")[-limit:]


@app.get("/v1/mock/{system}")
def mock(system: str):
    name = {"jira": "mock_jira", "calendar": "mock_calendar_events"}.get(system)
    if not name:
        raise HTTPException(404, "jira or calendar")
    return store.read_jsonl(name)


FRONTEND = config.ROOT / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
