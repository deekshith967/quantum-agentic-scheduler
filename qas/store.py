"""Simple file-based persistence for sessions, decisions, audit events and mock systems.

The report proposes PostgreSQL + Redis; the prototype stores JSON files under
data/runtime so it runs with no external services (see IMPLEMENTATION_STATUS.md).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from . import config
from .schemas import AuditEvent

_lock = threading.Lock()


def _dir(name: str) -> Path:
    p = config.RUNTIME / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(kind: str, key: str, obj: Any) -> None:
    with _lock:
        (_dir(kind) / f"{key}.json").write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def load_json(kind: str, key: str) -> Any | None:
    p = _dir(kind) / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_json(kind: str) -> list[str]:
    return sorted(p.stem for p in _dir(kind).glob("*.json"))


def append_jsonl(name: str, obj: Any) -> None:
    with _lock:
        with open(config.RUNTIME / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")


def read_jsonl(name: str) -> list[dict]:
    p = config.RUNTIME / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(actor: str, action: str, ref: str = "", **detail) -> AuditEvent:
    ev = AuditEvent(actor=actor, action=action, ref=ref, detail=detail)
    append_jsonl("audit", ev.model_dump())
    return ev
