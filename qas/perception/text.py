"""Layer 2, text and transcript understanding.

This is a deterministic, rule-based extractor that stands in for the SLM extractor
described in the report (Section 6.1). It covers the phrasings used in the demo and
the benchmark generator. Confidence values are heuristic: they reflect how
ambiguous the mapping from phrase to formal constraint is, multiplied by the
speech-to-text confidence when the text came from audio.
"""
from __future__ import annotations

import difflib
import re

from ..schemas import Extracted

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def _names_in(text: str, roster: list[str]) -> list[str]:
    return [n for n in roster if re.search(rf"\b{re.escape(n)}\b", text, re.I)]


def match_task(phrase: str, titles: list[str]) -> tuple[str | None, float]:
    """Fuzzy-match a spoken phrase to a known task title. Returns (title, ratio)."""
    p = re.sub(r"\b(the|a|an)\b", " ", phrase.lower())
    p = re.sub(r"\s+", " ", p).strip()
    best, score = None, 0.0
    for t in titles:
        r = difflib.SequenceMatcher(None, p, t.lower()).ratio()
        if t.lower() in p or p in t.lower():
            r = max(r, 0.95)
        if r > score:
            best, score = t, r
    return (best, score) if score >= 0.6 else (None, score)


def extract_brief(text: str, roster: list[str]) -> list[Extracted]:
    out: list[Extracted] = []
    names = _names_in(text, roster)
    if names:
        out.append(Extracted(kind="person", value={"ids": names}, src="brief", conf=0.97, text=text))
    for d in DAYS:
        if re.search(rf"\b{d}\b", text, re.I):
            out.append(Extracted(kind="date", value={"day_name": d.capitalize()}, src="brief", conf=0.95, text=d))
            break
    if re.search(r"\b(fair|balance[d]?|even(ly)?)\b", text, re.I):
        out.append(Extracted(kind="objective", value={"minimise": ["effort", "imbalance"], "mu": 0.1}, src="brief", conf=0.90,
                             text="keep the load fair"))
    else:
        out.append(Extracted(kind="objective", value={"minimise": ["effort"], "mu": 0.0}, src="brief", conf=0.80, text="(no fairness phrase)"))
    return out


def extract_transcript(text: str, roster: list[str], task_titles: list[str], stt_conf: float = 1.0, src: str = "voice#1") -> list[Extracted]:
    out: list[Extracted] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        names = _names_in(sent, roster)
        who = names[0] if names else None
        if not who:
            continue
        # availability phrases
        if re.search(r"half[- ]day", sent, re.I):
            out.append(Extracted(kind="availability", value={"who": who, "hours": 4.0}, src=src, conf=round(0.88 * stt_conf, 3), text=sent))
        elif m := re.search(r"(?:only|just)?\s*(\d+|one|two|three|four|five|six|seven|eight)\s*hours?", sent, re.I):
            h = m.group(1).lower()
            out.append(Extracted(kind="availability", value={"who": who, "hours": float(_NUM.get(h, h))}, src=src,
                                 conf=round(0.85 * stt_conf, 3), text=sent))
        elif re.search(r"\b(off|on leave|out|sick)\b", sent, re.I) and not re.search(r"half", sent, re.I):
            out.append(Extracted(kind="availability", value={"who": who, "hours": 0.0}, src=src, conf=round(0.86 * stt_conf, 3), text=sent))
        # "at most one of the A and B"
        if m := re.search(r"at most one of (.+?)(?:[.!?]|$)", sent, re.I):
            parts = re.split(r",\s*|\s+and\s+|\s+or\s+", m.group(1))
            matched = [match_task(pp, task_titles) for pp in parts if pp.strip()]
            titles = [t for t, s in matched if t]
            if len(titles) >= 2:
                q = min(s for t, s in matched if t)
                conf = 0.81 * min(1.0, q / 0.95) * stt_conf
                out.append(Extracted(kind="constraint", value={"type": "at_most_one", "who": who, "titles": titles}, src=src,
                                     conf=round(conf, 3), text=sent))
    return out
