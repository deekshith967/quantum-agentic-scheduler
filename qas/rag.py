"""Layer 5: knowledge base with hybrid retrieval and similarity-weighted effort estimates.

* Lexical retrieval: Okapi BM25 implemented here (k1 = 1.5, b = 0.75).
* Dense retrieval: TF-IDF followed by truncated SVD (latent semantic analysis) and
  cosine similarity. This is a lightweight offline substitute for the bge-small
  embedding model proposed in the report; swap `DenseIndex` to use real embeddings.
* Fusion: reciprocal rank fusion with k = 60 (Cormack et al., 2009).
* Effort estimate: similarity-weighted mean of the top-k past tasks done by the same
  person with a matching skill in the last 12 months (Calculation 6.1).
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

_tok = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _tok.findall(s.lower())


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [tokenize(d.text) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(t for toks in self.toks for t in set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            for t in q:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


class DenseIndex:
    def __init__(self, docs: list[Doc], dim: int = 48, seed: int = 0):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform([d.text for d in docs])
        k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=seed)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def embed(self, text: str) -> np.ndarray:
        z = self.svd.transform(self.vec.transform([text]))[0]
        return z / (np.linalg.norm(z) + 1e-12)

    def scores(self, query: str) -> np.ndarray:
        return self.Z @ self.embed(query)


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking, 1):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + r)
    return out


def load_corpus() -> list[Doc]:
    docs: list[Doc] = []
    for p in sorted(config.CORPUS_DIR.glob("*.md")):
        title, buf = None, []
        for line in p.read_text(encoding="utf-8").splitlines() + ["## __end__"]:
            if line.startswith("## "):
                if title and buf:
                    docs.append(Doc(f"{p.stem}#{len(docs)}", f"{title}. " + " ".join(buf).strip(), {"file": p.name, "section": title}))
                title, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line)
    return docs


def load_history(path=None) -> list[dict]:
    rows: list[dict] = []
    paths = [path] if path else [config.HISTORY_CSV, config.RUNTIME / "effort_outcomes.csv"]
    for pth in paths:
        if pth and pth.exists():
            with open(pth, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    for r in rows:
        r["actual_h"] = float(r["actual_h"])
    return rows


class KnowledgeBase:
    def __init__(self, today: str | None = None):
        self.policy_docs = load_corpus()
        self.policy_bm25 = BM25(self.policy_docs)
        self.policy_dense = DenseIndex(self.policy_docs)
        self.today = dt.date.fromisoformat(today) if today else dt.date.today()
        self.reload_history()

    def reload_history(self):
        self.history = load_history()
        self.hist_docs = [Doc(r["id"], f"{r['title']} {r['skill']}", r) for r in self.history]
        self.hist_dense = DenseIndex(self.hist_docs, dim=24)
        self.hist_bm25 = BM25(self.hist_docs)

    def search_policy(self, query: str, k: int = 3) -> list[dict]:
        b = self.policy_bm25.scores(query)
        d = self.policy_dense.scores(query)
        fused = rrf([list(np.argsort(-b)), list(np.argsort(-d))])
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"doc_id": self.policy_docs[i].doc_id, "section": self.policy_docs[i].meta["section"],
                 "text": self.policy_docs[i].text, "rrf": round(fused[i], 5), "bm25": round(float(b[i]), 3),
                 "dense": round(float(d[i]), 3)} for i in top]

    def estimate_effort(self, title: str, skill: str | None, person: str, k: int = config.EFFORT_TOP_K) -> dict:
        query = f"{title} {skill or ''}"
        sims = self.hist_dense.scores(query)
        cutoff = self.today - dt.timedelta(days=config.EFFORT_RECENCY_DAYS)
        cands = []
        for i, r in enumerate(self.history):
            if r["person"] != person or dt.date.fromisoformat(r["date"]) < cutoff:
                continue
            if skill and r["skill"] != skill:
                continue
            cands.append((float(sims[i]), r))
        cands.sort(key=lambda x: -x[0])
        top = [(max(s, 1e-3), r) for s, r in cands[:k]]
        if not top:
            return {"hours": None, "raw": None, "evidence": [], "flag": "no history"}
        raw = sum(s * r["actual_h"] for s, r in top) / sum(s for s, _ in top)
        unit = config.PLANNING_UNIT_H
        hours = max(unit, round(raw / unit) * unit)
        ev = [{"doc_id": r["id"], "title": r["title"], "similarity": round(s, 3), "actual_h": r["actual_h"]} for s, r in top]
        return {"hours": hours, "raw": round(raw, 3), "evidence": ev, "flag": "few examples" if len(top) < 2 else None}
