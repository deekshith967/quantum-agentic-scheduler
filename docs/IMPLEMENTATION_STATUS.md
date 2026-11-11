# Implementation status

This file separates what the prototype actually does from what the report proposes.
Labels:

* **Implemented**: working code, covered by tests or by the benchmark.
* **Simplified**: working code, but simpler than the report's design (the difference is stated).
* **Simulated**: runs, but on a stand-in for the real thing (named).
* **Future work**: designed in the report, not built here.

| Layer | Report design | Status | What the prototype does | Where |
|---|---|---|---|---|
| L1 Multimodal interaction | React UI, uploads, object storage | **Simplified** | Single-page vanilla JS UI (no build step) with text, image and audio/transcript upload; files stored under `data/runtime/uploads` | `frontend/`, `qas/api.py` |
| L2 Vision | PaddleOCR or VLM, column clustering | **Implemented** (with Tesseract) | Colour segmentation of notes, Tesseract OCR per note with confidence, nearest-header column assignment, Done column excluded | `qas/perception/vision.py` |
| L2 Audio | faster-whisper | **Simplified** | `.txt` transcripts are the default path; real audio is transcribed only if `faster-whisper` is installed (optional dependency, not exercised in tests because model weights could not be downloaded in the build environment) | `qas/perception/audio.py` |
| L2 Text / extraction SLM | Schema-constrained SLM | **Simulated** | Deterministic rule-based extractor covering the demo and benchmark phrasings. Confidence values are heuristic, not model probabilities | `qas/perception/text.py` |
| L3 Project state | Pydantic, PostgreSQL, Redis | **Simplified** | Pydantic `ProjectState`, merge rule of Algorithm 5.1; persisted as JSON files instead of PostgreSQL/Redis | `qas/state.py`, `qas/schemas.py`, `qas/store.py` |
| L4 LLM/SLM reasoning | Reference LLM + SLM | **Simulated** (LLM optional) | Default planner is rule-based and follows the prompt contract; an OpenAI-compatible LLM planner (e.g. Ollama) is wired in via env vars but **was not tested** here. Explanations are template-based from the decision object | `qas/llm.py` |
| L5 RAG and memory | pgvector + BM25 + bge-small, RRF | **Implemented** (dense part simplified) | BM25 implemented from scratch; dense retrieval uses TF-IDF + truncated SVD (LSA) instead of a neural embedding model; RRF fusion; similarity-weighted effort estimates with recency and skill filters; validated outcomes feed back into history | `qas/rag.py`, `data/corpus/`, `data/history/` |
| L6 Tools | Gateway, mocks, approval-gated writes | **Implemented** (external systems mocked) | Typed tool registry, audit log, write tools refuse without a valid approval token; Jira and calendar are JSON mocks | `qas/tools.py` |
| L7 ReAct + multi-hop | LangGraph, hop limit 8 | **Implemented** (no LangGraph) | Bounded loop with hop limit, repeated-call and no-new-information guards; reproduces the report's 6-step trace | `qas/react.py` |
| L8 RL strategy | Contextual bandit (UCB) | **Implemented** | UCB selection per context bucket, constant step-size update from the reward of Equation 11.1, persisted Q-table. The starting Q-table for the scenario bucket is the report's **illustrative prior**, not learned data | `qas/bandit.py` |
| L9 Formulation | QUBO, Ising, decomposition | **Implemented** (decomposition simplified) | Binary encoding for two people, one-hot for more; effort + mu * pairwise imbalance + rule penalties; Ising conversion cross-checked with qiskit-optimization; decomposition into consecutive chunks with carried load | `qas/qubo.py` |
| L10 QAOA | Qiskit, Aer, COBYLA | **Simulated** (quantum on a classical simulator) | `QAOAAnsatz` circuits sampled with Qiskit Aer (ideal or illustrative depolarising noise). Angle optimisation uses an exact statevector, either Qiskit's or an equivalent NumPy one (default, about 20x faster, equality tested to 1e-9). **No quantum hardware was used.** The IBM Runtime path exists in code but is untested | `qas/solvers/qaoa.py` |
| L10 Classical | CP-SAT, brute force, time slots | **Implemented** | Exact brute force, CP-SAT assignment with hard capacity and rule constraints, CP-SAT 30-minute timetable with precedence | `qas/solvers/classical.py` |
| L11 Decision | Decode, verify, accept within tau | **Implemented** | Independent feasibility check (catches capacity, which the QUBO does not encode), most-frequent-feasible-sample rule, tau = 5 % acceptance, classical fallback, HITL triggers | `qas/decision.py` |
| L12 HITL | Review console, state machine | **Implemented** | Standard vs flagged review, flags must be acknowledged, edits re-verified, rejection with comment, single-decision approval tokens | `qas/hitl.py`, UI section 6 |
| L13 Ground truth and evaluation | 3 levels, 30-case benchmark | **Implemented** (synthetic data) | Exact optimum, PM label and actual hours recorded per decision; 30 synthetic cases generated, rendered and run end to end; metrics in `benchmarks/RESULTS.md` | `qas/evaluation.py`, `scripts/run_benchmark.py` |
| L14 Feedback | RAG, Q-table, SFT/DPO, QLoRA | **Partly implemented** | Promotion rule, Q-table update, actual hours appended to effort history, SFT/DPO JSONL export. **QLoRA/DPO training is future work**; `scripts/finetune_slm.py` is an unrun template | `qas/feedback.py` |
| L15 Deployment | Compose with many services, AWS design, CI/CD | **Simplified** | Single-container Dockerfile and Compose file, GitHub Actions CI (tests + small benchmark gate). The Docker image was not built in the development environment (no Docker daemon available). The multi-service and AWS designs remain future work | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `docs/DEPLOYMENT.md` |

## Things that are explicitly not claimed

* No quantum advantage. On every case in the benchmark CP-SAT is faster and exact.
* No result from real quantum hardware.
* No trained or fine-tuned model.
* No evaluation of a real LLM planner.
* No real user data: all people, tasks, calendars and histories are synthetic.

## Numbers that differ from the report, and why

* **Bandit update after the scenario.** The report uses the Qiskit-statevector optimiser latency (about 4.8 s per optimisation run), giving r = 1.52 and Q = 1.264; `tests/test_governance.py` checks exactly this. The prototype's default NumPy optimiser takes about 0.15 s per run, so the live demo gives r of about 1.98 and Q of about 1.36. Both follow Equation 11.1; only the measured latency differs.
* **QAOA expectation values** depend on the random restarts. The test suite fixes the report's p = 1 angles and checks F = 10.19 and P(010) = 0.385.
