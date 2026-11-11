# Deployment

## 1. Local, no Docker (recommended for development)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Tesseract OCR binary (needed for the whiteboard photo):
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
#   Windows:       install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH
uvicorn qas.api:app --reload --port 8000
# open http://localhost:8000 and click "Load demo scenario", then "Run agent"
```

Optional:

```bash
pip install -r requirements-optional.txt   # faster-whisper for real audio, qiskit-ibm-runtime for hardware
```

## 2. Docker

```bash
docker compose up --build
# open http://localhost:8000
```

The image installs Tesseract and the Python requirements. The runtime folder (sessions,
audit log, bandit table, outcomes) is kept in the `runtime` volume. The Dockerfile was
written for this prototype but was not built in the development environment, so build
it once locally before relying on it.

## 3. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `QUANTUM_BACKEND` | `aer` | `aer` (ideal simulator), `aer_noisy` (illustrative depolarising model), `ibm_runtime` (real device, untested, needs `QiskitRuntimeService` credentials) |
| `QAS_OPTIMISER_SIM` | `numpy` | Statevector used while optimising QAOA angles: `numpy` (fast) or `qiskit` |
| `QAS_QAOA_RESTARTS` | `10` | COBYLA random restarts per depth |
| `QAS_RUNTIME` | `data/runtime` | Where sessions, audit events and learning state are written |
| `QAS_TODAY` | today | Fixes "today" for the recency filter (the demo uses 2026-11-11) |
| `QAS_LLM_BASE_URL`, `QAS_LLM_MODEL`, `QAS_LLM_API_KEY` | empty | Enable the optional OpenAI-compatible LLM planner (for example Ollama at `http://localhost:11434/v1`) |
| `QAS_WHISPER_MODEL` | `small` | faster-whisper model size when audio files are uploaded |

## 4. Production target from the report (not built)

The report (Chapter 14) proposes separate containers for the orchestrator, perception
worker, quantum worker and classical solver behind a Redis queue, PostgreSQL with
pgvector for state and retrieval, MinIO/S3 for media, Langfuse and MLflow for tracing
and model versions, and an AWS layout with ECS Fargate, RDS, ElastiCache and S3.
The prototype keeps the same module boundaries (`qas/perception`, `qas/solvers`,
`qas/rag.py`, `qas/store.py`), so each can be moved behind its own service later.
See `docs/figures/f14_1_compose.png` and `docs/figures/f14_2_cloud.png`.

## 5. Publishing to GitHub

```bash
cd quantum-agentic-scheduler
git init && git add . && git commit -m "Quantum-enhanced agentic scheduler prototype"
git branch -M main
git remote add origin https://github.com/<your-user>/quantum-agentic-scheduler.git
git push -u origin main
```

The CI workflow in `.github/workflows/ci.yml` runs the tests and a 6-case benchmark on every push.
