FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY qas ./qas
COPY frontend ./frontend
COPY data ./data
COPY scripts ./scripts
ENV QAS_RUNTIME=/app/data/runtime QUANTUM_BACKEND=aer QAS_OPTIMISER_SIM=numpy QAS_TODAY=2026-11-11
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "qas.api:app", "--host", "0.0.0.0", "--port", "8000"]
