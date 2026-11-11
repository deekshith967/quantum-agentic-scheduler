.PHONY: install run test demo bench docker
install:   ; pip install -r requirements.txt
run:       ; uvicorn qas.api:app --reload --port 8000
test:      ; pytest -q
demo:      ; python scripts/run_demo.py
bench:     ; python scripts/run_benchmark.py --cases 30 --restarts 6
docker:    ; docker compose up --build
