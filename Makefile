install:
	pip install --upgrade pip && pip install -r requirements_dev.txt
lint:
	pylint main.py src
format:
	isort . --profile black --multi-line 3 && black .
test:
	python -m pytest tests
loadtest:
	python -m locust --config locust.conf --host $${HOST:="http://127.0.0.1:8000"}
run:
	uvicorn main:app --reload
run-tester:
	uvicorn frontend.kg_tester_app:app --reload --host 127.0.0.1 --port 8010
run-sgp-tester:
	uvicorn frontend.sgp_ai_tester_app:app --reload --host $${SGP_TESTER_HOST:-127.0.0.1} --port $${SGP_TESTER_PORT:-8015}
run-sgp-local:
	$${PYTHON:-.venv/bin/python} scripts/run_sgp_ai_local.py --reload
