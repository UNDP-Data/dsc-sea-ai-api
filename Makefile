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
