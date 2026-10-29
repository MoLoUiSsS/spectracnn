.PHONY: install data train train-fast serve test lint docker compose clean

install:
	uv venv --python 3.12
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install -e ".[dev,mlops]"

train:
	uv run spectracnn-train

train-fast:
	uv run spectracnn-train --samples-per-class 300 --epochs 3

serve:
	uv run spectracnn-serve

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

docker:
	docker build -t spectracnn:latest .

compose:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
