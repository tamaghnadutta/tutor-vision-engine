.PHONY: setup dev test eval demo lint format clean docker-build docker-run

# Setup and Installation
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	mkdir -p data/eval_results
	mkdir -p logs
	@echo "Setup complete! Copy .env.sample to .env and configure your API keys."

# Development
dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Evaluation
eval:
	python scripts/run_eval.py

# Demo
demo:
	python scripts/demo.py

# Create Dataset from provided sample images
create-dataset:
	python scripts/create_dataset.py

# Code Quality
lint:
	ruff check src tests scripts
	mypy src

format:
	ruff format src tests scripts
	isort src tests scripts

# Performance Testing
load-test:
	python scripts/load_test.py

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Docker
docker-build:
	docker build -t error-detection-api .

docker-run:
	docker run -p 8000:8000 --env-file .env error-detection-api

