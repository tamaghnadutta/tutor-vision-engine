.PHONY: setup dev test eval eval-detailed eval-assignment demo demo-ocr-llm demo-vlm demo-hybrid demo-compare test-ocr-llm test-vlm-direct test-hybrid create-dataset analytics analytics-charts analytics-report analytics-export analytics-clean monitoring-up monitoring-down monitoring-logs lint format load-test clean docker-build docker-run

# Help command
help:
	@echo "Available commands:"
	@echo "  setup           - Install dependencies and initialize project"
	@echo "  dev             - Start development server"
	@echo "  test            - Run test suite"
	@echo ""
	@echo "Evaluation (Assignment Requirements):"
	@echo "  eval            - Run comprehensive evaluation (all 3 approaches)"
	@echo "  eval-detailed   - Run evaluation with detailed output"
	@echo "  eval-assignment - Assignment-compliant baseline vs improvement"
	@echo ""
	@echo "Demo (Assignment Requirements):"
	@echo "  demo            - Demo all three approaches"
	@echo "  demo-ocr-llm    - Demo OCR→LLM approach only"
	@echo "  demo-vlm        - Demo Direct VLM approach only"
	@echo "  demo-hybrid     - Demo Hybrid approach only"
	@echo "  demo-compare    - Quick comparison of all approaches"
	@echo ""
	@echo "Testing with Environment Variables:"
	@echo "  test-ocr-llm    - Test OCR→LLM with env var"
	@echo "  test-vlm-direct - Test Direct VLM with env var"
	@echo "  test-hybrid     - Test Hybrid with env var"
	@echo "  test-monitoring - Generate API traffic for monitoring dashboard"
	@echo "  test-complete-areas - Test complete solution areas (use with ERROR_DETECTION_APPROACH)"
	@echo "  load-test       - Run performance load test"
	@echo "  load-test-ocr   - Load test OCR→LLM approach"
	@echo "  load-test-vlm   - Load test Direct VLM approach"
	@echo "  load-test-hybrid- Load test Hybrid approach"
	@echo ""
	@echo "Analytics:"
	@echo "  analytics       - Show analytics summary (last 7 days)"
	@echo "  analytics-charts- Generate performance visualization charts"
	@echo "  analytics-report- Generate detailed analytics report"
	@echo "  analytics-export- Export analytics data to CSV"
	@echo "  analytics-clean - Clean old cache and data"
	@echo ""
	@echo "Monitoring:"
	@echo "  monitoring-up   - Start Prometheus & Grafana monitoring stack"
	@echo "  monitoring-down - Stop monitoring stack"
	@echo "  monitoring-logs - View monitoring logs"
	@echo ""
	@echo "Other:"
	@echo "  create-dataset  - Generate dataset from sample images"
	@echo "  lint            - Check code quality"
	@echo "  load-test       - Run performance tests"
	@echo "  clean           - Clean build artifacts"

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

# Evaluation (comprehensive - all three approaches)
eval:
	python scripts/run_eval.py

# Evaluation with specific parameters
eval-detailed:
	python scripts/run_eval.py --output ./data/eval_results/comprehensive_evaluation.json --export-cases ./data/eval_results/detailed_cases.json --max-concurrent 2

# Demo (all approaches)
demo:
	python scripts/demo.py --approach all

# Demo specific approaches
demo-ocr-llm:
	python scripts/demo.py --approach ocr_llm

demo-vlm:
	python scripts/demo.py --approach vlm_direct

demo-hybrid:
	python scripts/demo.py --approach hybrid

# Quick approach comparison (just one sample)
demo-compare:
	python scripts/demo.py --approach all

# Test specific approaches with environment variables
test-ocr-llm:
	ERROR_DETECTION_APPROACH=ocr_llm python scripts/demo.py --approach ocr_llm

test-vlm-direct:
	ERROR_DETECTION_APPROACH=vlm_direct python scripts/demo.py --approach vlm_direct

test-hybrid:
	ERROR_DETECTION_APPROACH=hybrid python scripts/demo.py --approach hybrid

# Test API with monitoring dashboard
test-monitoring:
	python test_api_with_monitoring.py

# Test complete solution areas (use with ERROR_DETECTION_APPROACH env var)
test-complete-areas:
	python test_complete_solution_areas.py

# Evaluation for assignment submission (baseline vs improvement)
eval-assignment:
	@echo "Running assignment-compliant evaluation: OCR→LLM (baseline) vs Hybrid (improvement)"
	python scripts/run_eval.py --output ./data/eval_results/assignment_evaluation.json --export-cases ./data/eval_results/assignment_detailed.json
	@echo "✅ Evaluation complete! Check ./data/eval_results/ for results"

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

# Load test specific approaches
load-test-ocr:
	python scripts/load_test.py --approach ocr_llm --requests 30 --concurrent 5

load-test-vlm:
	python scripts/load_test.py --approach vlm_direct --requests 30 --concurrent 5

load-test-hybrid:
	python scripts/load_test.py --approach hybrid --requests 30 --concurrent 5

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Analytics
analytics:
	python scripts/analytics_dashboard.py --summary

analytics-charts:
	python scripts/analytics_dashboard.py --charts
	@echo "📊 Charts generated in ./data/analytics/"

analytics-report:
	python scripts/analytics_dashboard.py --report
	@echo "📝 Detailed report generated in ./data/analytics_report.txt"

analytics-export:
	python scripts/analytics_dashboard.py --export
	@echo "📤 Data exported to CSV in ./data/exports/"

analytics-clean:
	python scripts/analytics_dashboard.py --clean
	@echo "🧹 Old cache and data cleaned"

# Monitoring
monitoring-up:
	@echo "🚀 Starting Prometheus & Grafana monitoring stack..."
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started!"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "⚡ API Metrics: http://localhost:8000/metrics"

monitoring-down:
	@echo "🛑 Stopping monitoring stack..."
	docker-compose -f docker-compose.monitoring.yml down
	@echo "✅ Monitoring stack stopped"

monitoring-logs:
	docker-compose -f docker-compose.monitoring.yml logs -f

# Docker
docker-build:
	docker build -t error-detection-api .

docker-run:
	docker run -p 8000:8000 --env-file .env error-detection-api

