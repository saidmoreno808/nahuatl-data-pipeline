# Makefile for CORC-NAH Project
# Run commands with: make <target>

.PHONY: help setup test lint format clean golden parity install install-dev

# Default target
help:
	@echo "CORC-NAH Project Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup          Complete project setup"
	@echo ""
	@echo "Day 0 (Golden Dataset):"
	@echo "  make golden         Generate golden dataset"
	@echo "  make parity         Run parity tests"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-parity    Run parity tests only"
	@echo "  make coverage       Generate coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (flake8, mypy)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove generated files"
	@echo "  make clean-all      Remove all generated files + venv"

# ============================================================================
# Installation
# ============================================================================

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

setup: install-dev
	pre-commit install
	@echo "✅ Project setup complete"

# ============================================================================
# Day 0: Golden Dataset Generation
# ============================================================================

golden:
	@echo "📊 Generating golden dataset..."
	python scripts/unify_datasets.py
	mkdir -p benchmark/
	cp data/gold/train_v1.jsonl benchmark/golden_train_v1.jsonl
	cp data/gold/validation_v1.jsonl benchmark/golden_validation_v1.jsonl
	cp data/gold/test_v1.jsonl benchmark/golden_test_v1.jsonl
	md5sum benchmark/golden_*.jsonl > benchmark/checksums.txt
	python benchmark/generate_stats.py
	@echo "✅ Golden dataset generated"

parity:
	@echo "🔍 Running parity tests..."
	pytest tests/integration/test_parity_with_legacy.py -v -m parity

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-parity:
	pytest tests/integration/test_parity_with_legacy.py -v

coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated: htmlcov/index.html"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "🔍 Running flake8..."
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__,.venv
	@echo "🔍 Running mypy..."
	mypy src/ --ignore-missing-imports

format:
	@echo "🎨 Formatting with black..."
	black src/ tests/ benchmark/ --line-length=100
	@echo "🎨 Sorting imports with isort..."
	isort src/ tests/ benchmark/ --profile black

check: format lint test
	@echo "✅ All quality checks passed"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "🧹 Deep cleaning (removing venv)..."
	rm -rf .venv/
	@echo "✅ Deep cleanup complete"

# ============================================================================
# Documentation
# ============================================================================

docs:
	cd docs && sphinx-build -b html . _build
	@echo "📚 Documentation generated: docs/_build/index.html"

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t corc-nah:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# ============================================================================
# Development Shortcuts
# ============================================================================

watch-tests:
	pytest-watch tests/

jupyter:
	jupyter lab --no-browser --ip=0.0.0.0

shell:
	python -i -c "from src import *"
