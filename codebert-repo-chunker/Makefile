# Makefile for CodeBERT Repository Chunker

.PHONY: help install test clean run analyze format lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean output files"
	@echo "  make run       - Process a repository (REPO=path)"
	@echo "  make analyze   - Analyze a repository (REPO=path)"
	@echo "  make format    - Format code with black"
	@echo "  make lint      - Run linting"

install:
	pip install -r requirements.txt
	pip install -e .
	python scripts/download_models.py

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	rm -rf output/*
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

run:
	@if [ -z "$(REPO)" ]; then \
		echo "Usage: make run REPO=/path/to/repo"; \
		exit 1; \
	fi
	python scripts/process_repo.py $(REPO)

analyze:
	@if [ -z "$(REPO)" ]; then \
		echo "Usage: make analyze REPO=/path/to/repo"; \
		exit 1; \
	fi
	python scripts/analyze_repo.py $(REPO)

format:
	black src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/
	mypy src/