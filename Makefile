.PHONY: help install test lint format clean build publish

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing -f

lint: ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black and isort
	black .
	isort .

format-check: ## Check if code is formatted correctly
	black . --check
	isort . --check-only

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

publish: ## Publish to PyPI (requires twine)
	twine upload dist/*

demo: ## Run demo shape detection
	python demo_shape_detector.py hq720.jpg

demo-all: ## Run all detection systems
	python shape_detection.py
	python advanced_shape_detection.py
	python intelligent_shape_detector.py
	python demo_shape_detector.py hq720.jpg

check: ## Run all checks (lint, test, format)
	make lint
	make format-check
	make test

pre-commit: ## Run pre-commit checks
	make format
	make lint
	make test

setup-git: ## Setup Git hooks
	pre-commit install

docs: ## Generate documentation
	python -c "import pdoc; pdoc.render_docs('shape_detection.py', output_dir='docs')"

serve-docs: ## Serve documentation locally
	python -m http.server 8000 --directory docs/

docker-build: ## Build Docker image
	docker build -t advanced-shape-detection .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/app advanced-shape-detection

docker-test: ## Run tests in Docker
	docker run -it --rm -v $(PWD):/app advanced-shape-detection make test
