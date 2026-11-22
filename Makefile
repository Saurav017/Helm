.PHONY: help install dev lint format typecheck test test-cov ci clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install helmx in editable mode
	pip install -e .

dev: ## Install with all dev dependencies
	pip install -e ".[dev]"

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

typecheck: ## Run mypy strict type checking
	mypy --strict helm/

test: ## Run all tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=helm --cov-report=html --cov-report=term-missing

ci: format lint typecheck test ## Run full CI pipeline locally

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
