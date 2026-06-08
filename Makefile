.PHONY: help format lint check fix type-check test ci clean pre-commit-install pre-commit-run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==================== Code Quality ====================

format: ## Format all Python files with ruff
	ruff format .

lint: ## Run ruff linter (report only)
	ruff check .

fix: ## Run ruff linter with auto-fix
	ruff check --fix .

check: ## Run both lint and format check (CI-friendly, no modification)
	ruff check .
	ruff format --check .

type-check: ## Run mypy type checker
	mypy dlrouter/

# ==================== Testing ====================

test: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=dlrouter --cov-report=term-missing

# ==================== Pre-commit ====================

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# ==================== Maintenance ====================

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== All-in-one ====================

ci: check type-check test ## CI pipeline (check only, no file modification)

all: fix format type-check test ## Fix, format, type-check, and test
