# OoC-Optimizer developer tasks
# Run 'make help' to see available targets.

.PHONY: help env install test test-fast lint verify clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda environment from environment.yml
	conda env create -f environment.yml

install:  ## Install package in editable mode (run inside conda env)
	pip install -e ".[dev]"

test:  ## Run all tests
	pytest tests/ -v

test-fast:  ## Run tests excluding slow/openfoam markers
	pytest tests/ -v -m "not slow and not openfoam"

lint:  ## Run ruff linter
	ruff check ooc_optimizer/ tests/ scripts/

verify:  ## Run Module 1.1 Poiseuille verification
	python scripts/run_verification.py --config configs/default_config.yaml

clean:  ## Remove generated artifacts
	rm -rf data/stl/* data/cases/* data/results/* figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
