# Root Makefile for the OncAI monorepo.
# Core Python package commands run from repo root; review-app commands delegate
# into apps/oncai-review, where that app keeps its own tooling and build rules.

REVIEW_APP_DIR := apps/oncai-review
PORT ?=

.DEFAULT_GOAL := help

.PHONY: \
	help \
	install install-core install-review \
	lint lint-core lint-review \
	test test-core test-review test-review-py test-review-js \
	format format-core format-review \
	check check-core check-review \
	secrets \
	review-start review-demo build-review build-review-app clean clean-core clean-review

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: install-core install-review ## Install core and review-app dev tooling

install-core: ## Install core package dev dependencies
	uv sync --extra dev

install-review: ## Install review-app dev dependencies
	$(MAKE) -C $(REVIEW_APP_DIR) install

lint: lint-core lint-review ## Lint/type-check core and review app

lint-core: ## Lint/type-check the core oncai package
	uv run ruff check src/oncai
	uv run ty check src/oncai

lint-review: ## Lint/type-check the review app
	$(MAKE) -C $(REVIEW_APP_DIR) lint

test: test-core test-review ## Run core and review-app tests

test-core: ## Run the core Python test suite
	uv run pytest tests -q

test-review: test-review-py test-review-js ## Run all review-app tests

test-review-py: ## Run review-app Python tests
	$(MAKE) -C $(REVIEW_APP_DIR) test

test-review-js: ## Run review-app front-end tests
	$(MAKE) -C $(REVIEW_APP_DIR) test-js

format: format-core format-review ## Format/fix core and review app

format-core: ## Format the core Python package and tests
	uv run ruff format src/oncai tests
	uv run ruff check --fix src/oncai

format-review: ## Format/fix the review app
	$(MAKE) -C $(REVIEW_APP_DIR) format

check: check-core check-review ## Run all root checks

check-core: lint-core test-core ## Run core lint/type-check/tests

check-review: ## Run review-app lint/type-check/tests
	$(MAKE) -C $(REVIEW_APP_DIR) check

secrets: ## Scan the whole git history for committed secrets (needs gitleaks)
	@command -v gitleaks >/dev/null 2>&1 || { \
		echo "gitleaks not found — install it first (e.g. 'brew install gitleaks')"; \
		echo "see https://github.com/gitleaks/gitleaks#installing"; \
		exit 1; \
	}
	gitleaks git . --redact --verbose

review-start: ## Start the review app from repo root (optional: PORT=9000)
	$(MAKE) -C $(REVIEW_APP_DIR) start PORT=$(PORT)

review-demo: ## Start the review app with its synthetic demo package
	$(MAKE) -C $(REVIEW_APP_DIR) demo PORT=$(PORT)

build-review: ## Build the review-app single-file executable
	$(MAKE) -C $(REVIEW_APP_DIR) build

build-review-app: ## Build the double-clickable macOS review app bundle
	$(MAKE) -C $(REVIEW_APP_DIR) build-app

clean: clean-core clean-review ## Remove build/test artifacts for root and review app

clean-core: ## Remove core build artifacts
	rm -rf build dist wheels *.egg-info

clean-review: ## Remove review-app build artifacts
	$(MAKE) -C $(REVIEW_APP_DIR) clean
