# Makefile for oncai-review — local dev, build, and run.
# Requires: uv (https://docs.astral.sh/uv/). JS tooling (Node + npm) is needed
# for `make test-js`, `make lint`, and `make format` — run `make install` first.

# PyInstaller's --add-data separator differs by OS: ';' on Windows, ':' elsewhere.
ifeq ($(OS),Windows_NT)
  DATA_SEP := ;
  BINARY := dist/oncai-review.exe
  ICON_FLAG := --icon assets/icon.ico
else
  DATA_SEP := :
  BINARY := dist/oncai-review
  ICON_FLAG :=
endif

# By default no port is forced, so the server uses its default and auto-falls
# back to an open one if it's taken. Pin it with: make start PORT=9000
PORT ?=
PORT_ARG := $(if $(PORT),--port $(PORT))

# Version (single source of truth: pyproject.toml) + arch, for naming bundles.
VERSION := $(shell python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])" 2>/dev/null)
ARCH := $(shell uname -m)

.DEFAULT_GOAL := help
.PHONY: help install start demo build build-app test test-js lint format check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dev tooling (uv dev group + npm dev deps)
	uv sync --group dev
	npm install

start: ## Run the review server (optional: PORT=9000)
	python server.py $(PORT_ARG)

demo: ## Run the server with the bundled synthetic demo package
	python server.py --package examples/demo.review_pkg.json $(PORT_ARG)

build: ## Build a single-file executable into dist/ (PyInstaller)
	uvx pyinstaller --onefile $(ICON_FLAG) --name oncai-review \
		--add-data "web$(DATA_SEP)web" \
		--add-data "pyproject.toml$(DATA_SEP)." server.py
	@echo "Built $(BINARY)"

build-app: ## Build a double-clickable macOS .app (opens in Terminal), zipped (mac only)
	# Ship a console binary wrapped in a tiny .app that opens it in Terminal.
	# See scripts/build-macos-app.sh for why (a windowed app loses its Dock icon
	# and hangs when you double-click it again while it's running).
	uvx pyinstaller --onefile --icon assets/icon.icns --name oncai-review \
		--add-data "web$(DATA_SEP)web" \
		--add-data "pyproject.toml$(DATA_SEP)." server.py
	bash scripts/build-macos-app.sh dist/oncai-review assets/icon.icns $(VERSION) dist
	ditto -c -k --keepParent dist/oncai-review.app dist/oncai-review-$(VERSION)-macos-$(ARCH).zip
	@echo "Built dist/oncai-review-$(VERSION)-macos-$(ARCH).zip  (see docs/RUNNING-ON-MAC.md)"

lint: ## Lint & type-check everything (ruff, ty, eslint, prettier --check)
	uvx ruff check .
	uv run --group dev ty check .
	npx eslint .
	npx prettier --check .

format: ## Auto-format & auto-fix (ruff, prettier, eslint --fix)
	uvx ruff format .
	npx prettier --write .
	npx eslint . --fix

test: ## Run the Python test suite
	uv run --group dev pytest -q

test-js: ## Run the front-end test suite (Node's built-in runner)
	node --test "web/*.test.js"

check: lint test test-js ## Run everything: lint + all tests

clean: ## Remove build artifacts (dist/, build/, *.spec)
	rm -rf build dist *.spec
