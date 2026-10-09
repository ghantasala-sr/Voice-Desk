.DEFAULT_GOAL := help
.PHONY: help venv audio retrieval up down clean migrate psql traces \
        lint-corpus ingest reingest dev chat script demo-scripts faults test lint fmt

# Everything runs out of the project virtualenv. Create it once with `make venv`.
PY      := .venv/bin/python
PIP     := .venv/bin/pip
DOMAIN  ?= ticketing
FILE    ?= tests/scripts/basic.yaml
APP_PORT ?= 8088

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

venv:  ## Create .venv (python 3.12) and install week-1 dependencies
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'
	@echo "done — copy .env.example to .env and add ANTHROPIC_API_KEY"

audio:  ## Install the audio extra, needed only by `make dev`
	$(PIP) install -e '.[audio]'

retrieval:  ## Install the retrieval extra (sentence-transformers + torch, ~2GB)
	$(PIP) install -e '.[retrieval]'

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

up:  ## Start Postgres + Jaeger and wait for readiness
	docker compose up -d
	./scripts/wait_for_pg.sh
	@echo "postgres  :$${VD_PG_PORT:-5434}"
	@echo "jaeger UI http://localhost:$${VD_JAEGER_UI_PORT:-16687}"

down:  ## Stop containers (keeps data)
	docker compose down

clean:  ## Stop containers AND delete all data. Destroys the database.
	docker compose down -v

migrate:  ## Re-apply migrations to a running database
	./scripts/migrate.sh

psql:  ## Open a psql shell
	docker compose exec postgres psql -U voicedesk -d voicedesk

traces:  ## Open the Jaeger UI
	open http://localhost:$${VD_JAEGER_UI_PORT:-16687}

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

lint-corpus:  ## Check the corpus against the writing rules (docs/05 §1)
	$(PY) -m retrieval.lint_corpus --domain $(DOMAIN)

ingest: lint-corpus  ## Build the knowledge base. Idempotent.
	$(PY) -m retrieval.ingest --domain $(DOMAIN)

reingest:  ## Re-embed everything, ignoring content hashes
	$(PY) -m retrieval.ingest --domain $(DOMAIN) --force

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

dev:  ## Run the app with reload — browser client at http://localhost:$(APP_PORT)
	$(PY) -m uvicorn app.main:app --reload --port $(APP_PORT)

chat:  ## Text-mode REPL against the harness. No audio, no microphone.
	$(PY) -m eval.text_driver --domain $(DOMAIN) --interactive

script:  ## Run a text-driver script: make script FILE=tests/scripts/interrupted.yaml
	$(PY) -m eval.text_driver --script $(FILE)

demo-scripts:  ## Run the four demo beats, recorded to Postgres (watch /console)
	@for f in demo/*.yaml; do echo "\n=== $$f"; $(PY) -m eval.text_driver --script $$f || exit 1; done

faults:  ## Fault-injection suite (docs/12 §7)
	$(PY) -m pytest eval/fault_injection -q

# ---------------------------------------------------------------------------
# Checks — run before every commit
# ---------------------------------------------------------------------------

test:  ## Unit tests + the architecture contract
	$(PY) -m pytest -q
	@$(PY) -m importlinter.cli lint_imports 2>/dev/null \
		|| echo "  (import-linter not installed; tests/test_architecture.py covers the same rule)"

lint:  ## Lint and type-check
	$(PY) -m ruff check .
	$(PY) -m mypy harness core providers

fmt:  ## Format and auto-fix
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .
