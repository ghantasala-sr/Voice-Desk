# 02 — Repo layout and tooling

## 1. Tree

```
voicedesk/
  pyproject.toml
  uv.lock
  Makefile
  docker-compose.yml
  Dockerfile
  .env.example
  README.md
  BUILD.md
  docs/

  core/                       # shared primitives; depends on nothing of ours
    settings.py               # pydantic-settings, every tunable in one place
    ids.py                    # ULID generation
    clock.py                  # injectable Clock + Stopwatch (deterministic tests)
    logging.py                # structlog + contextvar-bound call_id/turn_id
    db.py                     # psycopg pool, liveness probe
    tracing.py                # OTel setup + the span() helper

  app/
    main.py                   # FastAPI app, /ws endpoint, /health, /static
    pipeline.py               # Pipecat wiring — the ONLY file importing pipecat
    bridge.py                 # Pipecat frame <-> harness event adapter
    static/
      index.html              # browser client: mic capture, playback, transcript
      client.js

  harness/
    __init__.py
    loop.py                   # turn loop
    state.py                  # CallState, TurnState, ToolExecution, enums
    events.py                 # harness-level event types (in/out)
    prompts.py                # system prompt assembly, templates
    segmenter.py              # token stream -> clause/sentence boundaries
    checker.py                # grounding verdicts
    normalize.py              # spoken-form normalization
    handoff.py                # hard rules, soft scoring, summary
    tools/
      registry.py             # loads tools.yaml, validates schemas
      execution.py            # idempotency, dispatch, reconciliation
      readback.py             # read-back templates + confirmation detection
      mocks/
        __init__.py
        ticketing.py          # mock implementations
        faults.py             # fault profiles (latency, duplicate, ambiguity)

  providers/
    stt/   base.py  deepgram.py  faster_whisper.py  null.py
    tts/   base.py  cartesia.py  null.py
    llm/   base.py  anthropic.py  openai.py  replay.py

  retrieval/
    ingest.py                 # corpus -> chunks -> embeddings -> Postgres
    chunk.py                  # heading-aware chunker
    search.py                 # hybrid BM25 + dense with RRF
    embed.py                  # embedding model wrapper

  analytics/
    events.py                 # structured event writer (closed vocabulary)
    store.py                  # calls/turns persistence; Null + Postgres
    gap_report.py             # clustering + ranking job
    report.py                 # static HTML renderer
    templates/report.html.j2

  domains/
    ticketing/
      corpus/                 # 12 markdown docs (9 ingested, 3 held out)
      corpus_manifest.yaml    # which docs are ingested vs held out
      tools.yaml
      handoff.yaml
      prompts.yaml            # domain-specific persona + guardrail text
    ecommerce-returns/        # stretch

  eval/
    dataset/
      cases.yaml              # ~50 labelled cases
      audio/                  # generated; gitignored except manifest
      audio_manifest.json
    synth.py                  # case text -> TTS audio, 3 voices
    noise_mix.py              # SNR mixing over active speech
    telephony.py              # 8 kHz mu-law variant
    replay.py                 # audio -> full pipeline, no mic
    text_driver.py            # typed input, mocked STT/TTS
    judge.py                  # LLM-as-judge with rubric
    metrics.py                # WER, entity accuracy, precision/recall, Wilson CI
    caller_sim/
      personas.yaml
      simulate.py
    fault_injection/
      conftest.py
      test_tools.py
      test_interruption.py
      test_stt.py
      test_retrieval.py

  migrations/                 # numbered SQL, applied in order
    001_init.sql
    002_corpus.sql
    003_eval.sql

  results/
    README.md                 # the reporting rules
    <YYYY-MM-DD>-<slug>/

  tests/
    test_harness_loop.py
    test_checker.py
    test_idempotency.py
    test_handoff.py
    test_retrieval.py
    test_no_pipecat_in_harness.py
```

## 2. Python and dependency management

**Python 3.12**, in a project-local `.venv`:

```bash
make venv        # python3.12 -m venv .venv && pip install -e '.[dev]'
make audio       # only when you need `make dev` (pipecat + silero)
```

Every Make target runs `.venv/bin/python`, so there is nothing to activate and
no way to accidentally run against the system interpreter. `uv` works as a
drop-in replacement (`uv sync`, `uv run`) if you prefer it; the dependency
declarations below are the same either way.

> Pin exact versions in `uv.lock` at install time. The constraints below are floors;
> do not copy a version number from this document into `pyproject.toml` without checking
> what resolves today.

```toml
[project]
name = "voicedesk"
requires-python = ">=3.12,<3.13"

dependencies = [
  # audio pipeline
  "pipecat-ai[deepgram,cartesia,silero,websocket]",

  # web
  "fastapi",
  "uvicorn[standard]",
  "websockets",

  # model providers
  "anthropic",
  "openai",

  # retrieval
  "psycopg[binary,pool]",
  "pgvector",
  "sentence-transformers",

  # config / validation
  "pydantic>=2",
  "pydantic-settings",
  "pyyaml",

  # observability
  "opentelemetry-sdk",
  "opentelemetry-exporter-otlp-proto-grpc",
  "opentelemetry-instrumentation-fastapi",

  # utility
  "structlog",
  "typer",
  "jinja2",
  "numpy",
  "scipy",
]

[project.optional-dependencies]
local-stt = ["faster-whisper", "ctranslate2"]
eval = ["jiwer", "soundfile", "librosa", "pandas", "matplotlib", "scikit-learn"]
dev  = ["pytest", "pytest-asyncio", "pytest-cov", "ruff", "mypy", "import-linter"]
```

### Notable choices

- **`psycopg` 3, not SQLAlchemy.** The schema is small and the retrieval query is a
  hand-written CTE doing RRF. An ORM adds indirection over SQL that is the interesting part.
- **`import-linter`** enforces the architectural rule from
  [01-architecture.md](01-architecture.md) §1 — `harness` may not import `pipecat`, `app`,
  or `eval`. This is the contract that keeps the text driver possible, so it is checked by
  CI rather than by discipline.
- **`faster-whisper` is an optional extra.** It drags in CTranslate2 and model weights; the
  default demo path does not need it.

```toml
# contract enforced in CI — see tests/test_no_pipecat_in_harness.py as a backstop
[[tool.importlinter.contracts]]
name = "harness is audio-agnostic"
type = "forbidden"
source_modules = ["harness", "retrieval"]
forbidden_modules = ["pipecat", "app", "eval"]
```

## 3. Environment variables

`.env.example`, loaded via `pydantic-settings`. Every setting has a default except keys.

```bash
# ---- providers ----
ANTHROPIC_API_KEY=
OPENAI_API_KEY=                        # optional, for --llm openai
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
CARTESIA_VOICE_ID=

VD_LLM_PROVIDER=anthropic              # anthropic | openai | replay
VD_LLM_MODEL=claude-sonnet-5
VD_STT_PROVIDER=deepgram               # deepgram | faster_whisper | null
VD_TTS_PROVIDER=cartesia               # cartesia | null

# ---- database ----
VD_DATABASE_URL=postgresql://voicedesk:voicedesk@localhost:5432/voicedesk

# ---- domain ----
VD_DOMAIN=ticketing

# ---- turn-taking (see docs/10) ----
VD_VAD_STOP_SECS=0.4
VD_VAD_START_SECS=0.2
VD_VAD_CONFIDENCE=0.7
VD_SILENCE_TIMEOUT_MS=8000

# ---- budgets (see docs/01 §5) ----
VD_TURN_BUDGET_MS=2500
VD_TOOL_FILLER_MS=700
VD_TOOL_TIMEOUT_MS=4000
VD_MAX_TOOL_ITERATIONS=3

# ---- retrieval ----
VD_EMBED_MODEL=BAAI/bge-small-en-v1.5
VD_RETRIEVAL_TOP_K=4
VD_RRF_K=60

# ---- observability ----
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=voicedesk
VD_LOG_LEVEL=INFO
```

Rule: **no magic numbers in code.** Anything a tuning log might change lives in settings or
in `handoff.yaml`.

## 4. docker-compose.yml

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: voicedesk
      POSTGRES_PASSWORD: voicedesk
      POSTGRES_DB: voicedesk
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U voicedesk"]
      interval: 5s
      retries: 10

  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      COLLECTOR_OTLP_ENABLED: "true"
    ports:
      - "16686:16686"   # UI
      - "4317:4317"     # OTLP gRPC

volumes:
  pgdata:
```

The app itself runs on the host during development (`make dev`) for fast reload, and in the
Dockerfile image for deployment.

## 5. Makefile

```make
up:          ## start postgres + jaeger
	docker compose up -d && ./scripts/wait_for_pg.sh

dev: up      ## run the app with reload
	uv run uvicorn app.main:app --reload --port 8000

ingest:      ## build the knowledge base from the active domain
	uv run python -m retrieval.ingest --domain $(DOMAIN)

chat:        ## text-mode REPL against the harness, no audio
	uv run python -m eval.text_driver --domain $(DOMAIN) --interactive

demo: dev    ## open the browser client

test:        ## unit + contract tests
	uv run pytest tests/ -q
	uv run lint-imports

faults:      ## fault injection suite
	uv run pytest eval/fault_injection/ -q

synth:       ## generate eval audio from cases.yaml
	uv run python -m eval.synth && uv run python -m eval.noise_mix && uv run python -m eval.telephony

eval:        ## full replay eval -> results/<date>-<slug>/
	uv run python -m eval.replay --suite $(SUITE) --label $(LABEL)

simulate:    ## generate synthetic call population
	uv run python -m eval.caller_sim.simulate --n $(N)

gap-report:  ## cluster failures, emit gap report
	uv run python -m analytics.gap_report

report:      ## render static HTML report
	uv run python -m analytics.report

fmt:
	uv run ruff format . && uv run ruff check --fix .

DOMAIN ?= ticketing
SUITE  ?= all
N      ?= 300
```

## 6. Dockerfile

Multi-stage, non-root, no model weights baked in unless `local-stt` is requested.

```dockerfile
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM base AS runtime
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The embedding model (`bge-small`, ~130 MB) is downloaded on first ingest and cached in a
volume. Do not bake it into the image; it makes rebuilds slow for no benefit.

## 7. Test strategy

| Layer | Tool | Speed | Runs |
|---|---|---|---|
| Harness unit | pytest + `text_driver` + `providers/llm/replay.py` | ms | every commit |
| Architecture contract | `import-linter` | ms | every commit |
| Idempotency / faults | pytest + adversarial mocks | seconds | every commit |
| Retrieval | pytest against a seeded test DB | seconds | every commit |
| Replay eval | `eval/replay.py` over audio | minutes | on demand, recorded to `results/` |

`providers/llm/replay.py` is a deterministic `LLMProvider` that replays recorded token
streams from a fixture file. It is what makes harness tests fast and free: the loop's
logic is exercised without a network call, including tool-call sequences and mid-stream
interruptions.
