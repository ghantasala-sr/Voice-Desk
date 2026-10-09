# Week 1 — status

**Voice loop and harness skeleton.** Everything below either passes a test in
this repo or is explicitly marked as not yet verified.

```
57 passed, 4 skipped in 1.37s     # skipped = database tests, need `make up`
ruff: all checks passed
```

---

## Quick start

```bash
make venv                                   # .venv on python 3.12 + dev deps
cp .env.example .env                        # add ANTHROPIC_API_KEY
make test                                   # 57 tests, ~1.4s, no network
make script FILE=tests/scripts/interrupted.yaml
make chat                                   # needs ANTHROPIC_API_KEY
```

Nothing above needs Docker, a microphone, or an audio stack. That is the
architectural boundary paying for itself on day one.

For the browser demo:

```bash
make up            # postgres + jaeger
make audio         # pipecat + silero
make dev           # http://localhost:8000
```

---

## What each task delivered

| Task | State | Where | Acceptance |
|---|---|---|---|
| **W1-1** scaffold | done | `pyproject.toml`, `Makefile`, `docker-compose.yml`, `.env.example`, `core/settings.py` | `make test` green on a clean venv; every non-secret has a default |
| **W1-2** architecture contract | **done + demonstrated** | `tests/test_architecture.py`, importlinter config in `pyproject.toml` | Adding `import app` to `harness/segmenter.py` fails with `harness/segmenter.py imports 'app'`. Verified, then reverted. |
| **W1-3** migrations | written, **not applied** | `migrations/001–003.sql`, `tests/test_migrations.py` | Docker is not running on this machine, so the schema has not touched a live Postgres. The tests are written and will run on `make up`. |
| **W1-4** provider interfaces | done | `providers/{llm,stt,tts}/base.py`, `replay.py`, `null.py` | 6 tests incl. ordering, multi-response fixtures, and real-await cancellation |
| **W1-5** harness + text driver | done | `harness/{state,events,loop,segmenter,context,prompts}.py`, `eval/text_driver.py` | 13 loop tests; every turn persists with exactly one `stop_reason` |
| **W1-6** real providers | written, **not called** | `providers/llm/anthropic.py`, `openai.py`, `stt/deepgram.py`, `tts/cartesia.py` | No API keys in this environment. Structure is tested; the wire format is not. |
| **W1-7** pipeline + browser client | partial | `app/main.py`, `app/bridge.py`, `app/pipeline.py`, `app/static/*` | `/health` and the client are served and tested. The Pipecat path needs `make audio` + 3 keys. |
| **W1-8** barge-in + context hygiene | done | `harness/context.py`, `harness/loop.py` | 18 tests across `test_context_hygiene.py` and the loop suite |
| **W1-9** endpointing tuning | **not started** | — | Needs a microphone and recorded pause utterances. See below. |
| **W1-10** OTel | wired, **not observed** | `core/tracing.py`, spans throughout `loop.py` | Degrades to no-op without a collector; not yet seen in Jaeger. |

---

## The three ideas worth reviewing

**1. `generated ⊇ approved ⊇ spoken`.** Three separate texts on every turn
(`harness/state.py`). On a barge-in, history is rewritten to `spoken_text` —
what the caller actually heard — not to what we generated. The estimate is
deliberately biased low (`harness/context.py::estimate_spoken_text`): the agent
may repeat itself, but it will never believe it said something the caller never
heard.

**2. Interruption is detected while waiting for the first token.**
`harness/loop.py::_stream_until` races three things on every step: the next LLM
event, the interrupt flag, and a one-shot budget timer. A plain `async for`
would block on the next token — and the moment a caller is most likely to cut
in is exactly while the agent is still thinking.

**3. A failure never becomes an improvised answer.** Provider errors retry once
(only if budget allows), then speak a *fixed template line* and raise a handoff
signal. `test_provider_error_retries_once_then_falls_back` asserts the exact
string. There is no path in `loop.py` where an exception handler produces
caller-facing text the model wrote.

---

## Deviations from the spec

| Change | Why |
|---|---|
| Added a `core/` package (settings, ids, clock, logging, db, tracing) | The flat tree in docs/02 had no home for shared primitives. `core` is forbidden from importing anything of ours, so it cannot become a junk drawer. Recorded in docs/02 §1. |
| Added `harness/context.py` and `analytics/store.py` | Context hygiene and turn persistence were described in the docs but had no module. docs/02 updated. |
| `.venv` + pip instead of `uv` | `uv` is not installed here. Every Make target runs `.venv/bin/python`, so there is nothing to activate. `uv sync`/`uv run` still work as a drop-in. |
| `app/bridge.py` is Pipecat-free | Playout accounting — turning delivered audio bytes into `played_chars` — is the one piece of real logic on that boundary, and it feeds `spoken_text`. Keeping it out of the Pipecat shim means it is unit-tested now (`tests/test_bridge.py`) rather than after an audio install. |
| `/health` boots without Postgres | Opening the pool against a dead port blocked for the full timeout, which made `make dev` a 10-second wait. It now probes first and records the call only if the database is up. |

---

## What is not verified, and what it would take

1. **The database schema has never been applied.** Docker is not running here.
   `make up && make test` runs the four skipped tests, including the one that
   proves the partial unique index really blocks a duplicate side effect.
2. **No provider has been called.** `anthropic.py`, `openai.py`,
   `deepgram.py`, `cartesia.py` are structurally sound and lint clean, but
   their request and response shapes have not met a real API. The Cartesia
   `Cartesia-Version` header and Deepgram's `nova-3` options in particular
   should be checked against current docs on first run.
3. **The Pipecat pipeline has not run.** `app/pipeline.py` isolates every
   Pipecat import inside `_run_audio()` so a version bump is a change to one
   function. Expect to adjust the service constructors on first install.
4. **W1-9 endpointing tuning is outstanding.** It needs you, a microphone, and
   ten utterances — three with deliberate mid-sentence pauses. It is the
   largest single term in voice-to-voice latency and the main driver of whether
   the agent feels rude, so it should not be skipped. Protocol in
   docs/10-audio-pipeline.md §2.
5. **No latency number exists yet.** W1-10's acceptance criterion is a run
   directory with p50/p95 over ≥150 turns. That needs 2 and 3 first.

---

## Next

W1-6, W1-7 and W1-9 need a machine with Docker, API keys, and a microphone.
Everything else in Week 1 is done and tested. Week 2 (`BUILD.md`) starts with
**W2-1, the corpus** — written before the retrieval code, so the corpus is not
shaped to flatter a retriever that already exists.
