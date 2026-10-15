# 17 — The web app

`http://localhost:8088/` (any port the app runs on): one page with a top nav — **Talk**,
**Voice**, **Console ↗**, **Insights**, **Results**. Served by the same FastAPI app as the
voice socket and the console.

```bash
.venv/bin/python -m uvicorn app.main:app --port 8088
```

Code: `app/chat.py` (text-chat socket), `app/insights.py` (read-only API),
`app/static/app/` (vanilla HTML/CSS/JS, no build step, no CDN). Tests:
`tests/test_frontend.py`.

Two labels keep the numbers honest (docs/15): everything in **Talk** is tagged
**this call** — one conversation, no run directory, never quoted. **Insights** and
**Results** are tagged **measured runs** — every number comes from a `results/<run>/`
directory or the latest `gap_reports` row, and every proportion carries its `n`.

---

## 1. Views

| View | Shows |
|---|---|
| **Talk** (default, `#talk`) | Text chat with the production agent: real hybrid retrieval, tools against the mock backend, read-back confirmations, grounding checker, handoff policy — the same `Harness` and `build_stack` the voice pipeline uses. Under each reply a collapsed **details** panel: retrieved chunks by heading path (cited ones marked), retrieval top vec / lex / fused scores, every checker verdict (released or replaced, with the withheld text struck through), tool calls with their status (proposed → awaiting confirmation → confirmed → committed, or failed / declined / cancelled), handoff signals, latency stages. Fixed lines (acknowledgments, safe lines) are greyed: they are spoken but are not part of the answer. A **read-back** renders as a "Confirmation needed" bubble; the caller answers by typing — deliberately no button, so `readback.classify` sees the caller's own words. A **handoff** renders the human-agent summary (`call.handoff["summary_text"]`) as its own card, ends the session, and offers **Start new call**. A banner names the agent: with `VD_LLM_PROVIDER=anthropic` and no `ANTHROPIC_API_KEY` the chat runs the offline extractive provider (`providers/llm/extractive.py`) and says so. One-click example prompts cover a policy question, a held-out topic (VIP — currently a gap, so it hands off), the two-step transfer, and an explicit handoff. Sessions are recorded to Postgres with `source = "text"` when the database is up, so they show up in the console. |
| **Voice** (`#voice`) | The old voice client, now at `/voice`, embedded when its requirements are met. Otherwise a checklist of what is missing (`DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, an OpenAI key only if OpenAI is the configured LLM — with no Anthropic key the pipeline falls back to the offline extractive agent (`providers.offline_fallback`) — and the `[audio]` extra). |
| **Console ↗** | A link to `/console` (docs/16). |
| **Insights** (`#insights`) | The week-4 loop. **Latest gap report** (newest `gap_reports` row): window and call count, the report's ranked actions, a bar chart of cluster priority (clusters at priority 0 are listed but not drawn), and one card per cluster — diagnosis, failure rate with its fraction of `n_calls`, share of the window, median top vec score, keywords, example utterances, covering docs, held-out match, suggested action. **Latest feedback cycle** (newest `results/*-feedback-cycle/metrics.json`): the change that was made, before → after tiles for containment and resolved rate and the guardrails (handoff rate, wrong-answer rate, duplicate side effects), a before/after chart with 95% CI whiskers, the paired comparison, containment by intent, and whether the gap report found the planted gaps. Every chart has a table view. Empty state: "Run `make simulate && make gap-report && make cycle`". |
| **Results** (`#results`, `#results/<run>`) | One card per `results/<run>/` directory: name, date, the README's first heading, the first paragraph of its "What it showed" section, up to four headline numbers from `metrics.json` (the shallowest `{value, n, successes, ci95}` objects, each with its fraction and CI), and `sampled` / `timing_valid` flags. Click for the README rendered as markdown, the run's `*.png` figures, and the raw `metrics.json`. |

Light and dark themes follow `prefers-color-scheme`; the dark palette is the console's.
Layout works down to ~380 px.

**Untrusted text.** Replies, transcripts, chunk headings, cluster labels and READMEs reach
the DOM only via `textContent` / `createTextNode`. The markdown renderer
(`app/static/app/markdown.js`) builds DOM nodes, never an HTML string; it supports
headings, fenced code, tables, lists, blockquotes, rules, inline code, bold, italic and
links (http(s) only, or a PNG in the same run). `tests/test_frontend.py` fails if any app
script uses `innerHTML`.

## 2. Endpoints

| Route | Returns |
|---|---|
| `GET/HEAD /` | the app shell (`app/static/app/index.html`) |
| `GET/HEAD /voice` | the voice client (`app/static/index.html`) |
| `GET /api/status` | `{domain, agent: {configured_provider, provider, model, offline, notice}, keys: {anthropic, openai, deepgram, cartesia}, voice: {ready, missing, audio_extra}, database, chunks}`. Keys are **booleans only**; values never leave the process. |
| `GET /api/results` | `{runs: [{name, date, title, summary, has_readme, headline: [{label, value, n, successes, ci95}], flags, pngs}]}`, newest first |
| `GET /api/results/{name}` | the card fields plus `readme` (raw markdown text) and `metrics` (parsed `metrics.json` or `null`) |
| `GET /api/results/{name}/files/{file}` | a PNG from that run directory |
| `GET /api/gap-report/latest` | `{report_id, generated_at, window_start, window_end, call_count, clusters, actions}`; `404 {"error"}` if none; `503 {"error": "database unavailable"}` without the pool |
| `GET /api/feedback-cycle/latest` | `{run, has_readme, metrics}`; `404 {"error"}` if none |
| `WS /ws/chat` | the text chat, §3 |

**Path safety.** `name` and `file` are never joined onto a path as given: they must equal
an entry produced by listing `results/` (a directory, not a symlink, directly under it) or
that run's PNGs. `..`, encoded separators and unknown names are all 404.

## 3. Chat protocol (`/ws/chat`)

JSON text frames.

Client → server:

| Message | Meaning |
|---|---|
| `{"type": "user", "text": "..."}` | one caller utterance → `TranscriptIn(text, provider="text")` |
| `{"type": "end"}` | hang up → `Shutdown("caller_hung_up")` |

Closing the socket is also a `Shutdown`.

Server → client, in order of appearance:

| Message | When |
|---|---|
| `{"type": "status", "state": "connecting", "detail"}` | immediately; the stack (embedding model) is loading |
| `{"type": "ready", "call_id", "agent", "stack", "recorded", "domain"}` | the harness is running; `stack` is `build_stack`'s banner, `recorded` says whether Postgres has the call |
| `{"type": "turn_started", "turn_id", "user_text"}` | a turn began |
| `{"type": "say", "turn_id", "segment_id", "text"}` | one `Speak` segment |
| `{"type": "event", "turn_id", "name", "payload"}` | every `Emit`, verbatim (the same rows `call_events` gets) |
| `{"type": "turn", "turn_id", "details"}` | `TurnCompleted`. `details`: `stop_reason, approved_text, llm_iterations, retrieval {top_score, top_vec_score, top_lex_score, chunks [{rank, chunk_id, heading, cited}]}, checker [{segment_index, text, verdict, action, spans, checker_ms, reason}], tools [{tool, status, args, error, attempt, duplicate_suppressed}], signals, stages_ms, ack_bypass_used, budget_exceeded, awaiting_confirmation {tool, text} \| null` |
| `{"type": "handoff", "summary_text", "reason", "rule_type", "summary"}` | the call was handed off; sent just before `ended` |
| `{"type": "ended", "outcome", "reason"}` | `EndCall`; the server then closes |
| `{"type": "error", "code", "message"}` | `bad_message`, `empty`, `too_long` (the call continues) or `internal` (the call is over). Never a stack trace. |

`details` is built from the persisted `TurnState`, not reconstructed from events, so the
panel matches what the console shows for the same turn. `awaiting_confirmation` is set
whenever `call.pending_readback` is non-null after the turn: the caller's next message will
be classified against it.

**Per connection:** one `CallState(source="text")`, one `build_stack(settings,
domain=settings.domain, retrieval="real", tools=True, persist_ledger=<recorded>)`, one
`Harness`. Recording uses `PostgresStore` / `PostgresEventWriter` when the app's pool is
open (`app.state.db_ready`) and the database is reachable, flushing events per turn so the
console's live feed shows the chat as it happens; otherwise `NullStore` /
`NullEventWriter`.

**Testing seams.** `app.chat.get_chat_settings`, `get_stack_factory` and `get_persist` are
FastAPI dependencies; `tests/test_frontend.py` overrides them to run the socket with the
extractive provider, scripted chunks (`FakeRetriever`) and the in-memory ledger — no
Postgres, no embedding model, no key.

## 4. Known limits

- The first chat after the app starts spends a second or more loading the embedding model
  ("connecting…"). Later chats reuse it: `retrieval.embed.get_embedder` caches one instance
  per model name (it used to load a new one per connection, blocking the event loop).
- Text mode has no audio stages: `stages_ms` stops at `first_content`; endpoint, TTS and
  playout are absent rather than invented.
