# VoiceDesk

A real-time voice support agent that answers calls from a policy knowledge base,
takes actions through tools, checks every sentence for grounding before it is
spoken, and hands off to a human with a written summary when it is not
confident.

The engine is domain-agnostic. The reference deployment is Tier-1 phone support
for a fictional event-ticketing company, *Tide & Tempo Tickets*.

> **Status:** all four weeks built, plus a web app. 366 tests; every reported
> number is backed by a run directory in [`results/`](results/). What is and is
> not verified is spelled out in [WEEK1.md](WEEK1.md), [WEEK2.md](WEEK2.md),
> [WEEK3.md](WEEK3.md), [WEEK4.md](WEEK4.md). No API key was available while
> building, so measured agent behaviour is from an **offline extractive agent**,
> labelled as such everywhere.

---

## Contents

1. [System architecture](#1-system-architecture)
2. [The one boundary that matters](#2-the-one-boundary-that-matters)
3. [How one turn flows](#3-how-one-turn-flows)
4. [From model token to caller's ear](#4-from-model-token-to-callers-ear)
5. [Tools, read-back, and idempotency](#5-tools-read-back-and-idempotency)
6. [Barge-in](#6-barge-in)
7. [Handoff](#7-handoff)
8. [Data model](#8-data-model)
9. [Evaluation and the feedback loop](#9-evaluation-and-the-feedback-loop)
10. [Latency budget](#10-latency-budget)
11. [Measured results](#11-measured-results)
12. [Running it](#12-running-it)
13. [Repository map](#13-repository-map)
14. [Documents](#14-documents)

---

## 1. System architecture

```mermaid
flowchart LR
    subgraph clients["Callers"]
        BROWSER["Browser client<br/>mic + playback worklets"]
        PHONE["Phone - Twilio<br/>(stretch)"]
        TEXT["Text driver<br/>make chat / make script"]
        WEB["Web app  /<br/>Talk · Insights · Results"]
        SIM["Caller simulator<br/>personas × intents"]
    end

    subgraph audio["app/  —  audio domain (Pipecat)"]
        WS["FastAPI /ws<br/>PCM16 + JSON control"]
        VAD["Silero VAD<br/>endpointing"]
        STT["STT service<br/>Deepgram / faster-whisper"]
        TTS["TTS service<br/>Cartesia"]
        BRIDGE["bridge.py<br/>playout accounting"]
    end

    subgraph harness["harness/  —  conversation domain (plain Python, no audio imports)"]
        LOOP["loop.py<br/>turn loop"]
        STATE["state.py<br/>CallState · TurnState"]
        CHK["checker.py<br/>grounding rules 1 + 2"]
        NORM["normalize.py<br/>spoken form"]
        HO["handoff.py<br/>hard rules · soft score · summary"]
        EXEC["tools/execution.py<br/>idempotency ledger"]
        RB["tools/readback.py<br/>template read-back"]
    end

    subgraph services["Providers and retrieval"]
        LLM["LLMProvider<br/>Claude · OpenAI · replay · extractive"]
        RET["Hybrid search<br/>BM25 + pgvector, RRF"]
        MOCK["Mock ticketing backend<br/>+ fault profiles"]
    end

    subgraph store["PostgreSQL + pgvector"]
        KB[("documents · chunks")]
        CALLS[("calls · turns · call_events")]
        LEDGER[("tool_executions<br/>partial unique index")]
        CV[("checker_verdicts · handoffs")]
    end

    subgraph observe["Observability"]
        JAEGER["Jaeger<br/>OTel traces"]
        CONSOLE["Live console<br/>/console · SSE"]
        GAPR["Gap report +<br/>static report"]
    end

    BROWSER --> WS
    PHONE -.-> WS
    WS --> VAD --> STT -->|final transcript| BRIDGE
    VAD -.->|barge-in| BRIDGE
    BRIDGE <--> LOOP
    TEXT <--> LOOP
    WEB <-->|/ws/chat| LOOP
    SIM <--> LOOP
    LOOP --> TTS --> WS

    LOOP --> RET --> KB
    LOOP --> LLM
    LOOP --> CHK --> NORM
    LOOP --> HO
    LOOP --> RB --> EXEC --> MOCK
    EXEC --> LEDGER
    LOOP --> STATE

    LOOP -->|events + turns| CALLS
    HO --> CV
    CHK --> CV
    LOOP -.->|spans| JAEGER
    CALLS --> CONSOLE
    LEDGER --> CONSOLE
    CV --> CONSOLE
    CALLS --> GAPR
```

| Layer | Choice | Why |
|---|---|---|
| Audio | Pipecat, Silero VAD | transport, VAD and barge-in events, nothing else |
| LLM | Claude (default), OpenAI, deterministic replay, offline extractive | replay makes the harness testable without a network; extractive runs the whole system with no key, and says so |
| STT | Deepgram (streaming), faster-whisper (local batch) | comparing them is a result in itself |
| TTS | Cartesia | lowest time-to-first-byte, which sits inside the latency budget |
| Retrieval | PostgreSQL + pgvector + full-text, BGE-small embeddings | hybrid, because policy text turns on exact terms like "will-call" |
| State | PostgreSQL | calls, turns, events, the idempotency ledger, eval results |
| Observability | OpenTelemetry → Jaeger, a live console over SSE, a web app, a static HTML report | traces for one call, tables for many, provenance for every number |

---

## 2. The one boundary that matters

> **Pipecat owns audio. The harness owns the conversation.**

```mermaid
flowchart LR
    subgraph A["app/ — may import pipecat"]
        P["pipeline.py"]
        B["bridge.py"]
    end
    subgraph H["harness/ + retrieval/ — may NOT import pipecat, app, or eval"]
        L["loop.py and everything it uses"]
    end
    subgraph T["eval/ — test surfaces"]
        D["text_driver.py"]
    end
    P --> B -->|InEvents| L
    L -->|OutCommands| B
    D -->|same InEvents| L
    L -->|same OutCommands| D
```

The harness receives events (`TranscriptIn`, `BargeIn`, `SilenceTimeout`,
`Shutdown`) and emits commands (`Speak`, `CancelSpeech`, `EndCall`, `Emit`,
`TurnCompleted`). It never touches an audio device. That is what lets the entire
conversational system run under pytest with typed text in milliseconds. The rule
is enforced by `tests/test_architecture.py` and an import-linter contract, so it
cannot erode quietly.

---

## 3. How one turn flows

```mermaid
flowchart TD
    START(["Final transcript arrives"]) --> MK["Build TurnState<br/>append user message"]
    MK --> PG{"1 · Hard utterance rule?<br/>'talk to a manager', chargeback, emergency"}
    PG -->|yes| HOFF1["Hand off<br/>0 LLM calls"]
    PG -->|no| RBQ{"1b · Read-back pending?"}

    RBQ -->|yes| CLS{"Classify reply"}
    CLS -->|confirm| DISP["Dispatch the action<br/>outcome becomes an action note"]
    CLS -->|deny or correction| CANC["Cancel the pending claim<br/>note: NOT performed"]
    CLS -->|unclear, first time| REASK["Re-ask the read-back verbatim"] --> DONE
    CLS -->|unclear, second time| SIG1["Raise readback_unclear"] --> DEC1
    DISP --> SIGS
    CANC --> SIGS
    RBQ -->|no| SIGS

    SIGS["2 · Utterance signals<br/>rephrase · low-confidence entity"] --> RET["3 · Hybrid retrieval<br/>4 signals recorded"]
    RET --> DEC1{"4 · Decide before the model<br/>hard state rules + soft score"}
    DEC1 -->|hand off| HOFF2["Hand off"]
    DEC1 -->|continue| GEN

    subgraph GEN["5 · Iterate — up to max_tool_iterations LLM calls"]
        STREAM["Stream LLM response"] --> SEG["Segment text"]
        SEG --> REL["Release each segment<br/>strip citations → check → normalize → speak"]
        STREAM --> TC{"Tool call?"}
        TC -->|read-only| RUN["Execute · append tool_use + tool_result"] --> STREAM
        TC -->|needs confirmation| RBS["Speak template read-back<br/>claim key as pending"]
        REL -->|checker fails| HALT["Speak safe line · stop generating"]
    end

    GEN --> STOP{"6 · Stop"}
    STOP -->|barge-in| INT["Rewrite history to what was heard"]
    STOP -->|finished| FIN["Append assistant message<br/>retrieval signal if it answered from knowledge"]
    FIN --> DEC2{"Decide again"}
    DEC2 -->|hand off| HOFF3["Speak handoff line · build summary"]
    DEC2 -->|continue| DONE
    INT --> DONE
    HOFF1 --> DONE
    HOFF2 --> DONE
    HOFF3 --> DONE
    DONE(["7 · Persist turn · exactly one stop_reason"])
```

Every turn ends with exactly one `stop_reason`: `answered`, `handoff`,
`budget_exceeded`, `interrupted`, or `error`. The loop asserts on it.

The invariant above all others: **a failure becomes a handoff signal, never an
improvised answer.** Every caller-facing line on a failure path is a fixed template.

---

## 4. From model token to caller's ear

```mermaid
flowchart LR
    TOK["LLM text deltas"] --> SEG["Segmenter<br/>sentence and clause boundaries<br/>abbreviation and decimal aware"]
    SEG --> CIT["Strip [K] citations<br/>record cited chunks"]
    CIT --> CHK{"Grounding checker"}
    CHK -->|"Rule 1: every number, price, date,<br/>order id, email, venue grounded"| PASS
    CHK -->|"Rule 2: policy claim with nothing<br/>anchoring it and weak retrieval"| FAIL
    PASS["pass"] --> NORM["Spoken form<br/>$25 → twenty-five dollars<br/>TK-88213 → T K, eight eight..."]
    NORM --> SPEAK["Speak → TTS"]
    FAIL["fail"] --> SAFE["Fixed safe line<br/>+ handoff signal<br/>+ generation stops"]
```

Grounding sources, in order: the retrieved chunk text, tool results from this
turn, and everything the caller has said. Repeating the caller's own order
number back is not a hallucination. The checker never trusts the model's own
`[K]` citations; it checks against the chunk text.

**Declared blind spots**, measured and published rather than hidden:

- **Contradiction.** "Tickets are refundable" against a chunk that says non-refundable.
- **Binding.** The right number attached to the wrong fact.

A deterministic rule cannot see either.

Two first-audio clocks are kept. `first_speak` counts an acknowledgment line
("Let me check that for you"). `first_content` does not, and is the honest
headline.

---

## 5. Tools, read-back, and idempotency

### Read-back before any change to an order

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant H as Harness
    participant M as Model
    participant X as Executor
    participant L as Ledger (Postgres)
    participant B as Backend

    C->>H: "Send two tickets to friend@example.com"
    H->>M: request with tool schemas
    M-->>H: tool_call transfer_tickets(qty=2, ...)
    H->>X: prepare()
    X->>L: INSERT key (pending)
    H->>C: "Transferring two tickets for the Saturday show to friend at example dot com. Should I go ahead?"
    Note over H,C: rendered from the TEMPLATE using the validated arguments —<br/>the model never writes the confirmation sentence
    C->>H: "Yes"
    H->>X: dispatch()
    X->>L: UPDATE dispatched (written BEFORE the call)
    X->>B: transfer_tickets(..., idempotency_key)
    B-->>X: committed
    X->>L: UPDATE committed
    H->>M: action note: confirmed, result {...}
    M-->>H: "Done — they're on their way."
    H->>C: speaks it
```

### The execution state machine

```mermaid
stateDiagram-v2
    [*] --> pending: prepare() claims the key
    pending --> dispatched: written before the outbound call
    pending --> cancelled: caller said no / talked over the read-back
    dispatched --> committed: success
    dispatched --> failed: typed error — did not happen
    dispatched --> unknown: timeout — MAY have happened
    unknown --> committed: reconcile finds it
    unknown --> failed: reconcile finds nothing
    unknown --> unknown: reconcile itself fails → HARD handoff
    cancelled --> pending: same request again (attempt + 1)
    failed --> pending: retry (attempt + 1)
    committed --> committed: same request again → "already done", not repeated
```

**The one rule:** `cancelled` is reachable only from `pending`. Once a request has
left the process it is resolved or reconciled, never cancelled. Marking a
dispatched call "cancelled" on a barge-in is exactly how a duplicate charge gets
created.

**The enforcement point is the database, not this code.** A partial unique index
on `idempotency_key WHERE side_effecting` makes a duplicate lose to the index,
however it arrives. The key is deterministic, scoped to the call, and covers
only the fields that define the action's identity. Free text like a refund
reason is excluded, so a rephrased retry still collides.

`request_refund` has **no `amount` parameter**. The agent cannot promise money,
because the schema gives it nowhere to put a figure.

---

## 6. Barge-in

```mermaid
sequenceDiagram
    participant C as Caller
    participant T as Transport / TTS
    participant H as Harness

    H->>T: Speak "Refunds are available within thirty days..."
    T->>C: audio playing
    C->>T: starts talking
    T->>H: BargeIn(played_chars=25)
    H->>T: CancelSpeech  — FIRST, before any bookkeeping
    Note over H: spoken_text = approved text cut to what was heard,<br/>minus a playout margin, back to a word boundary
    H->>H: rewrite last assistant message to spoken_text
    H->>H: drop orphaned tool results · cancel PENDING actions only
    H->>H: assert the history invariant
```

Three texts are kept per turn, and conflating them is the classic bug:

| | |
|---|---|
| `generated_text` | everything the model wrote, including segments the checker rejected |
| `approved_text` | what passed the checker and went to TTS, in spoken form |
| `spoken_text` | what the caller actually heard, after a barge-in |

The estimate deliberately errs toward *"the caller heard less."* The agent may
repeat itself, but it will never believe it said something nobody heard.

A turn completes when its text reaches TTS, not when playback ends. So most
live barge-ins land *after* the turn finished generating, and
`_barge_in_after_turn` rewrites history for those too. Talking over a
read-back always cancels it.

---

## 7. Handoff

```mermaid
flowchart TD
    U["Utterance"] --> P{"Hard utterance pattern?"}
    P -->|yes| H1["HAND OFF — no model call"]
    P -->|no| S{"Hard state rule?<br/>verification failed twice · tool outcome unknown ·<br/>retrieval down · checker failed twice ·<br/>read-back unclear twice · step limit"}
    S -->|yes| H2["HAND OFF"]
    S -->|no| W["Soft score = Σ weight × 0.5^age"]
    W --> T{"score ≥ 0.6?"}
    T -->|yes| H3["HAND OFF"]
    T -->|no| A["Answer"]

    subgraph signals["Soft signals"]
        R1["retrieval_very_weak — top_vec below 0.62 → weight 0.60, can hand off alone"]
        R2["retrieval_insufficient — top_vec below 0.688 → weight 0.45"]
        O["checker_failure 0.40 · tool_failure 0.50 · rephrase 0.35 ·<br/>low-confidence entity 0.30 · budget twice 0.30"]
    end
    signals -.-> W
```

Both retrieval thresholds are **calibrated**, not guessed. They were chosen in
[`results/2026-10-13-retrieval-calibration/`](results/2026-10-13-retrieval-calibration/)
and [`results/2026-10-13-handoff-calibration/`](results/2026-10-13-handoff-calibration/).
The signal is the raw dense cosine of the best chunk. The originally planned
fused RRF margin scored AUC 0.53, a coin toss.

Weak retrieval only counts on turns that answered *from knowledge*. "Yes" or
"send two tickets" retrieves weakly by nature, and counting those would hand off
in the middle of a transfer that was going fine.

The human receives a summary built from call state: what the caller asked, what
the agent did and said, the relevant policy, a suggested next step, and the line
that matters most, **"The agent has NOT promised any refund, credit or other
money decision."**

---

## 8. Data model

```mermaid
erDiagram
    calls ||--o{ turns : has
    calls ||--o{ call_events : logs
    calls ||--o{ tool_executions : ledger
    calls ||--o{ checker_verdicts : checks
    calls ||--o{ handoffs : ends_in
    documents ||--o{ chunks : split_into

    calls {
        text call_id PK
        text source
        text outcome
        jsonb config_snapshot
    }
    turns {
        text turn_id PK
        text stop_reason
        text generated_text
        text approved_text
        text spoken_text
        real retrieval_top_vec_score
        int first_content_ms
        int voice_to_voice_ms
    }
    tool_executions {
        text idempotency_key "partial UNIQUE where side_effecting"
        text status
        int attempt
    }
    checker_verdicts {
        text verdict
        text seeded_class
    }
    chunks {
        text heading_path
        vector embedding "384-dim"
        tsvector tsv "generated"
    }
```

Every call stores a `config_snapshot` (providers, model ids, thresholds, git
sha, corpus hash). A number from two weeks ago is uninterpretable without it.
Full schema: [docs/03-data-model.md](docs/03-data-model.md) and
[`migrations/`](migrations/).

---

## 9. Evaluation and the feedback loop

```mermaid
flowchart LR
    subgraph data["Eval data"]
        CASES["94 labelled cases<br/>caller phrasing"]
        AUD["900 audio files<br/>3 voices · babble/pink/brown ·<br/>20/10/5 dB · 8 kHz mu-law"]
    end

    subgraph runners["Runners"]
        TD["Text driver + replay LLM<br/>every commit"]
        FI["Fault injection<br/>34 tests"]
        CAL["Calibration<br/>retrieval · handoff · checker"]
        STTE["STT benchmark"]
        JUDGE["LLM judge<br/>3 shuffled runs, median"]
    end

    subgraph out["Run directories"]
        RES["results/date-name/<br/>README · metrics.json · config.json"]
    end

    CASES --> TD & CAL & JUDGE
    AUD --> STTE
    TD --> FI
    CAL & STTE & FI & JUDGE --> RES

    subgraph loop["Week 4 — the feedback loop (make cycle)"]
        CSIM["Caller simulator<br/>300 calls · fixed seed"] --> PG[("Postgres call log")]
        PG --> GAP["Gap report<br/>actions ranked by troubled calls"]
        GAP --> FIX["Apply ONE change<br/>ingest a held-out document"]
        FIX -->|identical seeds| CSIM
        GAP -.->|after the fact| LABELS["Hidden labels<br/>score the report"]
    end
    RES -.-> loop
```

Three of the twelve knowledge-base documents are **held out** at baseline. The
week-4 gap report has to *discover* which missing topic is costing calls. That
makes the before/after containment number earned rather than staged.

It found all three, with the right diagnosis, at ranks 2, 4 and 5. The two actions
above them were real problems nobody planted: refund questions retrieving the
transfers document, and vague callers. Applying the top action that needed no
human author (ingest the weather document) moved containment from **0.737 to
0.793** on identical seeds, with the wrong-answer rate down, not up
([`results/2026-10-15-feedback-cycle/`](results/2026-10-15-feedback-cycle/)).

---

## 10. Latency budget

```mermaid
flowchart LR
    A["A · VAD endpointing<br/>400 ms — a SETTING"] --> B["B · STT final<br/>80"] --> C["C · retrieval<br/>40"] --> D["D · LLM first token<br/>400"] --> E["E · first complete segment<br/>250"] --> F["F · checker<br/>5"] --> G["G · TTS first byte<br/>150"] --> H["H · playout<br/>50"]
```

| Reported as | Stages | p50 target | p95 target |
|---|---|---|---|
| voice-to-voice | A–H | ≤ 1375 ms | ≤ 2100 ms |
| harness-side | B–H | ≤ 975 ms | ≤ 1600 ms |

Endpointing is always reported separately, because lowering it buys a better
number while making the agent interrupt people. Stage E is the cost the
checker's existence imposes. It is measured on its own and never folded into
the LLM stage.

No live voice latency has been measured yet; that needs API keys. The static
report refuses to render latency from any run that is not `timing_valid`. See §11.

---

## 11. Measured results

All from [`results/`](results/). *n* is small throughout, and each run README
says so. `make report` renders all of this, with provenance, into
`reports/index.html`.

| What | Result |
|---|---|
| Duplicate side effects under fault injection | **0** across 34 fault tests, including commit-then-timeout, concurrent duplicates and interruption mid-read-back |
| Concurrent identical transfers, real Postgres | **5 → 1** effect, 1 committed row (the race this exposed is fixed; WEEK4.md) |
| Handoff recall / precision (deterministic policy) | **0.86** [0.72, 0.93] / **0.82** [0.68, 0.90], up from recall 0.31 before calibration |
| Hard-rule handoffs | **0** LLM calls |
| Checker: invented numbers / dates / entities | **23/23 · 5/5 · 4/4** caught |
| Checker: fabricated policy claims | **7/8** caught |
| Checker false positives | **2/46** |
| Checker latency | p95 **0.07 ms** per sentence |
| Retrieval sufficiency signal | dense cosine AUC **0.93** vs fused RRF margin **0.51** |
| Retrieval, right document top-1 / top-4 | **11/14 · 14/14** on caller phrasing |
| STT key-entity accuracy, clean → 5 dB | **0.83 → 0.62** (WER only 0.025 → 0.052); spoken emails **0.67 → 0.43** |
| Feedback cycle, containment *(300 simulated calls, offline agent)* | **0.737 → 0.793**, paired +17 / −0, McNemar p = 1.5 × 10⁻⁵ |
| Feedback cycle, wrong-answer rate (guardrail) | **0.136 → 0.113** |
| Gap report vs hidden labels | 3/3 planted gaps found (ranks 2, 4, 5), 1/3 in the top 3, cluster purity 0.985 |

**Not yet measured:** live voice latency, the Deepgram comparison, answer
correctness via the judge, handoffs triggered by a live model's own answers, and
the feedback cycle with Claude as the agent. All of them need API keys.

---

## 12. Running it

```bash
make venv                 # .venv on Python 3.12
cp .env.example .env      # add ANTHROPIC_API_KEY for live chat
make up                   # Postgres :5434, Jaeger UI :16687
make retrieval && make ingest
```

```bash
make test                 # 366 tests, ~20 s, no network
make faults               # fault-injection suite
make demo-scripts         # four scripted demo beats, recorded to Postgres
make dev                  # web app at http://localhost:8088/ · voice client /voice · console /console
make chat                 # text-mode REPL against the full stack
```

```bash
make simulate N=300       # simulated callers → results/<date>-simulation-<label>/
make gap-report           # gap report on the latest simulation run
make cycle                # baseline → gap report → ONE change → identical seeds
make report               # reports/index.html from every results/*/metrics.json
```

With no `ANTHROPIC_API_KEY`, the web app, the voice pipeline and the simulator
all fall back to the offline extractive agent, and say so in a banner and in
every result.

Ports avoid common collisions: 5434, 8088, 16687 and 4319 rather than 5432,
8000, 16686 and 4317. All are overridable in `.env`.

---

## 13. Repository map

```
app/            FastAPI, /ws, Pipecat wiring (the only pipecat import), console, web app (/ws/chat, insights API)
harness/        turn loop, state, checker, normalization, handoff, tools, read-back, assembly
providers/      LLM (Claude, OpenAI, replay, extractive) · STT (Deepgram, faster-whisper, null) · TTS (Cartesia, null)
retrieval/      chunking, corpus linter, embeddings, ingestion, hybrid search
analytics/      event writer (closed vocabulary), persistence, gap report, static report
core/           settings, ids, clock, logging, db, tracing — depends on nothing of ours
domains/        ticketing: 12-document corpus, tools.yaml, handoff.yaml, prompts.yaml
eval/           text driver, dataset, metrics, calibration, checker seeding, synth,
                noise, telephony, STT benchmark, judge, fault-injection suite,
                caller simulator, feedback cycle
demo/           scripted demo beats
migrations/     numbered SQL
results/        one directory per measured run
reports/        the rendered static report
tests/          unit, integration, and architecture-contract tests
docs/           design documents 00–17
```

---

## 14. Documents

| File | What it covers |
|---|---|
| [BUILD.md](BUILD.md) | The week-by-week plan: every task, its dependencies and acceptance criteria |
| [WEEK1.md](WEEK1.md) · [WEEK2.md](WEEK2.md) · [WEEK3.md](WEEK3.md) · [WEEK4.md](WEEK4.md) | What was built, what was verified, what was not |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | One-page summary and resume bullets backed by measurements |
| [docs/00-decisions.md](docs/00-decisions.md) | Every locked decision, with rationale and reversal cost |
| [docs/01-architecture.md](docs/01-architecture.md) | Component boundaries, turn data flow, latency budget |
| [docs/02-repo-and-tooling.md](docs/02-repo-and-tooling.md) | Repo layout, dependencies, Docker, Make targets, env vars |
| [docs/03-data-model.md](docs/03-data-model.md) | Complete Postgres schema |
| [docs/04-providers.md](docs/04-providers.md) | STT / TTS / LLM interfaces and implementations |
| [docs/05-retrieval.md](docs/05-retrieval.md) | Corpus authoring, chunking, hybrid search, calibration |
| [docs/06-harness.md](docs/06-harness.md) | State, the turn loop, stop rules, context hygiene |
| [docs/07-tools.md](docs/07-tools.md) | Tool registry, idempotency protocol, read-back |
| [docs/08-checker.md](docs/08-checker.md) | Grounding checker and spoken-form normalization |
| [docs/09-handoff.md](docs/09-handoff.md) | Hard rules, soft scoring, the handoff summary |
| [docs/10-audio-pipeline.md](docs/10-audio-pipeline.md) | Pipecat, endpointing, barge-in, WebSocket protocol |
| [docs/11-observability.md](docs/11-observability.md) | OTel span tree and metric definitions |
| [docs/12-eval.md](docs/12-eval.md) | Eval set, audio, replay, judge, fault injection |
| [docs/13-caller-sim.md](docs/13-caller-sim.md) | The synthetic caller population for week 4 |
| [docs/14-analytics.md](docs/14-analytics.md) | Gap report and the feedback-cycle protocol |
| [docs/15-results-and-claims.md](docs/15-results-and-claims.md) | Reporting rules and what the project can and cannot claim |
| [docs/16-console.md](docs/16-console.md) | The live observability console |
| [docs/17-frontend.md](docs/17-frontend.md) | The web app: views, endpoints, chat protocol, limits |

## The one rule

Never report a number without the run directory that produced it.
