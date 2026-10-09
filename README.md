# VoiceDesk

A self-improving real-time voice support agent. Answers support calls grounded in a
document set, acts through tools, hands off to a human with a structured summary when
it isn't confident, and mines its own call history to find what it got wrong.

The engine is domain-agnostic. The reference deployment is Tier-1 phone support for a
fictional event ticketing company.

## Documents

| File | What it covers |
|---|---|
| [BUILD.md](BUILD.md) | The week-by-week build plan: every task, its dependencies, and its acceptance criteria. **Start here.** |
| [docs/00-decisions.md](docs/00-decisions.md) | Every locked decision, with rationale and reversal cost |
| [docs/01-architecture.md](docs/01-architecture.md) | Component boundaries, turn data flow, latency budget |
| [docs/02-repo-and-tooling.md](docs/02-repo-and-tooling.md) | Repo layout, dependencies, Docker, Make targets, env vars |
| [docs/03-data-model.md](docs/03-data-model.md) | Complete Postgres schema |
| [docs/04-providers.md](docs/04-providers.md) | STT / TTS / LLM provider interfaces and implementations |
| [docs/05-retrieval.md](docs/05-retrieval.md) | Corpus authoring, chunking, hybrid search, threshold calibration |
| [docs/06-harness.md](docs/06-harness.md) | State objects, the turn loop, stop rules, context hygiene |
| [docs/07-tools.md](docs/07-tools.md) | Tool registry, idempotency protocol, read-back protocol |
| [docs/08-checker.md](docs/08-checker.md) | Sentence-level grounding checker and spoken-form normalization |
| [docs/09-handoff.md](docs/09-handoff.md) | Hard rules, soft-signal scoring, handoff summary schema |
| [docs/10-audio-pipeline.md](docs/10-audio-pipeline.md) | Pipecat wiring, endpointing, barge-in, WebSocket protocol, browser client |
| [docs/11-observability.md](docs/11-observability.md) | OTel span tree, attributes, metric definitions |
| [docs/12-eval.md](docs/12-eval.md) | Eval dataset, audio synthesis, noise mixing, replay, judge, fault injection |
| [docs/13-caller-sim.md](docs/13-caller-sim.md) | Synthetic caller population that feeds the feedback loop |
| [docs/14-analytics.md](docs/14-analytics.md) | Call events, gap report, static report, feedback-cycle protocol |
| [docs/15-results-and-claims.md](docs/15-results-and-claims.md) | Results file format, reporting rules, resume bullets |

## The one rule

Never report a number without the run directory that produced it. See
[docs/15-results-and-claims.md](docs/15-results-and-claims.md).
