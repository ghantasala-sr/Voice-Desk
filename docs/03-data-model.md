# 03 — Data model

One Postgres instance holds three logically separate things: the **knowledge base**, the
**call record**, and the **eval record**. They share a database so a gap report can join
call failures against corpus coverage in one query.

Migrations are numbered SQL files applied in order. No ORM — see
[02-repo-and-tooling.md](02-repo-and-tooling.md) §2.

---

## 1. Conventions

- Primary keys are `TEXT` holding a ULID (sortable, generated in Python). Not UUIDv4 —
  sortability matters because almost every query is "the most recent N."
- Every table has `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`.
- Durations are `_ms INTEGER`. Never floats, never seconds.
- Anything with an open-ended shape is `JSONB`; anything you will filter, group, or
  threshold on is a real column. The dividing line is: *will a metric query touch it?*
- Enum-like values are `TEXT` with a `CHECK` constraint, not Postgres `ENUM` types —
  adding a value to a Postgres enum inside a transaction is a nuisance and these lists
  will change.

---

## 2. `001_init.sql` — call record

### `calls`

One row per call, written at call start and updated at call end. Flushed incrementally so
a dropped WebSocket does not lose the call.

```sql
CREATE TABLE calls (
  call_id          TEXT PRIMARY KEY,
  domain_config    TEXT NOT NULL,
  source           TEXT NOT NULL
                     CHECK (source IN ('browser','phone','replay','simulation','text')),
  caller_verified  BOOLEAN NOT NULL DEFAULT FALSE,
  verified_at      TIMESTAMPTZ,
  order_id         TEXT,                      -- populated after verify_caller
  outcome          TEXT
                     CHECK (outcome IN ('contained','handoff','dropped','error')),
  handoff_reason   TEXT,
  rephrase_count   INTEGER NOT NULL DEFAULT 0,
  turn_count       INTEGER NOT NULL DEFAULT 0,
  started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at         TIMESTAMPTZ,
  duration_ms      INTEGER,
  -- provenance: which code and config produced this call
  config_snapshot  JSONB NOT NULL,            -- providers, model ids, thresholds, git sha
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX calls_outcome_idx   ON calls (outcome, started_at DESC);
CREATE INDEX calls_domain_idx    ON calls (domain_config, started_at DESC);
```

`config_snapshot` is what makes results reproducible. Without it a containment number from
two weeks ago is uninterpretable. It is required, not nullable.

### `turns`

One row per `TurnState`. This is the table most metrics are computed from, so the latency
stages are real columns, not JSONB.

```sql
CREATE TABLE turns (
  turn_id              TEXT PRIMARY KEY,
  call_id              TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  seq                  INTEGER NOT NULL,
  trigger              TEXT NOT NULL
                         CHECK (trigger IN ('transcript','barge_in','silence','tool_result')),

  user_text            TEXT NOT NULL DEFAULT '',
  stt_confidence       REAL,
  stt_provider         TEXT,

  retrieved_chunk_ids  TEXT[] NOT NULL DEFAULT '{}',
  retrieval_top_score  REAL,
  retrieval_margin     REAL,          -- top1 - top2 fused score; see docs/05 §6

  llm_iterations       INTEGER NOT NULL DEFAULT 0,
  generated_text       TEXT NOT NULL DEFAULT '',
  spoken_text          TEXT NOT NULL DEFAULT '',   -- what the caller actually heard

  stop_reason          TEXT
                         CHECK (stop_reason IN
                           ('answered','handoff','budget_exceeded','interrupted','error')),
  handoff_signals      TEXT[] NOT NULL DEFAULT '{}',
  handoff_score        REAL,

  -- latency stages, docs/01 §5. NULL means the stage did not run.
  endpoint_ms          INTEGER,   -- [A] VAD silence window actually waited
  stt_final_ms         INTEGER,   -- [B]
  retrieval_ms         INTEGER,   -- [C]
  llm_ttft_ms          INTEGER,   -- [D]
  sentence_buffer_ms   INTEGER,   -- [E] first approved segment completion
  checker_ms           INTEGER,   -- [F]
  tts_ttfb_ms          INTEGER,   -- [G]
  playout_ms           INTEGER,   -- [H]
  first_audio_ms       INTEGER,   -- harness-side total, B..H
  voice_to_voice_ms    INTEGER,   -- A..H
  ack_bypass_used      BOOLEAN NOT NULL DEFAULT FALSE,   -- docs/01 §5 note

  trace_id             TEXT,      -- links to Jaeger
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (call_id, seq)
);

CREATE INDEX turns_call_idx  ON turns (call_id, seq);
CREATE INDEX turns_stop_idx  ON turns (stop_reason);
-- latency percentile queries scan this; keep it covering
CREATE INDEX turns_latency_idx ON turns (created_at DESC)
  WHERE voice_to_voice_ms IS NOT NULL;
```

### `call_events`

Append-only narrative log. Everything the harness does emits one. This is the raw material
for trace replay and for debugging a call after the fact.

```sql
CREATE TABLE call_events (
  event_id    BIGSERIAL PRIMARY KEY,
  call_id     TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id     TEXT,
  seq         INTEGER NOT NULL,
  event_type  TEXT NOT NULL,
  payload     JSONB NOT NULL DEFAULT '{}',
  ts          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX call_events_call_idx ON call_events (call_id, seq);
CREATE INDEX call_events_type_idx ON call_events (event_type, ts DESC);
```

`event_type` vocabulary (closed set, asserted in `analytics/events.py`):

```
call_started            turn_started           retrieval_completed
stt_final               simulation_label       llm_iteration_started
llm_tool_call           llm_text_completed     checker_verdict
readback_spoken         readback_confirmed     readback_rejected
tool_dispatched         tool_committed         tool_failed
tool_cancelled          tool_unknown           tool_reconciled
filler_spoken           barge_in               context_trimmed
handoff_signal_raised   handoff_triggered      budget_exceeded
turn_completed          call_ended             error
silence_reprompt
```

`stt_partial` was in the original vocabulary and was removed in week 4 by the
event-completeness audit (W4-2): partial transcripts stay inside Pipecat and never
reach the harness, so nothing could emit it. A vocabulary entry that cannot occur
is a query that silently returns nothing.

### `tool_executions`

The idempotency ledger. Protocol in [07-tools.md](07-tools.md) §3.

```sql
CREATE TABLE tool_executions (
  execution_id     TEXT PRIMARY KEY,
  call_id          TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id          TEXT,
  tool_name        TEXT NOT NULL,
  side_effecting   BOOLEAN NOT NULL,

  idempotency_key  TEXT NOT NULL,
  args_canonical   JSONB NOT NULL,     -- canonicalized args the key was derived from

  status           TEXT NOT NULL
                     CHECK (status IN
                       ('pending','dispatched','committed','failed','cancelled','unknown')),
  attempt          INTEGER NOT NULL DEFAULT 1,

  request_payload  JSONB,
  response_payload JSONB,
  error_type       TEXT,               -- timeout | malformed | upstream_error | ...
  error_detail     TEXT,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  dispatched_at    TIMESTAMPTZ,
  resolved_at      TIMESTAMPTZ,
  duration_ms      INTEGER
);

-- THE constraint that makes duplicate side effects impossible.
CREATE UNIQUE INDEX tool_exec_idem_idx
  ON tool_executions (idempotency_key)
  WHERE side_effecting;

CREATE INDEX tool_exec_call_idx   ON tool_executions (call_id, created_at);
CREATE INDEX tool_exec_status_idx ON tool_executions (status)
  WHERE status IN ('unknown','pending','dispatched');
```

The partial unique index is the enforcement point for the "zero duplicate side effects"
claim. It is a database constraint, not application logic — an interrupted, retried, or
concurrently-issued duplicate loses to the index, deterministically.

### `handoffs`

```sql
CREATE TABLE handoffs (
  handoff_id    TEXT PRIMARY KEY,
  call_id       TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id       TEXT,
  reason        TEXT NOT NULL,              -- the dominant reason
  rule_type     TEXT NOT NULL CHECK (rule_type IN ('hard','soft')),
  signals       JSONB NOT NULL,             -- [{signal, weight, value}]
  score         REAL,
  summary       JSONB NOT NULL,             -- docs/09 §4 schema
  summary_text  TEXT NOT NULL,              -- rendered for the human agent
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### `checker_verdicts`

Separate table, not a JSONB column on `turns`, because catch rate is computed **per
verdict** and grouped by hallucination class.

```sql
CREATE TABLE checker_verdicts (
  verdict_id    BIGSERIAL PRIMARY KEY,
  call_id       TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id       TEXT NOT NULL,
  segment_index INTEGER NOT NULL,
  segment_text  TEXT NOT NULL,
  verdict       TEXT NOT NULL
                  CHECK (verdict IN
                    ('pass','no_checkable_content',
                     'fail_ungrounded_entity','fail_unsupported_claim')),
  checkable_spans JSONB NOT NULL DEFAULT '[]',  -- [{type, surface, normalized, grounded}]
  action_taken  TEXT NOT NULL
                  CHECK (action_taken IN ('released','replaced','handoff')),
  checker_ms    INTEGER,
  -- eval only: was this segment deliberately corrupted, and how
  seeded_class  TEXT,    -- NULL | numeric | date | entity | policy_claim | citation
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX checker_verdicts_turn_idx    ON checker_verdicts (turn_id);
CREATE INDEX checker_verdicts_seeded_idx  ON checker_verdicts (seeded_class)
  WHERE seeded_class IS NOT NULL;
```

`seeded_class` is what makes [00-decisions.md](00-decisions.md) D10 reportable: catch rate
grouped by `seeded_class` shows the covered classes and the uncovered one in the same table.

---

## 3. `002_corpus.sql` — knowledge base

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  doc_id        TEXT PRIMARY KEY,
  domain_config TEXT NOT NULL,
  path          TEXT NOT NULL,
  title         TEXT NOT NULL,
  topics        TEXT[] NOT NULL DEFAULT '{}',
  content_hash  TEXT NOT NULL,
  ingested      BOOLEAN NOT NULL DEFAULT TRUE,   -- FALSE = held out, docs/00 D4
  ingested_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (domain_config, path)
);

CREATE TABLE chunks (
  chunk_id      TEXT PRIMARY KEY,
  doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  domain_config TEXT NOT NULL,
  seq           INTEGER NOT NULL,
  heading_path  TEXT NOT NULL,        -- "Refunds > Event cancelled by organizer"
  text          TEXT NOT NULL,        -- the chunk body, 60-120 words
  embed_text    TEXT NOT NULL,        -- heading_path + text, what was embedded
  word_count    INTEGER NOT NULL,
  embedding     vector(384) NOT NULL, -- BAAI/bge-small-en-v1.5
  tsv           tsvector GENERATED ALWAYS AS
                  (to_tsvector('english', coalesce(heading_path,'') || ' ' || text)) STORED,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (doc_id, seq)
);

CREATE INDEX chunks_tsv_idx    ON chunks USING GIN (tsv);
CREATE INDEX chunks_vec_idx    ON chunks USING hnsw (embedding vector_cosine_ops)
                                  WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_domain_idx ON chunks (domain_config);
```

Notes.

- `embedding` dimension is tied to the model in `VD_EMBED_MODEL`. Changing the model is a
  migration, not a config flip. `ingest.py` asserts the dimension matches on startup.
- **HNSW over a few hundred rows is pointless for recall** but costs nothing and keeps the
  query identical if the corpus grows. Set `hnsw.ef_search` high (or rely on the exact scan
  Postgres picks for tiny tables) so recall is not silently capped during calibration.
- `ingested = FALSE` rows stay in `documents` with no `chunks`. This is deliberate: the gap
  report can then say "a document exists for this topic but is not indexed," which is a
  different finding from "no document covers this topic."

---

## 4. `003_eval.sql` — eval record

```sql
CREATE TABLE eval_runs (
  run_id        TEXT PRIMARY KEY,
  label         TEXT NOT NULL,
  suite         TEXT NOT NULL,             -- all | clean | noise | telephony | faults
  git_sha       TEXT NOT NULL,
  config_snapshot JSONB NOT NULL,
  corpus_hash   TEXT NOT NULL,             -- hash of ingested doc set, docs/14 §5
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at      TIMESTAMPTZ,
  metrics       JSONB,                     -- the computed summary, mirrors metrics.json
  results_dir   TEXT NOT NULL              -- results/<date>-<slug>/
);

CREATE TABLE eval_results (
  result_id       TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
  case_id         TEXT NOT NULL,
  variant         TEXT NOT NULL,           -- clean | snr20 | snr10 | snr5 | telephony
  noise_type      TEXT,                    -- cafe | street | babble
  voice_id        TEXT,                    -- which synth voice, or 'human'
  call_id         TEXT REFERENCES calls(call_id),

  -- STT
  reference_text  TEXT NOT NULL,
  hypothesis_text TEXT,
  wer             REAL,
  entity_total    INTEGER,
  entity_correct  INTEGER,

  -- answer
  judge_correctness INTEGER CHECK (judge_correctness BETWEEN 0 AND 2),
  judge_citation    INTEGER CHECK (judge_citation BETWEEN 0 AND 2),
  judge_safety      INTEGER CHECK (judge_safety BETWEEN 0 AND 2),
  judge_runs        JSONB,                 -- all 3 runs, docs/12 §6
  judge_rationale   TEXT,

  -- handoff
  expected_escalate BOOLEAN,
  actual_escalate   BOOLEAN,

  passed          BOOLEAN,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, case_id, variant, voice_id)
);

CREATE INDEX eval_results_run_idx ON eval_results (run_id, variant);
```

```sql
CREATE TABLE gap_reports (
  report_id    TEXT PRIMARY KEY,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  window_start TIMESTAMPTZ NOT NULL,
  window_end   TIMESTAMPTZ NOT NULL,
  call_count   INTEGER NOT NULL,
  clusters     JSONB NOT NULL,   -- docs/14 §3 schema
  actions      JSONB NOT NULL    -- ranked suggested fixes
);
```

---

## 5. Queries the schema exists to serve

These are the reason for every denormalized column above. If one of them needs JSONB
extraction, a column is missing.

**Latency percentiles with sample size** ([00-decisions.md](00-decisions.md) D12):

```sql
SELECT count(*) AS n,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY voice_to_voice_ms) AS v2v_p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY voice_to_voice_ms) AS v2v_p95,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY first_audio_ms)    AS harness_p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY first_audio_ms)    AS harness_p95
FROM turns t JOIN calls c USING (call_id)
WHERE c.source = 'replay' AND t.voice_to_voice_ms IS NOT NULL;
```

**Per-stage breakdown** (which stage to optimize):

```sql
SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY endpoint_ms)        AS p50_endpoint,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY llm_ttft_ms)        AS p50_llm_ttft,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY sentence_buffer_ms) AS p50_buffer,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY tts_ttfb_ms)        AS p50_tts
FROM turns WHERE first_audio_ms IS NOT NULL;
```

**Duplicate side effects — must always return zero:**

```sql
SELECT idempotency_key, count(*)
FROM tool_executions
WHERE side_effecting AND status = 'committed'
GROUP BY 1 HAVING count(*) > 1;
```

**Checker catch rate by hallucination class** ([00-decisions.md](00-decisions.md) D10):

```sql
SELECT seeded_class,
       count(*) AS seeded,
       count(*) FILTER (WHERE verdict LIKE 'fail%') AS caught,
       round(100.0 * count(*) FILTER (WHERE verdict LIKE 'fail%') / count(*), 1) AS catch_pct
FROM checker_verdicts
WHERE seeded_class IS NOT NULL
GROUP BY 1 ORDER BY 1;
```

**Containment, before and after a feedback cycle:**

```sql
SELECT c.config_snapshot->>'corpus_hash' AS corpus,
       count(*) AS calls,
       round(100.0 * count(*) FILTER (WHERE outcome = 'contained') / count(*), 1) AS containment_pct
FROM calls c
WHERE c.source = 'simulation'
GROUP BY 1 ORDER BY min(started_at);
```

**Stop-reason distribution** (the harness health check):

```sql
SELECT stop_reason, count(*),
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM turns GROUP BY 1 ORDER BY 2 DESC;
```
