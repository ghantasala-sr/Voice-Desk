-- 001_init.sql — the call record.
--
-- Conventions (docs/03-data-model.md §1):
--   * TEXT primary keys holding ULIDs. Sortable, generated in Python.
--   * Durations are _ms INTEGER. Never floats, never seconds.
--   * Enum-like values are TEXT + CHECK, not Postgres ENUM types.
--   * A column exists when a metric query touches it. Everything else is JSONB.
--
-- Idempotent: safe to re-run via scripts/migrate.sh.

-- ---------------------------------------------------------------------------
-- calls — one row per call. Written at start, updated incrementally so a
-- dropped WebSocket never loses the call.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS calls (
  call_id          TEXT PRIMARY KEY,
  domain_config    TEXT NOT NULL,
  source           TEXT NOT NULL
                     CHECK (source IN ('browser','phone','replay','simulation','text')),
  caller_verified  BOOLEAN NOT NULL DEFAULT FALSE,
  verified_at      TIMESTAMPTZ,
  order_id         TEXT,
  outcome          TEXT
                     CHECK (outcome IN ('contained','handoff','dropped','error')),
  handoff_reason   TEXT,
  rephrase_count   INTEGER NOT NULL DEFAULT 0,
  turn_count       INTEGER NOT NULL DEFAULT 0,
  started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at         TIMESTAMPTZ,
  duration_ms      INTEGER,

  -- Providers, model ids, thresholds, git sha, corpus hash. REQUIRED, not
  -- nullable: a containment number from two weeks ago is uninterpretable
  -- without the config that produced it. See docs/15.
  config_snapshot  JSONB NOT NULL,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS calls_outcome_idx ON calls (outcome, started_at DESC);
CREATE INDEX IF NOT EXISTS calls_domain_idx  ON calls (domain_config, started_at DESC);
CREATE INDEX IF NOT EXISTS calls_source_idx  ON calls (source, started_at DESC);

-- ---------------------------------------------------------------------------
-- turns — one row per TurnState. Most metrics come from here, so the latency
-- stages are real columns rather than JSONB.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
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
  retrieval_margin     REAL,

  llm_iterations       INTEGER NOT NULL DEFAULT 0,

  -- Three distinct texts. generated ⊇ approved ⊇ spoken. Conflating them is
  -- the classic barge-in bug. See docs/06-harness.md §1.
  generated_text       TEXT NOT NULL DEFAULT '',
  approved_text        TEXT NOT NULL DEFAULT '',
  spoken_text          TEXT NOT NULL DEFAULT '',

  stop_reason          TEXT
                         CHECK (stop_reason IN
                           ('answered','handoff','budget_exceeded','interrupted','error')),
  handoff_signals      TEXT[] NOT NULL DEFAULT '{}',
  handoff_score        REAL,

  -- Latency stages [A]..[H] from docs/01-architecture.md §5.
  -- NULL means the stage did not run on this turn.
  endpoint_ms          INTEGER,   -- [A] VAD silence actually waited
  stt_final_ms         INTEGER,   -- [B]
  retrieval_ms         INTEGER,   -- [C]
  llm_ttft_ms          INTEGER,   -- [D]
  sentence_buffer_ms   INTEGER,   -- [E] wait for the first complete segment
  checker_ms           INTEGER,   -- [F]
  tts_ttfb_ms          INTEGER,   -- [G]
  playout_ms           INTEGER,   -- [H]
  first_audio_ms       INTEGER,   -- harness-side total, B..H
  voice_to_voice_ms    INTEGER,   -- A..H
  ack_bypass_used      BOOLEAN NOT NULL DEFAULT FALSE,

  trace_id             TEXT,      -- join key into Jaeger
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (call_id, seq)
);

CREATE INDEX IF NOT EXISTS turns_call_idx ON turns (call_id, seq);
CREATE INDEX IF NOT EXISTS turns_stop_idx ON turns (stop_reason);
CREATE INDEX IF NOT EXISTS turns_latency_idx ON turns (created_at DESC)
  WHERE voice_to_voice_ms IS NOT NULL;

-- ---------------------------------------------------------------------------
-- call_events — append-only narrative log. Traces expire; this does not.
-- Every eval failure must be replayable from Postgres alone, months later.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS call_events (
  event_id    BIGSERIAL PRIMARY KEY,
  call_id     TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id     TEXT,
  seq         INTEGER NOT NULL,
  event_type  TEXT NOT NULL,
  payload     JSONB NOT NULL DEFAULT '{}',
  ts          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS call_events_call_idx ON call_events (call_id, seq);
CREATE INDEX IF NOT EXISTS call_events_type_idx ON call_events (event_type, ts DESC);

-- ---------------------------------------------------------------------------
-- tool_executions — the idempotency ledger (week 2, docs/07 §3).
-- Created now so the schema is complete and the week-2 work is a code change
-- rather than a migration plus a code change.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tool_executions (
  execution_id     TEXT PRIMARY KEY,
  call_id          TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id          TEXT,
  tool_name        TEXT NOT NULL,
  side_effecting   BOOLEAN NOT NULL,

  idempotency_key  TEXT NOT NULL,
  args_canonical   JSONB NOT NULL,

  status           TEXT NOT NULL
                     CHECK (status IN
                       ('pending','dispatched','committed','failed','cancelled','unknown')),
  attempt          INTEGER NOT NULL DEFAULT 1,

  request_payload  JSONB,
  response_payload JSONB,
  error_type       TEXT,
  error_detail     TEXT,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  dispatched_at    TIMESTAMPTZ,
  resolved_at      TIMESTAMPTZ,
  duration_ms      INTEGER
);

-- THE constraint behind "zero duplicate side effects". A duplicate — from a
-- retry, an interruption, or a concurrent issue — loses to this index
-- deterministically. It is a database guarantee, not application logic.
CREATE UNIQUE INDEX IF NOT EXISTS tool_exec_idem_idx
  ON tool_executions (idempotency_key)
  WHERE side_effecting;

CREATE INDEX IF NOT EXISTS tool_exec_call_idx ON tool_executions (call_id, created_at);
CREATE INDEX IF NOT EXISTS tool_exec_status_idx ON tool_executions (status)
  WHERE status IN ('unknown','pending','dispatched');

-- ---------------------------------------------------------------------------
-- handoffs (week 2, docs/09 §4)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS handoffs (
  handoff_id    TEXT PRIMARY KEY,
  call_id       TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id       TEXT,
  reason        TEXT NOT NULL,
  rule_type     TEXT NOT NULL CHECK (rule_type IN ('hard','soft')),
  signals       JSONB NOT NULL,
  score         REAL,
  summary       JSONB NOT NULL,
  summary_text  TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS handoffs_call_idx ON handoffs (call_id);

-- ---------------------------------------------------------------------------
-- checker_verdicts (week 2, docs/08 §6)
--
-- A separate table, not a JSONB column on turns, because catch rate is
-- computed per verdict and grouped by hallucination class.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS checker_verdicts (
  verdict_id      BIGSERIAL PRIMARY KEY,
  call_id         TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  turn_id         TEXT NOT NULL,
  segment_index   INTEGER NOT NULL,
  segment_text    TEXT NOT NULL,
  verdict         TEXT NOT NULL
                    CHECK (verdict IN
                      ('pass','no_checkable_content',
                       'fail_ungrounded_entity','fail_unsupported_claim')),
  checkable_spans JSONB NOT NULL DEFAULT '[]',
  action_taken    TEXT NOT NULL
                    CHECK (action_taken IN ('released','replaced','handoff')),
  checker_ms      INTEGER,

  -- Eval only: was this segment deliberately corrupted, and how. This is what
  -- makes catch rate reportable BY CLASS, including the class we do not cover.
  seeded_class    TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS checker_verdicts_turn_idx ON checker_verdicts (turn_id);
CREATE INDEX IF NOT EXISTS checker_verdicts_seeded_idx ON checker_verdicts (seeded_class)
  WHERE seeded_class IS NOT NULL;
