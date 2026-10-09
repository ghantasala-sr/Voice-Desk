-- 003_eval.sql — the eval record (used from week 3).

-- ---------------------------------------------------------------------------
-- eval_runs — one row per `make eval`. Mirrors results/<date>-<label>/.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
  run_id          TEXT PRIMARY KEY,
  label           TEXT NOT NULL,
  suite           TEXT NOT NULL,      -- all | clean | noise | telephony | faults
  git_sha         TEXT NOT NULL,
  config_snapshot JSONB NOT NULL,

  -- Hash over the sorted (path, content_hash) of INGESTED documents only.
  -- This is how a before/after comparison proves the corpus actually changed.
  corpus_hash     TEXT NOT NULL,

  started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at        TIMESTAMPTZ,
  metrics         JSONB,              -- mirrors metrics.json
  results_dir     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS eval_runs_label_idx ON eval_runs (label, started_at DESC);

-- ---------------------------------------------------------------------------
-- eval_results — one row per (case, variant, voice).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_results (
  result_id         TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
  case_id           TEXT NOT NULL,
  variant           TEXT NOT NULL,    -- clean | snr20 | snr10 | snr5 | telephony_*
  noise_type        TEXT,             -- cafe | street | babble
  voice_id          TEXT,             -- synth voice id, or 'human'
  call_id           TEXT REFERENCES calls(call_id),

  -- STT
  reference_text    TEXT NOT NULL,
  hypothesis_text   TEXT,
  wer               REAL,
  -- Entity accuracy is per ENTITY, not per utterance. Per-utterance scoring
  -- hides that one utterance had three entities and two were wrong, which is
  -- the common case at 5 dB.
  entity_total      INTEGER,
  entity_correct    INTEGER,

  -- Answer quality — median of 3 judge runs; all 3 kept in judge_runs.
  judge_correctness INTEGER CHECK (judge_correctness BETWEEN 0 AND 2),
  judge_citation    INTEGER CHECK (judge_citation BETWEEN 0 AND 2),
  judge_safety      INTEGER CHECK (judge_safety BETWEEN 0 AND 2),
  judge_runs        JSONB,
  judge_rationale   TEXT,

  -- Handoff: scored mechanically against the label, never judged.
  expected_escalate BOOLEAN,
  actual_escalate   BOOLEAN,

  passed            BOOLEAN,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, case_id, variant, voice_id)
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx  ON eval_results (run_id, variant);
CREATE INDEX IF NOT EXISTS eval_results_case_idx ON eval_results (case_id);

-- ---------------------------------------------------------------------------
-- gap_reports — week 4, docs/14 §3
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gap_reports (
  report_id    TEXT PRIMARY KEY,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  window_start TIMESTAMPTZ NOT NULL,
  window_end   TIMESTAMPTZ NOT NULL,
  call_count   INTEGER NOT NULL,
  clusters     JSONB NOT NULL,
  actions      JSONB NOT NULL
);
