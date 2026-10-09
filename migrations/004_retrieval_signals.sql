-- 004_retrieval_signals.sql
--
-- The fused RRF score and its rank-margin turned out to be poor "is anything
-- relevant here" signals — the margin is inverted, because RRF margins are
-- rank arithmetic rather than relevance. The raw retriever scores separate
-- covered from uncovered queries cleanly.
--
-- See results/2026-10-08-retrieval-signal-selection/ and docs/05 §6.
--
-- All four are stored per turn so week-3 calibration (W3-7) picks the
-- operating point from the full eval distribution rather than from the 16
-- queries that prompted this change.

ALTER TABLE turns ADD COLUMN IF NOT EXISTS retrieval_top_vec_score REAL;
ALTER TABLE turns ADD COLUMN IF NOT EXISTS retrieval_top_lex_score REAL;
ALTER TABLE turns ADD COLUMN IF NOT EXISTS retrieval_vec_margin   REAL;

COMMENT ON COLUMN turns.retrieval_top_score IS
  'Fused RRF rank-1 score. Ranking provenance; a weak sufficiency signal.';
COMMENT ON COLUMN turns.retrieval_margin IS
  'fused(1) - fused(2). Kept for calibration. NOT a sufficiency signal: measured inverted.';
COMMENT ON COLUMN turns.retrieval_top_vec_score IS
  'Raw dense cosine of the best chunk. Primary sufficiency signal.';
