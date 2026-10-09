-- 005_grounding.sql — W2-5 / W2-6.
--
-- first_content_ms is the HONEST time-to-first-audio: the first model-written
-- segment, not an acknowledgment line. first_audio_ms counts an "ack" as
-- audio, which front-loads filler rather than information (docs/01 §5).
--
-- cited_chunk_ids is what the model CLAIMED to cite with [K] tags.
-- Self-reported, so it is evaluation input for the week-3 judge only; the
-- checker never trusts it.

ALTER TABLE turns ADD COLUMN IF NOT EXISTS first_content_ms INTEGER;
ALTER TABLE turns ADD COLUMN IF NOT EXISTS cited_chunk_ids TEXT[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN turns.first_audio_ms IS
  'Harness-side time to ANY audio, including an acknowledgment line.';
COMMENT ON COLUMN turns.first_content_ms IS
  'Harness-side time to the first model-written segment. The honest headline.';
