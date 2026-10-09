-- 002_corpus.sql — the knowledge base (used from week 2).
--
-- Created in week 1 so `make clean && make up` always produces the complete
-- schema and week 2 is a code change only.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- documents
--
-- ALL authored documents live here, including held-out ones with
-- ingested = FALSE and no chunks (docs/00-decisions.md D4). That distinction
-- lets the gap report say "a document exists for this topic but is not
-- indexed", which is a different finding from "no document covers this topic".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
  doc_id        TEXT PRIMARY KEY,
  domain_config TEXT NOT NULL,
  path          TEXT NOT NULL,
  title         TEXT NOT NULL,
  topics        TEXT[] NOT NULL DEFAULT '{}',
  content_hash  TEXT NOT NULL,
  ingested      BOOLEAN NOT NULL DEFAULT TRUE,
  ingested_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (domain_config, path)
);

CREATE INDEX IF NOT EXISTS documents_topics_idx ON documents USING GIN (topics);

-- ---------------------------------------------------------------------------
-- chunks
--
-- embedding dimension is tied to VD_EMBED_MODEL. Changing the model is a
-- migration, not a config flip — retrieval/ingest.py asserts the dimension
-- matches at startup and refuses to write silently-wrong vectors.
--
-- BAAI/bge-small-en-v1.5 -> 384 dimensions.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      TEXT PRIMARY KEY,
  doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  domain_config TEXT NOT NULL,
  seq           INTEGER NOT NULL,

  -- "Refunds and exchanges > Event cancelled by the organizer".
  -- Prepended to embed_text because a chunk body is often meaningless alone.
  heading_path  TEXT NOT NULL,
  text          TEXT NOT NULL,
  embed_text    TEXT NOT NULL,
  word_count    INTEGER NOT NULL,

  embedding     vector(384) NOT NULL,

  -- Lexical half of hybrid retrieval. Generated, so Postgres maintains it.
  tsv           tsvector GENERATED ALWAYS AS
                  (to_tsvector('english', coalesce(heading_path,'') || ' ' || text)) STORED,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_domain_idx ON chunks (domain_config);

-- HNSW over a few hundred rows does nothing for recall, but it costs nothing
-- and keeps the query identical if the corpus grows. Keep ef_search high
-- during threshold calibration so recall is never silently capped.
CREATE INDEX IF NOT EXISTS chunks_vec_idx ON chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
