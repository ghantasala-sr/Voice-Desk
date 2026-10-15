"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

Hybrid rather than pure vector because the corpus is a handful of policy
documents where exact terms carry the meaning — "will-call", "postponement",
"face value", "ADA". Pure dense retrieval over a tiny corpus is the weakest
option available, and the first soft handoff signal depends on retrieval score
quality (docs/00 D6).

FOUR signals come out of here, and which ones are worth thresholding on was
measured rather than assumed — see results/2026-10-08-retrieval-signal-selection/.

    top_score      fused rank-1 score        ranking provenance; WEAK separator
    margin         fused(1) - fused(2)       INVERTED. Do not threshold on this.
    top_vec_score  raw dense cosine, rank 1  PRIMARY sufficiency signal
    top_lex_score  raw BM25, rank 1          secondary sufficiency signal

The original design assumed the fused margin would be the strongest
"is anything relevant here" signal. It is not, and the reason is structural:
RRF margins are rank arithmetic, not relevance. A chunk ranked 1 in both
retrievers scores 2/61; one ranked 2 in both scores 2/62. The gap between them
is ~0.0005 whether the match is perfect or hopeless.

Measured medians over 12 covered and 4 uncovered queries:

    fused       covered 0.0327   uncovered 0.0320   barely separates
    margin      covered 0.0006   uncovered 0.0010   INVERTED
    top_vec     covered 0.795    uncovered 0.555    separates cleanly
    top_lex     covered 0.800    uncovered 0.100    separates cleanly

All four are persisted per turn so week-3 calibration (W3-7) picks the
operating point from the full eval distribution rather than from these 16
queries. Thresholds stay PROVISIONAL until then (docs/05 §6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core import db
from core.settings import Settings
from retrieval.embed import Embedder

# Candidate pool per retriever before fusion. Wider than top_k so a chunk that
# ranks 8th lexically and 2nd densely can still surface.
CANDIDATES = 20

# A short or anaphoric utterance is a follow-up, and querying on it alone is
# the single biggest source of retrieval misses in voice.
_FOLLOW_UP_MAX_WORDS = 5
_ANAPHORIC = re.compile(
    r"^\s*(and|but|what about|how about|so|then|ok|okay|that|those|it|they|them|"
    r"the other|which one|why)\b",
    re.IGNORECASE,
)

_SEARCH_SQL = """
WITH q AS (
  SELECT %(qvec)s::vector AS qvec,
         -- websearch_to_tsquery ANDs every term, so a natural-length question
         -- ("can I get my money back for a cancelled show") matches nothing
         -- and the lexical half of this hybrid search goes silently dead.
         -- Rewriting & to | keeps the stemming, stopword removal and phrase
         -- handling, and lets ts_rank_cd do the ranking by how many terms a
         -- chunk matches and how close together they are.
         replace(websearch_to_tsquery('english', %(qtext)s)::text, '&', '|')::tsquery AS qts
),
lex AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q.qts) DESC, c.chunk_id) AS rank,
         ts_rank_cd(c.tsv, q.qts) AS raw
  FROM chunks c, q
  WHERE c.domain_config = %(domain)s AND c.tsv @@ q.qts
  ORDER BY ts_rank_cd(c.tsv, q.qts) DESC, c.chunk_id
  LIMIT %(candidates)s
),
vec AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (ORDER BY c.embedding <=> q.qvec, c.chunk_id) AS rank,
         1 - (c.embedding <=> q.qvec) AS raw
  FROM chunks c, q
  WHERE c.domain_config = %(domain)s
  ORDER BY c.embedding <=> q.qvec, c.chunk_id
  LIMIT %(candidates)s
)
SELECT c.chunk_id, c.doc_id, d.path AS doc_path, c.heading_path, c.text,
       COALESCE(1.0 / (%(rrf_k)s + lex.rank), 0)
     + COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0) AS fused_score,
       lex.raw AS lex_score, vec.raw AS vec_score,
       lex.rank AS lex_rank, vec.rank AS vec_rank
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
LEFT JOIN lex ON lex.chunk_id = c.chunk_id
LEFT JOIN vec ON vec.chunk_id = c.chunk_id
WHERE lex.chunk_id IS NOT NULL OR vec.chunk_id IS NOT NULL
-- Ties are common (equal ts_rank_cd values) and Postgres breaks them by
-- physical row order, which a re-ingest changes. Break them explicitly.
ORDER BY fused_score DESC, vec.rank NULLS LAST, c.chunk_id
LIMIT %(top_k)s
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_path: str
    heading_path: str
    text: str
    fused_score: float
    lex_score: float | None
    vec_score: float | None
    lex_rank: int | None
    vec_rank: int | None

    @property
    def tag_source(self) -> str:
        return self.heading_path


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    #: Fused rank-1 score. Good for ordering, weak for sufficiency.
    top_score: float
    #: fused(1) - fused(2). Kept for calibration, NOT a sufficiency signal.
    margin: float
    #: Raw dense cosine of the best chunk. The primary sufficiency signal.
    top_vec_score: float
    #: Raw BM25 rank of the best chunk. Secondary sufficiency signal.
    top_lex_score: float
    #: dense(1) - dense(2).
    vec_margin: float
    query_text: str
    elapsed_ms: int

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def knowledge_block(self) -> str:
        """Format for the prompt so the model can cite and the checker can ground.

        The checker does NOT rely on the model emitting these tags honestly —
        it grounds against the full chunk text regardless of what the model
        claims. Self-reported citations are an evaluation convenience, never
        an enforcement mechanism (docs/05 §8).
        """
        if not self.chunks:
            return "<knowledge>\n(nothing relevant was found)\n</knowledge>"
        parts = [
            f"[K{index + 1}] {chunk.heading_path}\n{chunk.text}"
            for index, chunk in enumerate(self.chunks)
        ]
        return "<knowledge>\n" + "\n\n".join(parts) + "\n</knowledge>"

    def chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.chunks]


def build_query(utterance: str, previous: str | None) -> str:
    """Query text is not the raw transcript.

    A follow-up like "what about parking?" or "and the other one?" carries its
    subject in the PREVIOUS turn. Querying the bare utterance retrieves
    nothing useful, and conversational follow-ups are the common case in voice.
    """
    utterance = utterance.strip()
    if not previous:
        return utterance
    words = len(utterance.split())
    if words <= _FOLLOW_UP_MAX_WORDS or _ANAPHORIC.match(utterance):
        return f"{previous.strip()} {utterance}"
    return utterance


async def search(
    utterance: str,
    *,
    domain: str,
    embedder: Embedder,
    settings: Settings,
    previous_utterance: str | None = None,
) -> RetrievalResult:
    import time

    started = time.monotonic()
    query_text = build_query(utterance, previous_utterance)
    vector = embedder.embed_query(query_text)

    rows = await db.fetch_all(
        _SEARCH_SQL,
        {
            "qvec": "[" + ",".join(f"{x:.6f}" for x in vector) + "]",
            "qtext": query_text,
            "domain": domain,
            "candidates": CANDIDATES,
            "rrf_k": settings.rrf_k,
            "top_k": settings.retrieval_top_k,
        },
    )

    chunks = [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            doc_path=row["doc_path"],
            heading_path=row["heading_path"],
            text=row["text"],
            fused_score=float(row["fused_score"]),
            lex_score=float(row["lex_score"]) if row["lex_score"] is not None else None,
            vec_score=float(row["vec_score"]) if row["vec_score"] is not None else None,
            lex_rank=row["lex_rank"],
            vec_rank=row["vec_rank"],
        )
        for row in rows
    ]

    top = chunks[0].fused_score if chunks else 0.0
    second = chunks[1].fused_score if len(chunks) > 1 else 0.0

    # Sorted, not in fused order: top_vec_score is the best dense cosine
    # among the results. Until week 4 this read dense[0] in FUSED order, so a
    # lexical tie could swap which chunk's cosine became the "sufficiency
    # signal" — the same query scored 0.670 in one run and 0.621 in the next.
    dense = sorted((c.vec_score for c in chunks if c.vec_score is not None), reverse=True)
    lexical = [c.lex_score for c in chunks if c.lex_score is not None]

    return RetrievalResult(
        chunks=chunks,
        top_score=top,
        margin=top - second,
        top_vec_score=dense[0] if dense else 0.0,
        top_lex_score=max(lexical) if lexical else 0.0,
        vec_margin=(dense[0] - dense[1]) if len(dense) > 1 else 0.0,
        query_text=query_text,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
