# 05 — Retrieval

Retrieval has two jobs here, and the second one is the one people forget:

1. Find the chunk that answers the question.
2. Produce a **score you can threshold on**, because "the top score is too low" is the
   first soft handoff signal and the main defence against confident nonsense.

Job 2 is why this is hybrid search with a calibrated threshold rather than
`ORDER BY embedding <=> query LIMIT 4`.

---

## 1. Corpus authoring

Twelve markdown documents in `domains/ticketing/corpus/`, of which nine are ingested at
baseline and three are held out ([00-decisions.md](00-decisions.md) D4).

```
corpus/
  01-refunds-and-exchanges.md
  02-ticket-transfers.md
  03-event-cancellation-postponement.md
  04-delivery-mobile-and-willcall.md
  05-venue-bag-policy.md
  06-venue-parking.md
  07-accessibility.md
  08-payment-and-billing.md
  09-resale-and-face-value.md
  10-group-and-vip-packages.md        # HELD OUT
  11-weather-and-outdoor-events.md    # HELD OUT
  12-age-restrictions-and-id.md       # HELD OUT
```

`corpus_manifest.yaml` is the single source of truth about what is indexed:

```yaml
domain: ticketing
documents:
  - path: 01-refunds-and-exchanges.md
    title: Refunds and exchanges
    topics: [refund, exchange, cancellation_by_customer]
    ingested: true
  # ...
  - path: 10-group-and-vip-packages.md
    title: Group bookings and VIP packages
    topics: [group_booking, vip, hospitality]
    ingested: false
    held_out_reason: baseline coverage gap for the week-4 feedback cycle
```

### Writing rules

These are not style preferences; each one changes retrieval or speech quality.

- **Every document is a flat list of `##` sections, each answering one question.** The
  section is the retrieval unit. If a section answers two questions it will be retrieved for
  one and confuse the model on the other.
- **Section bodies are 60–120 words.** This is a *spoken answer* budget: the model must
  compress a chunk into one or two sentences, and a 400-word chunk invites it to summarize
  across facts, which is where ungrounded claims come from.
- **State numbers explicitly and once.** "Refunds are processed within **5–7 business
  days**" — not "about a week." The checker grounds on surface forms, and a vague corpus
  makes a vague checker.
- **Write the exact terms callers say**, including the awkward ones: "will-call", "box
  office", "e-ticket", "the app won't load my tickets". Lexical retrieval needs them.
- **No tables, no nested lists.** They chunk badly and read badly aloud.
- **Front-matter per document** carries `topics`, which the gap report joins against.

Three documents are deliberately *thin* on edge cases (e.g. refunds says nothing about
partial-event cancellation). These produce genuine low-score retrievals and exercise the
handoff path — a corpus that answers everything cannot demonstrate a handoff.

---

## 2. Chunking

`retrieval/chunk.py` — heading-aware, not sliding-window.

```python
@dataclass(frozen=True)
class Chunk:
    seq: int
    heading_path: str   # "Refunds and exchanges > Event cancelled by the organizer"
    text: str
    embed_text: str     # heading_path + "\n" + text
    word_count: int
```

Algorithm:

1. Parse markdown to a heading tree.
2. Emit one chunk per leaf `##`/`###` section.
3. If a section exceeds 140 words, split on paragraph boundaries into near-equal parts and
   **repeat the heading path on each part**.
4. If a section is under 25 words, merge it into the following sibling.
5. No overlap. Overlap exists to rescue sliding windows from cutting mid-thought; heading
   sections already end at thoughts, and overlap would duplicate facts across chunks, which
   inflates lexical scores and makes the threshold harder to calibrate.

**`embed_text` prepends the heading path.** A chunk body that reads "This is not available
for third-party resale purchases" is meaningless in isolation; with
`Refunds > Eligibility` in front of it, both the embedding and BM25 land correctly.

---

## 3. Embeddings

`BAAI/bge-small-en-v1.5`, 384 dimensions, CPU-friendly, strong for its size.

Two details that are easy to get wrong and expensive to debug:

- **Asymmetric prefixing.** BGE expects queries to be prefixed with
  `"Represent this sentence for searching relevant passages: "` and passages to be
  unprefixed. Getting this backwards, or applying it to both, measurably degrades retrieval
  while looking like it works. `embed.py` exposes `embed_query()` and `embed_passage()` as
  separate functions so the asymmetry cannot be accidentally collapsed.
- **Normalize to unit length** at ingest, then cosine distance (`<=>`) is a dot product and
  scores are comparable across queries.

```python
# retrieval/embed.py
def embed_query(text: str) -> list[float]: ...     # applies the BGE query prefix
def embed_passage(text: str) -> list[float]: ...   # no prefix
EMBED_DIM = 384
```

`ingest.py` asserts `EMBED_DIM` equals the `vector(N)` column width at startup and refuses
to run on mismatch, rather than writing silently-wrong vectors.

---

## 4. Ingestion

`make ingest DOMAIN=ticketing`:

1. Read `corpus_manifest.yaml`.
2. Upsert **all** documents into `documents`, including held-out ones with
   `ingested = FALSE`. They are recorded but unindexed — this lets the gap report
   distinguish *"no document covers this"* from *"a document exists but is not indexed."*
3. For each ingested document: hash content, skip if the hash is unchanged, otherwise
   delete its chunks and re-chunk.
4. Embed in batches of 32, write `chunks` rows. `tsv` is a generated column — Postgres
   fills it.
5. Compute and print the **corpus hash**: `sha256` over the sorted list of
   `(path, content_hash)` for ingested documents only. This value goes into every
   `config_snapshot` and is how a before/after comparison proves the corpus actually
   changed ([14-analytics.md](14-analytics.md) §5).

Ingestion is idempotent and safe to re-run. `--force` re-embeds everything.

---

## 5. Hybrid search

One SQL statement. Lexical and dense candidates are generated separately, then fused with
Reciprocal Rank Fusion.

```sql
WITH q AS (
  SELECT %(qvec)s::vector AS qvec,
         websearch_to_tsquery('english', %(qtext)s) AS qts
),
lex AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q.qts) DESC) AS rank,
         ts_rank_cd(c.tsv, q.qts) AS raw
  FROM chunks c, q
  WHERE c.domain_config = %(domain)s AND c.tsv @@ q.qts
  LIMIT 20
),
vec AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (ORDER BY c.embedding <=> q.qvec) AS rank,
         1 - (c.embedding <=> q.qvec) AS raw
  FROM chunks c, q
  WHERE c.domain_config = %(domain)s
  ORDER BY c.embedding <=> q.qvec
  LIMIT 20
)
SELECT c.chunk_id, c.doc_id, c.heading_path, c.text,
       COALESCE(1.0 / (%(rrf_k)s + lex.rank), 0)
     + COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0) AS fused_score,
       lex.raw AS lex_score, vec.raw AS vec_score,
       lex.rank AS lex_rank, vec.rank AS vec_rank
FROM chunks c
LEFT JOIN lex ON lex.chunk_id = c.chunk_id
LEFT JOIN vec ON vec.chunk_id = c.chunk_id
WHERE lex.chunk_id IS NOT NULL OR vec.chunk_id IS NOT NULL
ORDER BY fused_score DESC
LIMIT %(top_k)s;
```

- `websearch_to_tsquery` tolerates caller phrasing (quotes, `or`, `-`) without throwing on
  punctuation the way `to_tsquery` does.
- `rrf_k = 60` is the standard constant; it damps the influence of rank-1 so a single
  confident-but-wrong retriever cannot dominate.
- Both `lex_score` and `vec_score` are returned and logged even though fusion uses ranks —
  the calibration step in §6 needs the raw distributions.

### Query construction

The query text is **not** the raw transcript. `search.py` builds it from:

- the current user utterance, plus
- the previous user utterance if the current one is short (< 5 words) or starts with a
  pronoun/ellipsis ("what about parking?", "and the other one?").

Conversational follow-ups are the common case in voice and the single biggest source of
retrieval misses if you query on the bare utterance.

---

## 6. Threshold calibration

This is the part that turns an arbitrary constant into a result
([00-decisions.md](00-decisions.md) D7).

**The problem.** RRF fused scores are bounded and compressed — with `k=60` and `top_k=4`,
a perfect hit scores `2/61 ≈ 0.0328` and a mediocre one `1/63 ≈ 0.0159`. Bi-encoder cosine
is worse: relevant and irrelevant chunks routinely sit at 0.61 and 0.52. Picking a number
by intuition will be wrong, and the handoff recall metric will inherit that error.

**Two signals, not one.**

| Signal | Definition | Why |
|---|---|---|
| `retrieval_top_score` | fused score of rank 1 | absolute confidence |
| `retrieval_margin` | `fused(1) − fused(2)` | *discrimination* — a flat distribution means nothing stood out, which is the real signature of an uncovered topic |

Margin is often the stronger signal. A query about a held-out topic returns four
mediocre-and-similar chunks; a query that is well covered returns one that clearly wins.

**Procedure** (`eval/calibrate_retrieval.py`, run after the eval set exists):

1. For every eval case, run retrieval and record `top_score`, `margin`, and whether the
   labelled gold document appears in the top-k (`hit`).
2. Add the held-out-topic cases as known negatives: no correct chunk exists for them.
3. Plot overlaid histograms of `top_score` and `margin` for hit vs miss. Save to
   `results/<run>/retrieval_calibration.png`.
4. Sweep both thresholds over a grid, compute precision/recall of the
   "retrieval is insufficient" decision against the `hit` label, and pick the operating
   point that achieves **recall ≥ 0.90 on misses** — a missed handoff (confident wrong
   answer) is much worse than an unnecessary one.
5. Write the chosen values into `handoff.yaml` with a comment naming the run directory they
   came from.

Until this runs, `handoff.yaml` carries provisional values tagged `PROVISIONAL:`, and the
report must not quote handoff metrics as final.

---

## 7. Interface

```python
# retrieval/search.py
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    heading_path: str
    text: str
    fused_score: float
    lex_score: float | None
    vec_score: float | None
    lex_rank: int | None
    vec_rank: int | None

@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    top_score: float
    margin: float
    query_text: str          # the constructed query, for debugging
    elapsed_ms: int

async def search(
    utterance: str,
    *,
    domain: str,
    prev_utterance: str | None = None,
    top_k: int = 4,
) -> RetrievalResult: ...
```

Target: **under 30 ms p95** on a few-hundred-row corpus. If it is not, the connection pool
is the problem, not the query.

---

## 8. How chunks enter the prompt

Formatted so the model can cite and the checker can ground:

```
<knowledge>
[K1] Refunds and exchanges > Event cancelled by the organizer
If an event is cancelled by the organizer, the full ticket price including fees is
refunded automatically to the original payment method within 5-7 business days...

[K2] Refunds and exchanges > Eligibility
...
</knowledge>
```

The system prompt instructs the model to answer only from `<knowledge>`, to reference the
tag it used, and to say it does not know rather than fill a gap
([06-harness.md](06-harness.md) §3). The tag is captured into
`TurnState.retrieved_chunk_ids` and is what the LLM judge scores citation correctness
against.

The checker does **not** rely on the model emitting the tag honestly — it grounds against
the full retrieved chunk text regardless of what the model claims
([08-checker.md](08-checker.md)). Self-reported citations are a convenience for evaluation,
never an enforcement mechanism.
