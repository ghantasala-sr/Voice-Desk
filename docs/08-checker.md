# 08 — The sentence checker

Streaming means you cannot check a whole answer before speaking it. The checker inspects
each segment before it reaches TTS, releases it, replaces it, or escalates.

It is deterministic, costs under 5 ms, and makes **no LLM call**
([00-decisions.md](00-decisions.md) D10). Its coverage boundary is declared rather than
hidden.

---

## 1. Interface

```python
# harness/checker.py
@dataclass(frozen=True)
class Span:
    type: Literal["number","currency","date","time","duration","percent",
                  "order_id","email","proper_noun","policy_term"]
    surface: str          # as the model wrote it
    normalized: str       # canonical form used for matching
    grounded: bool
    matched_chunk_id: str | None

@dataclass(frozen=True)
class Verdict:
    verdict: Literal["pass","no_checkable_content",
                     "fail_ungrounded_entity","fail_unsupported_claim"]
    spans: list[Span]
    reason: str
    elapsed_ms: int

class Checker:
    def __init__(self, chunks: list[RetrievedChunk], lexicon: Lexicon): ...
    def check(self, segment: str, *, segment_index: int) -> Verdict: ...
```

The checker is constructed per turn from that turn's retrieved chunks. It grounds against
**the chunk text itself**, never against what the model claims to have cited — self-reported
citations are evaluation convenience, not enforcement
([05-retrieval.md](05-retrieval.md) §8).

---

## 2. Rule 1 — entity grounding

Extract checkable spans, then require each to appear in the union of retrieved chunk text.

### Extraction

Regex and lexicon based, in this order:

| Type | Pattern / source |
|---|---|
| `currency` | `$1,234.56`, `12 dollars`, `fifty cents` |
| `percent` | `15%`, `fifteen percent` |
| `duration` | `5-7 business days`, `two weeks`, `48 hours`, `30 days` |
| `date` | `March 14`, `3/14`, `next Saturday`, `2026-03-14` |
| `time` | `7:30 PM`, `seven thirty`, `doors at 6` |
| `number` | bare integers and decimals not captured above |
| `order_id` | `TK-\d{5}` |
| `email` | RFC-ish address pattern |
| `proper_noun` | capitalized runs **matched against the lexicon** (§4) |
| `policy_term` | lexicon terms: `will-call`, `face value`, `resale`, `service fee`, … |

### Normalization before matching

Both the span and the chunk text are normalized so that surface differences do not produce
false alarms. This matters more than the extraction does — an uncalibrated checker that
rejects correct sentences is worse than no checker, because the fallback line fires
constantly and the demo looks broken.

- numerals ↔ words: `5` ≡ `five` ≡ `Five`
- currency: `$45.00` ≡ `45 dollars` ≡ `forty-five dollars`
- ranges: `5-7` ≡ `5 to 7` ≡ `five to seven`
- dates to ISO where resolvable; relative dates resolved against call start
- case folding, punctuation stripping, whitespace collapse
- plural/singular for policy terms

A span that is **present in the user's own utterance** is automatically grounded. If the
caller said "order TK-88213", repeating it back is not a hallucination. Likewise, values
returned by a committed tool call in this turn are grounded — tool results are evidence.

So the grounding sources are, in order: retrieved chunks → tool results this turn → user
utterance this call.

### Verdict

- Every span grounded → `pass`.
- No spans at all → `no_checkable_content` → continue to Rule 2.
- Any span ungrounded → `fail_ungrounded_entity`.

---

## 3. Rule 2 — unsupported policy claim

This is the partial answer to the obvious hole: *"you can transfer tickets any time before
the event"* contains no entities, is entirely wrong, and Rule 1 passes it.

A segment is `fail_unsupported_claim` when **all** of the following hold:

1. It matches a **policy-modal pattern**:
   `you can` / `you can't` / `you may` / `you must` / `you'll need to` / `we will` /
   `we don't` / `is not allowed` / `is free` / `at no charge` / `any time` / `always` /
   `never` / `as long as`.
2. It has **no grounded span** from Rule 1 — nothing anchors it to a chunk.
3. **Lexical overlap with the retrieved chunks is below `VD_CLAIM_OVERLAP_MIN`** (default
   0.25), measured as the fraction of the segment's content words (stopwords removed,
   stemmed) appearing in the retrieved chunk text.
4. `retrieval_top_score` is below the calibrated sufficiency threshold, **or**
   `retrieval_margin` is below its threshold.

All four conditions are required because each alone over-fires. Condition 3 is what makes
it work: a correct policy sentence paraphrases its chunk and shares most of its content
words; a fabricated one does not.

### What this does and does not catch

| Catches | Misses |
|---|---|
| Confident policy claims when retrieval found nothing relevant | A fabricated claim that happens to reuse the chunk's vocabulary |
| Generalizing from an adjacent policy ("refunds work like transfers") | A subtly inverted claim — "refunds are **not** available" where the chunk says they are |
| Inventing a rule in a topic the corpus does not cover | Reasoning errors across two correctly-retrieved chunks |

Negation inversion is the known, unfixable-without-an-LLM gap. It is reported as its own
seeded class, and its catch rate will be near zero. Publishing that number is the point —
see §6.

---

## 4. The lexicon

`Lexicon` is built at ingest and cached per domain:

- all proper nouns appearing in the corpus (venues, event names, cities, partner brands);
- all `policy_term` entries declared in `domains/<d>/prompts.yaml`;
- the seeded mock dataset's entity names.

A capitalized run **not** in the lexicon and **not** in the user utterance is a strong
hallucination signal — an invented venue name, for instance — and is extracted as an
ungrounded `proper_noun` span.

Sentence-initial capitals and a stoplist of common capitalized words (`I`, weekday names,
month names, `Saturday`) are excluded to keep the false-positive rate down.

---

## 5. Actions

| Verdict | Action | Side effects |
|---|---|---|
| `pass` | Release to TTS | — |
| `no_checkable_content` | Release to TTS | — |
| `fail_ungrounded_entity` | **Replace** with a safe line; stop generation for this turn | signal `checker_ungrounded_entity` |
| `fail_unsupported_claim` | **Replace**; stop generation | signal `checker_unsupported_claim` |

Safe lines (fixed, template, no checkable content):

```yaml
safe_lines:
  ungrounded_entity: "Let me double-check that detail before I give you a number."
  unsupported_claim: "I want to be sure I get that right — let me bring in a colleague."
```

A `fail` does not necessarily end the call. The signal goes into the handoff scorer
([09-handoff.md](09-handoff.md)); two checker failures in one call is a hard handoff, but a
single one on an otherwise healthy call may be recoverable by re-querying.

**Already-spoken segments are not retractable.** If segment 3 fails, segments 1 and 2 are
already audio. This is inherent to streaming and is why the segmenter is conservative and
why `generated_text` is kept separately from `approved_text` — the metric must be able to
show what the caller heard before the catch.

---

## 6. Measuring the checker

`eval/seed_hallucinations.py` builds a seeded set by taking known-good answers from the
eval run and corrupting them, one class at a time, into
`checker_verdicts.seeded_class`:

| `seeded_class` | Corruption | Expected to catch |
|---|---|---|
| `numeric` | perturb a number/price/duration | Yes — Rule 1 |
| `date` | shift a date or time | Yes — Rule 1 |
| `entity` | swap a venue or event name for an invented one | Yes — Rule 1 + lexicon |
| `policy_claim` | fabricate a rule in an uncovered topic | Partly — Rule 2 |
| `negation` | invert a correct claim | **No — declared gap** |
| `citation` | attribute a correct fact to the wrong chunk tag | No — not an enforcement target |

Report ([15-results-and-claims.md](15-results-and-claims.md)):

```
Checker catch rate by hallucination class (n per class, run 2026-xx-xx-xxxx)
  numeric       xx/xx  xx%
  date          xx/xx  xx%
  entity        xx/xx  xx%
  policy_claim  xx/xx  xx%
  negation      xx/xx  xx%   ← known gap, no deterministic rule covers this
  citation      xx/xx  xx%   ← not an enforcement target
  ------------------------------------------------
  false-positive rate on known-good segments: x.x%
```

**The false-positive rate is mandatory in the same table.** A checker that rejects
everything scores 100% catch rate and makes the product unusable. Target: catch rate above
90% on `numeric`/`date`/`entity`, false positives under 3%.

---

## 7. Spoken-form normalization

Runs **after** the check passes, on the approved segment, before TTS. Separate from the
checker's internal normalization — that one is for matching, this one is for speech.

```python
# harness/normalize.py
def to_spoken(text: str) -> str: ...
```

| Written | Spoken |
|---|---|
| `$45.00` | `forty-five dollars` |
| `$1,250.50` | `one thousand two hundred fifty dollars and fifty cents` |
| `5-7 business days` | `five to seven business days` |
| `2026-03-14` | `March fourteenth, twenty twenty-six` |
| `7:30 PM` | `seven thirty P M` |
| `TK-88213` | `T K, eight eight two one three` (grouped, with pauses) |
| `sam@example.com` | `sam at example dot com` |
| `#4` | `number four` |
| `ADA` | `A D A` (acronym list) |
| `https://...` | removed; replaced with "I'll send that by email" |

Order IDs and emails are the ones that actually matter in the demo: a TTS engine reading
`TK-88213` as "tee kay eighty-eight thousand two hundred thirteen" is immediately
noticeable and makes the whole system feel unfinished. Digits are spoken individually with
comma pauses.

Unit tested with a table of roughly 40 input/output pairs, because every one of these is a
regression waiting to happen.

---

## 8. Latency accounting

`checker_ms` covers extraction + normalization + matching: target < 5 ms p95, measured on
every segment.

`sentence_buffer_ms` — the wait for a complete segment — is **not** checker time, but it is
a cost the checker's existence imposes, so it is reported as its own stage
([01-architecture.md](01-architecture.md) §5). Conflating them would understate the design's
true latency cost, which is the number an interviewer will care about.
