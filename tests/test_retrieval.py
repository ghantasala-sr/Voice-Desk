"""W2-3 — hybrid retrieval.

Marked `db`: needs `make up && make ingest`. The acceptance criteria from
BUILD.md are the tests, not a separate checklist.
"""

from __future__ import annotations

import pytest

from core import db
from core.settings import Settings
from retrieval.embed import get_embedder
from retrieval.search import build_query, search

settings = Settings()
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        not db.is_reachable(settings.database_url),
        reason="Postgres not reachable; run `make up && make ingest`",
    ),
]

# Question -> the document that should answer it. Document level, not chunk
# level: chunk ids break on every re-chunk, and you WILL re-chunk (docs/12 §1).
PAIRS = [
    ("can I get a refund if I can't make it anymore", "01-refunds-and-exchanges.md"),
    ("how long does a refund take to come back", "01-refunds-and-exchanges.md"),
    ("how do I send my tickets to a friend", "02-ticket-transfers.md"),
    ("can I cancel a transfer I already sent", "02-ticket-transfers.md"),
    ("my event got postponed, what are my options", "03-event-cancellation-postponement.md"),
    ("my tickets aren't showing up in the app", "04-delivery-mobile-and-willcall.md"),
    ("how does will-call pickup work", "04-delivery-mobile-and-willcall.md"),
    ("what size bag can I bring in", "05-venue-bag-policy.md"),
    ("is there parking at Harbour Pavilion", "06-venue-parking.md"),
    ("can I bring my service animal", "07-accessibility.md"),
    ("why was my card declined", "08-payment-and-billing.md"),
    ("can I resell tickets I can't use", "09-resale-and-face-value.md"),
]

# Topics covered only by HELD-OUT documents. Retrieval should not find a
# confident answer for these — they are the baseline coverage gaps.
UNCOVERED = [
    "what's included in the VIP package",
    "do you have group rates for fourteen people",
    "what happens if it rains during the show",
    "is the show eighteen plus",
]


@pytest.fixture(scope="module")
async def ctx():
    await db.open_pool(settings)
    yield get_embedder(settings)
    await db.close_pool()


async def _run(ctx, utterance: str, previous: str | None = None):
    return await search(
        utterance,
        domain="ticketing",
        embedder=ctx,
        settings=settings,
        previous_utterance=previous,
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
async def test_top1_document_accuracy(ctx) -> None:
    hits = []
    for question, expected in PAIRS:
        result = await _run(ctx, question)
        top = result.chunks[0].doc_path if result.chunks else None
        hits.append(top == expected)
        if top != expected:
            print(f"  MISS {question!r} -> {top} (wanted {expected})")

    accuracy = sum(hits) / len(hits)
    print(f"\n  top-1 document accuracy: {accuracy:.2f} ({sum(hits)}/{len(hits)})")
    assert accuracy >= 0.8


async def test_held_out_topics_are_not_confidently_answered(ctx) -> None:
    """The baseline gaps must LOOK like gaps, or the handoff signal cannot fire.

    Separation is measured on the RAW DENSE score, not on the fused margin.
    See results/2026-10-08-retrieval-signal-selection/: the fused margin is
    inverted, because RRF margins are rank arithmetic rather than relevance.
    """
    covered = sorted([(await _run(ctx, q)).top_vec_score for q, _ in PAIRS])
    uncovered = sorted([(await _run(ctx, q)).top_vec_score for q in UNCOVERED])

    print(f"\n  top_vec_score: covered min {covered[0]:.3f}, "
          f"uncovered max {uncovered[-1]:.3f}")

    # Clean separation with no overlap is what makes a threshold possible at
    # all. If these ever cross, the signal needs rethinking before W3-7.
    assert uncovered[-1] < covered[0], "covered and uncovered scores overlap"


async def test_fused_margin_is_not_used_as_a_sufficiency_signal(ctx) -> None:
    """Regression guard on a finding, not on behaviour.

    If someone later wires `margin` into handoff.yaml as a sufficiency
    threshold, this documents why that is wrong: the separation runs the wrong
    way round.
    """
    covered = sorted([(await _run(ctx, q)).margin for q, _ in PAIRS])
    uncovered = sorted([(await _run(ctx, q)).margin for q in UNCOVERED])
    covered_median = covered[len(covered) // 2]
    uncovered_median = uncovered[len(uncovered) // 2]

    assert uncovered_median >= covered_median, (
        "the fused margin now separates correctly — re-run the signal "
        "selection study and update docs/05 §6 before relying on it"
    )


async def test_held_out_documents_are_never_retrieved(ctx) -> None:
    for question in UNCOVERED:
        result = await _run(ctx, question)
        paths = {chunk.doc_path for chunk in result.chunks}
        assert not paths & {
            "10-group-and-vip-packages.md",
            "11-weather-and-outdoor-events.md",
            "12-age-restrictions-and-id.md",
        }, f"{question!r} retrieved a held-out document"


# ---------------------------------------------------------------------------
# Both retrievers are doing work
# ---------------------------------------------------------------------------
async def test_exact_policy_terms_are_found_lexically(ctx) -> None:
    """The reason this is hybrid: "will-call" is a term, not a concept."""
    result = await _run(ctx, "will-call")
    assert result.chunks
    assert result.chunks[0].lex_rank is not None, "BM25 found nothing for an exact term"


async def test_fusion_uses_both_retrievers(ctx) -> None:
    result = await _run(ctx, "can I get my money back for a cancelled show")
    assert any(c.lex_rank is not None for c in result.chunks)
    assert any(c.vec_rank is not None for c in result.chunks)


# ---------------------------------------------------------------------------
# Conversational follow-ups
# ---------------------------------------------------------------------------
def test_follow_ups_inherit_the_previous_subject() -> None:
    assert build_query("what about parking?", "is there a bag limit at Kestrel") == (
        "is there a bag limit at Kestrel what about parking?"
    )
    assert build_query("and the other one?", "where are my tickets").startswith(
        "where are my tickets"
    )
    # A full question stands on its own.
    full = "how long does a refund take to reach my bank account"
    assert build_query(full, "something earlier") == full


async def test_follow_up_context_changes_the_result(ctx) -> None:
    """Querying a bare follow-up is the single biggest source of retrieval
    misses in voice."""
    bare = await _run(ctx, "what about at Kestrel?")
    with_context = await _run(ctx, "what about at Kestrel?", previous="is there parking")

    assert with_context.chunks[0].doc_path == "06-venue-parking.md"
    assert with_context.query_text != bare.query_text


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
async def test_retrieval_is_fast_enough_for_the_budget(ctx) -> None:
    timings = []
    for question, _ in PAIRS:
        timings.append((await _run(ctx, question)).elapsed_ms)
    timings.sort()
    p95 = timings[int(len(timings) * 0.95) - 1]
    print(f"\n  retrieval p95: {p95} ms (stage [C] budget is 40 ms)")
    assert p95 < 40


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
async def test_knowledge_block_is_tagged_for_citation(ctx) -> None:
    result = await _run(ctx, "how long does a refund take")
    block = result.knowledge_block()
    assert block.startswith("<knowledge>")
    assert "[K1]" in block
    assert result.chunks[0].heading_path in block


async def test_empty_retrieval_says_so_rather_than_returning_nothing(ctx) -> None:
    result = await search(
        "zzzqqq", domain="nonexistent-domain", embedder=ctx, settings=settings
    )
    assert result.is_empty
    assert "nothing relevant" in result.knowledge_block()


async def test_top_vec_score_is_the_best_dense_cosine_not_the_fused_leader(ctx) -> None:
    """Week-4 bug: top_vec_score read the cosine of the FUSED rank-1 chunk, so
    a lexical tie decided the sufficiency signal. It must be the maximum."""
    for question in ("My friend needs some of my tickets.", "can I get my money back",
                     "where do I park", "Something looks off with my payment."):
        result = await _run(ctx, question)
        dense = [c.vec_score for c in result.chunks if c.vec_score is not None]
        assert result.top_vec_score == max(dense)
        assert result.vec_margin == pytest.approx(sorted(dense)[-1] - sorted(dense)[-2])


async def test_search_is_deterministic(ctx) -> None:
    question = "My friend needs some of my tickets."
    first = [c.chunk_id for c in (await _run(ctx, question)).chunks]
    for _ in range(5):
        assert [c.chunk_id for c in (await _run(ctx, question)).chunks] == first
