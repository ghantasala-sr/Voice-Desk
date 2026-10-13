"""W2-6 / W2-7 — the grounding checker.

The bar that matters most is the FALSE-POSITIVE rate on known-good answers. A
checker that rejects correct sentences is worse than no checker: the fallback
line fires constantly and the product looks broken. So the first block here
runs realistic answers against the REAL corpus chunks and requires zero
failures, before any test of what the checker catches.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.checker import Checker, GroundingSources, load_lexicon
from retrieval.chunk import chunk_document

ROOT = Path(__file__).resolve().parent.parent / "domains" / "ticketing"
LEXICON = load_lexicon("ticketing")


def chunks_of(doc: str) -> list[str]:
    """Every chunk of one shipped document, as the retriever would return them."""
    manifest = yaml.safe_load((ROOT / "corpus_manifest.yaml").read_text())
    entry = next(d for d in manifest["documents"] if d["path"].startswith(doc))
    text = (ROOT / "corpus" / entry["path"]).read_text()
    return [c.embed_text for c in chunk_document(text, title=entry["title"])]


def checker_for(doc: str, *, sufficient: bool = True, caller: list[str] | None = None) -> Checker:
    return Checker(
        GroundingSources(chunks=chunks_of(doc), caller=caller or []),
        lexicon=LEXICON,
        retrieval_sufficient=sufficient,
    )


# ---------------------------------------------------------------------------
# Zero false positives on known-good answers (W2-6 acceptance)
# ---------------------------------------------------------------------------
KNOWN_GOOD = [
    (entry["doc"], entry["answer"])
    for entry in yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "eval" / "dataset" / "checker_known_good.yaml").read_text()
    )["answers"]
]


@pytest.mark.parametrize("doc,answer", KNOWN_GOOD)
def test_known_good_answers_are_never_rejected(doc: str, answer: str) -> None:
    verdict = checker_for(doc).check(answer)
    assert verdict.passed, f"false positive: {verdict.verdict} — {verdict.reason}"


def test_false_positive_rate_is_zero() -> None:
    failures = [a for d, a in KNOWN_GOOD if not checker_for(d).check(a).passed]
    print(f"\n  false positives: {len(failures)}/{len(KNOWN_GOOD)}")
    assert not failures


# ---------------------------------------------------------------------------
# Rule 1 — what it catches, by class
# ---------------------------------------------------------------------------
CAUGHT = [
    ("numeric", "04", "Mobile tickets are released 24 hours before doors open."),
    ("numeric", "01", "Refunds are processed within 20 business days."),
    ("currency", "06", "The Harbour Pavilion garage is $35 for event parking."),
    ("currency", "08", "The processing fee is $4.50 per order."),
    ("percent", "09", "Resale prices are capped at face value plus 25 percent."),
    ("entity", "06", "Parking at Seaview Arena is 25 dollars."),
    ("entity", "05", "Backpacks are allowed at Riverside Hall if they fit the limit."),
    ("order_id", "02", "I've started the transfer for order TK-55555."),
]


@pytest.mark.parametrize("kind,doc,answer", CAUGHT)
def test_rule1_catches_ungrounded_entities(kind: str, doc: str, answer: str) -> None:
    verdict = checker_for(doc).check(answer)
    assert verdict.verdict == "fail_ungrounded_entity", f"{kind} slipped through: {answer}"


def test_the_callers_own_words_are_grounding() -> None:
    """Reading the caller's order number back is not a hallucination."""
    checker = checker_for("02", caller=["it's order T K 8 8 2 1 3, sam at example.com"])
    assert checker.check("I'll transfer the tickets on order TK-88213.").passed


def test_tool_results_are_grounding() -> None:
    checker = Checker(
        GroundingSources(chunks=chunks_of("04"), tool_results=['{"quantity": 2, "section": 112}']),
        lexicon=LEXICON,
    )
    assert checker.check("You have 2 tickets in section 112.").passed


def test_word_numbers_without_a_unit_are_not_checked() -> None:
    """'which one did you mean' must never trip the numeric rule."""
    verdict = checker_for("02").check("Which one did you mean?")
    assert verdict.verdict == "no_checkable_content"


# ---------------------------------------------------------------------------
# Rule 2 — unsupported policy claims (W2-7)
# ---------------------------------------------------------------------------
def test_rule2_catches_a_fabricated_claim_when_retrieval_is_weak() -> None:
    verdict = checker_for("05", sufficient=False).check(
        "You can transfer tickets any time before the event."
    )
    assert verdict.verdict == "fail_unsupported_claim"


def test_rule2_does_not_fire_when_retrieval_is_strong() -> None:
    """Condition 4: with a strong retrieval the model is probably paraphrasing."""
    verdict = checker_for("05", sufficient=True).check(
        "You can transfer tickets any time before the event."
    )
    assert verdict.passed


def test_rule2_allows_a_paraphrase_of_its_chunk_even_when_retrieval_is_weak() -> None:
    """Condition 3: a real paraphrase shares most of its chunk's vocabulary."""
    verdict = checker_for("02", sufficient=False).check(
        "You can cancel a pending transfer and the tickets return to your account."
    )
    assert verdict.passed, verdict.reason


def test_rule2_needs_a_policy_modal() -> None:
    verdict = checker_for("05", sufficient=False).check("Let me look into that for you.")
    assert verdict.verdict == "no_checkable_content"


# ---------------------------------------------------------------------------
# The declared gap
# ---------------------------------------------------------------------------
def test_negation_inversion_is_a_known_gap() -> None:
    """NOT a bug report — a recorded limitation.

    An inverted claim shares every entity and most vocabulary with its chunk,
    so no deterministic rule here can see it. Week 3 seeds this as its own
    hallucination class and reports a near-zero catch rate in the table
    rather than leaving it out (docs/08 §3, docs/00 D10).

    If this test starts failing, the checker improved: move the case into
    CAUGHT and update docs/08.
    """
    verdict = checker_for("01").check("Tickets are refundable once the order is complete.")
    assert verdict.passed


def test_number_binding_is_a_known_gap() -> None:
    """NOT a bug report — a recorded limitation.

    Rule 1 is presence-based. The refunds document mentions "10 business days"
    in the sentence about chasing a late refund with your bank, so an answer
    claiming refunds TAKE 10 business days is grounded by a number that
    belongs to a different fact. Catching this needs claim-to-span binding,
    which is an LLM-shaped problem and out of scope for a deterministic v1.
    """
    assert checker_for("01").check("Refunds are processed within 10 business days.").passed


# ---------------------------------------------------------------------------
# Latency (stage [F])
# ---------------------------------------------------------------------------
def test_checker_is_well_inside_its_budget() -> None:
    checker = checker_for("08")
    timings = sorted(checker.check(a).elapsed_ms for _, a in KNOWN_GOOD * 5)
    p95 = timings[int(len(timings) * 0.95)]
    print(f"\n  checker p95: {p95:.3f} ms (budget 5 ms)")
    assert p95 < 5


def test_an_event_date_from_a_tool_result_grounds_the_answer() -> None:
    """Found by the week-3 catch-rate study: ISO dates were being read as
    ranges ("2026 to 11-14"), so a looked-up event date could never ground
    "your show is on November 14" — every date read-back would have been
    replaced with a safe line."""
    checker = Checker(
        GroundingSources(chunks=[], tool_results=['{"event_date": "2026-11-14"}']),
        lexicon=LEXICON,
    )
    assert checker.check("Your show is on November 14.").passed
    assert not checker.check("Your show is on November 21.").passed
