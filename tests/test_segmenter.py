"""W2-5 groundwork — the segmenter decides how long we wait before speaking.

Every boundary rule here is a latency/quality trade-off, so each gets a test.
"""

from __future__ import annotations

import pytest

from harness.segmenter import Segmenter


def segment_all(text: str, **kwargs: int) -> list[str]:
    """Feed `text` one character at a time — the worst case for a streaming
    segmenter, and the one that catches off-by-one boundary bugs."""
    segmenter = Segmenter(**kwargs)
    out: list[str] = []
    for char in text:
        out.extend(segmenter.feed(char))
    tail = segmenter.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_sentences() -> None:
    assert segment_all("Refunds take five days. Transfers are free. ") == [
        "Refunds take five days.",
        "Transfers are free.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Alvarez has two tickets waiting at will-call for him.",
        "Your seat is in section No. 4 on the upper level near the stairs.",
        "The fee is approx. nine dollars per ticket on this particular order.",
    ],
)
def test_abbreviations_do_not_end_a_sentence(text: str) -> None:
    assert segment_all(text) == [text]


def test_decimals_do_not_end_a_sentence() -> None:
    text = "The service fee on that order is 5.50 per ticket."
    assert segment_all(text) == [text]


def test_splits_on_semicolon_and_colon() -> None:
    assert segment_all("Here is the rule; it is simple. ") == [
        "Here is the rule;",
        "it is simple.",
    ]


def test_short_fragment_merges_forward() -> None:
    """A two-word fragment must not reach TTS on its own — it dangles."""
    assert segment_all("Yes. Your tickets were delivered to the app this morning. ") == [
        "Yes. Your tickets were delivered to the app this morning.",
    ]


def test_clause_split_only_past_the_minimum() -> None:
    short = "I checked, and it is fine."
    assert segment_all(short) == [short]

    long = (
        "I checked the order and the tickets were delivered to the mobile app "
        "this morning, and you can find them under upcoming events."
    )
    parts = segment_all(long)
    assert len(parts) == 2
    assert parts[0].endswith(",")


def test_hard_cap_forces_a_boundary() -> None:
    runaway = " ".join(["word"] * 60)
    parts = segment_all(runaway, max_words=25)
    assert len(parts) >= 2
    assert all(len(part.split()) <= 25 for part in parts[:-1])


def test_flush_returns_the_tail() -> None:
    segmenter = Segmenter()
    assert segmenter.feed("No trailing punctuation here") == []
    assert segmenter.flush() == "No trailing punctuation here"
    assert segmenter.flush() is None
