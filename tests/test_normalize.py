"""W2-8 — spoken-form normalization.

Every row here is a regression waiting to happen, and most are judged by ear:
a TTS engine reading "TK-88213" as "tee kay eighty-eight thousand two hundred
thirteen" is the most noticeable thing that makes a voice agent feel
unfinished.
"""

from __future__ import annotations

import pytest

from harness.normalize import (
    canonicalize,
    number_to_words,
    ordinal_words,
    strip_citations,
    to_spoken,
    year_words,
)

SPOKEN = [
    # currency
    ("Parking is $25.", "Parking is twenty-five dollars."),
    ("That's $45.00.", "That's forty-five dollars."),
    ("It costs $1.", "It costs one dollar."),
    ("The fee is $3.50.", "The fee is three dollars and fifty cents."),
    ("That's $1,250.50 total.", "That's one thousand two hundred fifty dollars and fifty cents total."),
    ("Only $0.01 left.", "Only zero dollars and one cent left."),
    # ranges, percent, durations
    ("It takes 5-7 business days.", "It takes five to seven business days."),
    ("Expect 3\u20135 days.", "Expect three to five days."),  # en dash
    ("Fees are 8 to 18% of face value.", "Fees are eight to eighteen percent of face value."),
    ("A 10% fee applies.", "A ten percent fee applies."),
    ("Released 48 hours before doors.", "Released forty-eight hours before doors."),
    ("About 2,000 cars fit.", "About two thousand cars fit."),
    ("It opened in 2000.", "It opened in two thousand."),
    # order ids and emails — the two that matter most in a demo
    ("Order TK-88213 is confirmed.", "Order T K, eight eight two one three is confirmed."),
    ("Sent to sam@example.com.", "Sent to sam at example dot com."),
    ("Sent to sam.lee@mail.example.org.", "Sent to sam dot lee at mail dot example dot org."),
    ("Try first_last@example.com.", "Try first underscore last at example dot com."),
    ("Write to box-office@example.com.", "Write to box dash office at example dot com."),
    # dates and times
    ("The event is 2026-03-14.", "The event is March fourteenth, twenty twenty-six."),
    ("See you on March 14.", "See you on March fourteenth."),
    ("It moved to May 1st.", "It moved to May first."),
    ("Doors at 7:30 PM.", "Doors at seven thirty P M."),
    ("Doors at 7:05 pm.", "Doors at seven oh five P M."),
    ("Doors at 6:00.", "Doors at six."),
    ("Back in 2005.", "Back in two thousand five."),
    # misc
    ("You're in seat #4.", "You're in seat number four."),
    ("Section 112, row 3.", "Section one hundred twelve, row three."),
    ("The bag limit is 12.5 inches.", "The bag limit is twelve point five inches."),
    ("Bring photo ID.", "Bring photo I D."),
    ("ADA seating is available.", "A D A seating is available."),
    ("VIP entry is early.", "V I P entry is early."),
    ("OK, that works.", "OK, that works."),
    # citations and URLs never reach the caller
    ("Refunds take 5 days [K1].", "Refunds take five days."),
    ("See the policy [K2] [K3].", "See the policy."),
    ("Details at https://example.com/refunds today.", "Details at today."),
    ("Nothing to change here.", "Nothing to change here."),
]


@pytest.mark.parametrize("written,spoken", SPOKEN)
def test_to_spoken(written: str, spoken: str) -> None:
    assert to_spoken(written) == spoken


@pytest.mark.parametrize(
    "n,words",
    [(0, "zero"), (13, "thirteen"), (21, "twenty-one"), (100, "one hundred"),
     (105, "one hundred five"), (999, "nine hundred ninety-nine"),
     (1000, "one thousand"), (2600, "two thousand six hundred"),
     (1_000_000, "one million")],
)
def test_number_to_words(n: int, words: str) -> None:
    assert number_to_words(n) == words


@pytest.mark.parametrize(
    "n,words",
    [(1, "first"), (2, "second"), (3, "third"), (12, "twelfth"), (14, "fourteenth"),
     (20, "twentieth"), (21, "twenty-first"), (30, "thirtieth"), (31, "thirty-first")],
)
def test_ordinals(n: int, words: str) -> None:
    assert ordinal_words(n) == words


def test_years() -> None:
    assert year_words(2026) == "twenty twenty-six"
    assert year_words(2005) == "two thousand five"
    assert year_words(1999) == "nineteen ninety-nine"
    assert year_words(1905) == "nineteen oh five"


def test_citations_are_stripped_and_reported() -> None:
    text, tags = strip_citations("Refunds take 5 days [K1], fees too [K3].")
    assert text == "Refunds take 5 days, fees too."
    assert tags == [1, 3]


# ---------------------------------------------------------------------------
# canonicalize — for matching, not speech
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b",
    [
        ("five to seven business days", "5 to 7 business days"),
        ("5-7 business days", "5 to 7 business days"),
        ("twenty-five dollars", "$25"),
        ("$3.50", "3 dollars and 50 cents"),
        ("ten percent", "10%"),
        ("two thousand cars", "2,000 cars"),
        ("one hundred and fifty", "150"),
    ],
)
def test_equivalent_forms_canonicalize_the_same(a: str, b: str) -> None:
    assert canonicalize(a) == canonicalize(b)


def test_word_numbers_can_be_marked() -> None:
    """The checker needs to know a number was written as a word, so that
    'which one did you mean' is never treated as a quantity."""
    assert canonicalize("which one", mark_word_numbers=True) == "which ~1"
    assert canonicalize("which 1", mark_word_numbers=True) == "which 1"


@pytest.mark.parametrize("text, tags", [
    ("Transfers are free [K1].", [1]),
    ("They won't be downloadable yet [K2, K3].", [2, 3]),   # Gemini's form
    ("Free [K2][K4] and fast [K3,K5].", [2, 4, 3, 5]),
    ("Section [K 2; 3] is fine.", [2, 3]),
])
def test_grouped_citation_tags_are_never_spoken(text: str, tags: list[int]) -> None:
    clean, cited = strip_citations(text)
    assert "[" not in clean and "K" not in clean.replace("King", "")
    assert cited == tags
    assert "[" not in to_spoken(text)
