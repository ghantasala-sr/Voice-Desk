"""Text normalization, in two directions that must not be confused.

  to_spoken(text)      written -> SPEECH. Runs after the checker approves a
                       segment, right before TTS. "$45.00" -> "forty-five
                       dollars", "TK-88213" -> "T K, eight eight two one three".

  canonicalize(text)   any form -> MATCHABLE. Used by the checker so that
                       "five to seven business days" in the answer matches
                       "5 to 7 business days" in the chunk. Never spoken.

A TTS engine reading TK-88213 as "tee kay eighty-eight thousand two hundred
thirteen" is the single most noticeable thing that makes a voice agent feel
unfinished, which is why order ids and emails get the most care here.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Numbers <-> words
# ---------------------------------------------------------------------------
_SMALL = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]


def number_to_words(n: int) -> str:
    """Cardinal: 1250 -> "one thousand two hundred fifty"."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _SMALL[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_SMALL[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_SMALL[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    for size, name in _SCALES:
        if n >= size:
            head, rest = divmod(n, size)
            return f"{number_to_words(head)} {name}" + (
                f" {number_to_words(rest)}" if rest else ""
            )
    raise AssertionError("unreachable")


_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def ordinal_words(n: int) -> str:
    """14 -> "fourteenth", 21 -> "twenty-first"."""
    words = number_to_words(n)
    head, sep, last = words.rpartition("-") if "-" in words.split()[-1] else words.rpartition(" ")
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}" if sep else last


def year_words(year: int) -> str:
    """2026 -> "twenty twenty-six", 2005 -> "two thousand five"."""
    # 2000-2009 first: "two thousand", not "twenty hundred".
    if 2000 <= year < 2010:
        return number_to_words(year)
    century, rest = divmod(year, 100)
    if rest == 0:
        return f"{number_to_words(century)} hundred"
    return f"{number_to_words(century)} {number_to_words(rest) if rest >= 10 else 'oh ' + number_to_words(rest)}"


_WORD_VALUE = {word: value for value, word in enumerate(_SMALL)}
_WORD_VALUE.update({word: value * 10 for value, word in enumerate(_TENS) if word})
_SCALE_WORDS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_WORD = "|".join(sorted(list(_WORD_VALUE) + list(_SCALE_WORDS), key=len, reverse=True))
_NUMBER_RUN = re.compile(
    rf"\b(?:{_NUMBER_WORD})(?:(?:\s+|-)(?:and\s+)?(?:{_NUMBER_WORD}))*\b", re.IGNORECASE
)


def _parse_number_words(run: str) -> int:
    total = current = 0
    for word in re.split(r"[\s-]+", run.lower()):
        if word == "and":
            continue
        if word in _WORD_VALUE:
            current += _WORD_VALUE[word]
        elif word == "hundred":
            current = (current or 1) * 100
        else:  # thousand / million
            total += (current or 1) * _SCALE_WORDS[word]
            current = 0
    return total + current


# ---------------------------------------------------------------------------
# canonicalize — for MATCHING, never for speech
# ---------------------------------------------------------------------------
#: Marks a number that was written as words. The checker treats a bare
#: word-number with no unit ("which one did you mean") as uncheckable, so it
#: never false-alarms on a pronoun.
WORD_NUMBER_MARK = "~"

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]
_CURRENCY = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{2}))?")
_PERCENT = re.compile(r"(\d[\d,.]*)\s?%")
_RANGE = re.compile(r"(\d[\d,.]*)\s?[-\u2013\u2014]\s?(\d[\d,.]*)")  # hyphen, en dash, em dash
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def canonicalize(text: str, *, mark_word_numbers: bool = False) -> str:
    """Lowercase, numbers as digits, currency as "<n> dollars", ranges as "a to b"."""

    def currency(match: re.Match[str]) -> str:
        dollars = match.group(1).replace(",", "")
        cents = match.group(2)
        if cents and cents != "00":
            return f"{dollars} dollars and {int(cents)} cents"
        return f"{dollars} dollars"

    def words(match: re.Match[str]) -> str:
        value = _parse_number_words(match.group(0))
        return f"{WORD_NUMBER_MARK if mark_word_numbers else ''}{value}"

    # ISO dates FIRST. Otherwise the range rule reads "2026-11-14" as
    # "2026 to 11-14", and an event date returned by a tool can never ground
    # the agent saying "your show is on November 14".
    out = _ISO_DATE.sub(
        lambda m: f"{_MONTHS[int(m.group(2)) - 1]} {int(m.group(3))} {m.group(1)}", text
    )
    out = _CURRENCY.sub(currency, out)
    out = _PERCENT.sub(lambda m: f"{m.group(1)} percent", out)
    out = _RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", out)
    out = _THOUSANDS.sub("", out)
    out = _NUMBER_RUN.sub(words, out)
    out = out.lower()
    out = re.sub(r"\bper cent\b", "percent", out)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# to_spoken — for SPEECH
# ---------------------------------------------------------------------------
# One tag or a group: [K1] · [K2, K3] · [K2,K3] · [K2][K3]. Claude writes the
# first form; Gemini writes the second, which the old single-tag pattern left
# in the text — and the caller heard "K two, K three".
_CITATION = re.compile(r"\s*\[\s*K\s*\d+(?:\s*[,;]\s*K?\s*\d+)*\s*\]")
_CITED_NUMBER = re.compile(r"K?(\d+)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_ORDER_ID = re.compile(r"\b([A-Z]{2})-(\d{4,8})\b")
_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b"
)
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s?(am|pm|a\.m\.|p\.m\.)?", re.IGNORECASE)
_HASH_NUMBER = re.compile(r"#\s?(\d+)")
_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_INTEGER = re.compile(r"\b\d+\b")

#: Spelled letter by letter. Anything else all-caps is left for TTS to read.
_ACRONYMS = {"ADA", "ID", "PDF", "VIP", "CPAP", "SMS", "PIN", "ATM", "FAQ", "QR"}
_DIGIT_WORDS = {str(d): _SMALL[d] for d in range(10)}


def _spell_digits(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[d] for d in digits)


def strip_citations(text: str) -> tuple[str, list[int]]:
    """Remove [K2] tags and return the 1-based tag numbers that were cited.

    The model is told to cite; a caller must never hear "K two".
    """
    tags = [int(n) for group in _CITATION.findall(text) for n in _CITED_NUMBER.findall(group)]
    return re.sub(r"\s+([.,!?;:])", r"\1", _CITATION.sub("", text)).strip(), tags


def to_spoken(text: str) -> str:
    out, _ = strip_citations(text)
    out = _URL.sub("", out)

    def email(match: re.Match[str]) -> str:
        local, domain = match.group(0).split("@", 1)
        spoken_local = local.replace(".", " dot ").replace("_", " underscore ").replace(
            "-", " dash "
        ).replace("+", " plus ")
        return f"{spoken_local} at {domain.replace('.', ' dot ')}"

    out = _EMAIL.sub(email, out)
    # Order ids: letters spelled, digits one at a time, a pause between.
    out = _ORDER_ID.sub(lambda m: f"{' '.join(m.group(1))}, {_spell_digits(m.group(2))}", out)
    out = _ISO_DATE.sub(
        lambda m: f"{_MONTHS[int(m.group(2)) - 1]} {ordinal_words(int(m.group(3)))}, "
        f"{year_words(int(m.group(1)))}",
        out,
    )
    out = _MONTH_DAY.sub(lambda m: f"{m.group(1)} {ordinal_words(int(m.group(2)))}", out)

    def currency(match: re.Match[str]) -> str:
        dollars = int(match.group(1).replace(",", ""))
        cents = int(match.group(2) or 0)
        unit = "dollar" if dollars == 1 else "dollars"
        spoken = f"{number_to_words(dollars)} {unit}"
        if cents:
            spoken += f" and {number_to_words(cents)} cent{'s' if cents != 1 else ''}"
        return spoken

    out = _CURRENCY.sub(currency, out)
    out = _PERCENT.sub(lambda m: f"{m.group(1)} percent", out)
    out = _RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", out)

    def time(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = (match.group(3) or "").replace(".", "").upper()
        spoken = number_to_words(hour)
        if minute:
            spoken += f" {'oh ' if minute < 10 else ''}{number_to_words(minute)}"
        if suffix:
            spoken += f" {' '.join(suffix)}"
        return spoken

    out = _TIME.sub(time, out)
    out = _HASH_NUMBER.sub(lambda m: f"number {number_to_words(int(m.group(1)))}", out)
    # Years BEFORE stripping thousands commas, so "2,000 cars" is a quantity
    # and never becomes a four-digit "year".
    out = _YEAR.sub(lambda m: year_words(int(m.group(1))), out)
    out = _THOUSANDS.sub("", out)
    out = _DECIMAL.sub(
        lambda m: f"{number_to_words(int(m.group(1)))} point {_spell_digits(m.group(2))}", out
    )
    out = _INTEGER.sub(lambda m: number_to_words(int(m.group(0))), out)
    out = re.sub(
        r"\b[A-Z]{2,5}\b", lambda m: " ".join(m.group(0)) if m.group(0) in _ACRONYMS else m.group(0), out
    )
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()
