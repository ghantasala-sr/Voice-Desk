"""Turning a token stream into checkable, speakable units.

Why this exists: streaming TTS normally starts on the first token chunk, but
the week-2 checker cannot run until a unit is syntactically complete. So the
segmenter decides how long we wait — and that wait is a real latency cost the
design chooses (stage [E] in docs/01 §5).

Boundaries, in priority order:

  1. Sentence-final `.` `!` `?` followed by whitespace — but not after an
     abbreviation ("Mr.", "No.", "approx.") and not inside a decimal ("5.50").
  2. `;` and `:`.
  3. Clause boundaries (", and", ", but", …) once the buffer passes
     CLAUSE_MIN_WORDS. Long sentences get checked in pieces instead of holding
     audio hostage.
  4. A hard cap at MAX_WORDS.

A unit shorter than MIN_WORDS is held and merged forward, so TTS never gets a
fragment that prosodically dangles.
"""

from __future__ import annotations

import re

CLAUSE_MIN_WORDS = 12
MAX_WORDS = 25
MIN_WORDS = 3

# Abbreviations whose trailing period is not a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "no", "approx", "etc", "vs", "e.g", "i.e", "inc", "ltd", "co",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "a.m", "p.m", "am", "pm",
}

_SENTENCE_END = re.compile(r"[.!?]")
_CLAUSE_MARK = re.compile(r",\s+(and|but|so|which|or|then)\b", re.IGNORECASE)
_WORD = re.compile(r"\S+")


class Segmenter:
    """Feed token deltas in, get complete segments out."""

    def __init__(
        self,
        *,
        clause_min_words: int = CLAUSE_MIN_WORDS,
        max_words: int = MAX_WORDS,
        min_words: int = MIN_WORDS,
    ) -> None:
        self._buffer = ""
        self._clause_min_words = clause_min_words
        self._max_words = max_words
        self._min_words = min_words

    @property
    def buffered(self) -> str:
        return self._buffer

    def feed(self, delta: str) -> list[str]:
        """Add text; return zero or more complete segments."""
        self._buffer += delta
        segments: list[str] = []

        while True:
            cut = self._find_boundary(self._buffer)
            if cut is None:
                break
            candidate = self._buffer[:cut].strip()
            remainder = self._buffer[cut:]

            # Too short to speak on its own — keep accumulating. We cannot
            # look at what follows to decide this, because in a stream it has
            # not arrived yet. So we hold the boundary and wait; flush() emits
            # the fragment if the stream ends first.
            if _word_count(candidate) < self._min_words:
                break

            self._buffer = remainder
            if candidate:
                segments.append(candidate)

        return segments

    def flush(self) -> str | None:
        """Whatever is left when the stream ends."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None

    # -- boundary detection -------------------------------------------------

    def _find_boundary(self, text: str) -> int | None:
        """Index just past the first boundary in `text`, or None."""
        # 1. sentence-final punctuation
        for match in _SENTENCE_END.finditer(text):
            index = match.start()
            if _is_decimal_point(text, index) or _is_abbreviation(text, index):
                continue
            after = index + 1
            # Require whitespace after, so we do not cut a sentence that is
            # still arriving token by token.
            if after < len(text) and text[after].isspace():
                return after
            if after == len(text):
                return None  # might be an abbreviation still being typed

        # 2. semicolon / colon
        for mark in (";", ":"):
            index = text.find(mark)
            if index != -1 and index + 1 < len(text):
                return index + 1

        # 3. clause boundary, only once the buffer is long enough to be worth
        #    splitting. Below that, splitting costs prosody and buys nothing.
        if _word_count(text) >= self._clause_min_words:
            match = _CLAUSE_MARK.search(text)
            if match:
                return match.start() + 1  # keep the comma with the left side

        # 4. hard cap
        words = list(_WORD.finditer(text))
        if len(words) > self._max_words:
            return words[self._max_words].start()

        return None


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _is_decimal_point(text: str, index: int) -> bool:
    """`5.50` — digit on both sides of the period."""
    if text[index] != ".":
        return False
    before = text[index - 1] if index > 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    return before.isdigit() and after.isdigit()


def _is_abbreviation(text: str, index: int) -> bool:
    """`Mr.` / `No.` / `approx.` — the period belongs to the word."""
    if text[index] != ".":
        return False
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:index].lower().rstrip(".")
    return token in _ABBREVIATIONS
