"""The only place a number is computed. docs/12 §8.

Everything downstream — run READMEs, the report, resume bullets — reads the
`metrics.json` these functions produce, so a metric cannot drift between a
notebook and a report. Every proportion carries `n` and a Wilson interval; a
bare proportion is not permitted (docs/15 §4).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Sane at n=5 and at p=0 or 1, unlike the normal
    approximation — which is the whole reason to use it on small eval sets."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int

    @property
    def value(self) -> float | None:
        return self.successes / self.n if self.n else None

    def as_dict(self) -> dict[str, Any]:
        low, high = wilson_ci(self.successes, self.n)
        return {
            "value": round(self.value, 4) if self.value is not None else None,
            "n": self.n, "successes": self.successes,
            "ci95": [round(low, 4), round(high, 4)],
        }

    def __str__(self) -> str:
        if not self.n:
            return "—  (n=0)"
        low, high = wilson_ci(self.successes, self.n)
        return f"{self.value:.2f} [{low:.2f}, {high:.2f}]  ({self.successes}/{self.n})"


def precision_recall(predicted: list[bool], actual: list[bool]) -> dict[str, Any]:
    tp = sum(p and a for p, a in zip(predicted, actual, strict=True))
    fp = sum(p and not a for p, a in zip(predicted, actual, strict=True))
    fn = sum(a and not p for p, a in zip(predicted, actual, strict=True))
    precision = Proportion(tp, tp + fp)
    recall = Proportion(tp, tp + fn)
    p, r = precision.value or 0.0, recall.value or 0.0
    return {
        "precision": precision.as_dict(),
        "recall": recall.as_dict(),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "n": len(actual), "positives": sum(actual),
    }


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation, matching Postgres percentile_cont."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def percentiles(values: list[float], ps: tuple[int, ...] = (50, 95)) -> dict[str, Any]:
    return {"n": len(values), **{f"p{p}": percentile(values, p) for p in ps}}


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s@.+'-]")


_SPOKEN_EMAIL = re.compile(
    r"\b([a-z0-9]+(?: (?:dot|plus) [a-z0-9]+)*) at ([a-z0-9]+)(?: dot |\.)(com|org|net)\b"
)
# A run of single letters/digits, NOT glued to a preceding word ("i'm t k 5"
# must keep "i'm").
_SPELLED_RUN = re.compile(r"(?<![\w'])(?:[a-z0-9] ){2,}[a-z0-9]\b")
_ORDER_ID = re.compile(r"\btk ?-?(\d{5})\b")
#: British/American spellings. Not recognition errors — Whisper writes US
#: spelling, the corpus and the references are British.
_SPELLING = {"harbour": "harbor", "theatre": "theater", "amphitheatre": "amphitheater",
             "licence": "license", "recognise": "recognize", "colour": "color",
             "favourite": "favorite", "cancelled": "canceled", "millilitres": "milliliters"}


def normalize_transcript(text: str) -> str:
    """Lowercase, numbers as digits, punctuation out — so "Two tickets." and
    "2 tickets" do not count as a word error.

    Spoken identifiers are folded the same way on BOTH sides. A reference of
    "T K 8 8 2 1 3, sam at example dot com" and a hypothesis of
    "TK-88213, sam@example.com" are the same words; scoring them as five
    errors would measure formatting, not recognition.
    """
    from harness.normalize import canonicalize

    out = canonicalize(text)
    out = _SPOKEN_EMAIL.sub(
        lambda m: f"{m.group(1).replace(' dot ', '.').replace(' plus ', '+')}@{m.group(2)}.{m.group(3)}",
        out,
    )
    out = _PUNCT.sub(" ", out)
    # Keep "." inside emails and decimals; drop it at the end of a sentence.
    out = re.sub(r"\.(?!\w)", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # "t k 8 8 2 1 3" -> "tk88213", and "tk-88213" / "tk 88213" -> "tk88213".
    out = _SPELLED_RUN.sub(lambda m: m.group(0).replace(" ", ""), out)
    out = _ORDER_ID.sub(r"tk\1", out)
    return " ".join(_SPELLING.get(word, word) for word in out.split())


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref, hyp = normalize_transcript(reference), normalize_transcript(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def entity_hits(entities: list[dict[str, str]], hypothesis: str) -> tuple[int, int]:
    """(correct, total) — per ENTITY, not per utterance.

    Per-utterance scoring hides that one utterance had three entities and two
    were wrong, which is the common case at 5 dB. An entity counts as heard if
    its normalized value appears in the normalized transcript; order ids and
    emails are compared with spacing and "dot"/"at" collapsed, because callers
    spell them aloud and ASR renders that a dozen different ways.
    """
    if not entities:
        return (0, 0)
    flat = _flatten_spoken(hypothesis)
    correct = sum(_flatten_spoken(e["value"]) in flat for e in entities)
    return (correct, len(entities))


def _flatten_spoken(text: str) -> str:
    out = normalize_transcript(text)
    out = re.sub(r"\b(dot)\b", ".", out)
    out = re.sub(r"\b(at)\b", "@", out)
    out = re.sub(r"\b(plus)\b", "+", out)
    out = re.sub(r"\b(dash|hyphen)\b", "-", out)
    return re.sub(r"[\s-]", "", out)
