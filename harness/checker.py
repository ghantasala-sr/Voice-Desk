"""The sentence-level grounding checker.

Streaming means a whole answer cannot be checked before it is spoken, so each
segment is checked before it reaches TTS. Deterministic, under 5 ms, and makes
NO LLM call (docs/00 D10).

Two rules:

  Rule 1  every number, price, duration, date, order id, email, proper noun
          and policy term in the segment must appear in a grounding source.
  Rule 2  a policy claim ("you can…", "we never…") with nothing to anchor it,
          little vocabulary in common with the retrieved chunks, and weak
          retrieval, is an unsupported claim.

Grounding sources, in order: retrieved chunk text -> tool results this turn ->
everything the caller has said this call. Repeating back the caller's own
order number is not a hallucination.

The checker grounds against the CHUNK TEXT itself, never against what the
model claims to have cited. Self-reported citations are an evaluation
convenience, not an enforcement mechanism.

Declared coverage gaps — reported with their numbers in week 3, not hidden:

  * NEGATION. An inverted claim ("refunds are NOT available" where
the chunk says they are) shares every entity and most vocabulary with its
    chunk, and no deterministic rule here catches it.
  * BINDING. Rule 1 checks that a number is PRESENT in the retrieved text,
    not that it is attached to the right claim. "Refunds take 10 business
    days" passes against a chunk that says "if more than 10 business days
    have passed since that email" — same number, different fact.
"""

from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from harness.normalize import WORD_NUMBER_MARK, canonicalize

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"

VerdictKind = Literal[
    "pass", "no_checkable_content", "fail_ungrounded_entity", "fail_unsupported_claim"
]
SpanType = Literal[
    "number", "currency", "percent", "duration", "date", "time",
    "order_id", "email", "proper_noun", "policy_term",
]


@dataclass(frozen=True)
class Span:
    type: SpanType
    surface: str
    normalized: str
    grounded: bool
    source: str | None = None   # "chunk" | "tool" | "caller" | "lexicon"


@dataclass(frozen=True)
class Verdict:
    verdict: VerdictKind
    spans: list[Span]
    reason: str
    elapsed_ms: float

    @property
    def passed(self) -> bool:
        return self.verdict in ("pass", "no_checkable_content")


# ---------------------------------------------------------------------------
# Units — a number is only as checkable as its unit is known
# ---------------------------------------------------------------------------
_UNITS = {
    "dollar": "dollar", "dollars": "dollar", "cent": "cent", "cents": "cent",
    "percent": "percent",
    "day": "day", "days": "day", "hour": "hour", "hours": "hour",
    "minute": "minute", "minutes": "minute", "week": "week", "weeks": "week",
    "month": "month", "months": "month", "year": "year", "years": "year",
    "inch": "inch", "inches": "inch", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml", "mile": "mile", "miles": "mile",
    "space": "space", "spaces": "space", "car": "car", "cars": "car",
    "people": "person", "person": "person", "ticket": "ticket", "tickets": "ticket",
    "item": "item", "items": "item", "seat": "seat", "seats": "seat",
    "order": "order", "orders": "order",
}
_DURATION_UNITS = {"day", "hour", "minute", "week", "month", "year"}
_TWO_WORD_UNITS = {("business", "day"): "business_day", ("business", "days"): "business_day"}

#: Includes JSON punctuation: tool results arrive as '{"section": 112}'.
_TOKEN_PUNCT = ".,;:!?()[]{}\"'"
_NUMBER_TOKEN = re.compile(rf"({re.escape(WORD_NUMBER_MARK)}?)(\d+(?:\.\d+)?)")
_ORDER_ID = re.compile(r"\b[A-Z]{2}-\d{4,8}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTH_DAY = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+~?(\d{1,2})\b"
)
_ISO_DATE = re.compile(r"\b\d{4}-(\d{2})-(\d{2})\b")
_CAPITALIZED_RUN = re.compile(r"\b[A-Z][\w'\u2019-]*(?:\s+(?:&\s+)?[A-Z][\w'\u2019-]*)*")

#: Capitalized words that are not entities. Sentence starters, pronouns,
#: weekdays and months — the false positives that would otherwise make the
#: fallback line fire on perfectly good answers.
_NOT_ENTITIES = {
    "i", "i'm", "i'll", "i've", "i'd", "a", "an", "the", "if", "you", "your", "you'll",
    "you're", "we", "we'll", "we're", "our", "it", "it's", "that", "that's", "this",
    "these", "those", "there", "there's", "yes", "no", "sure", "sorry", "hi", "hello",
    "okay", "ok", "thanks", "thank", "let", "let's", "please", "one", "and", "but",
    "so", "or", "for", "to", "in", "on", "at", "of", "after", "before", "when",
    "once", "then", "also", "just", "unfortunately", "great", "good", "of course",
    # Sentence-opening function words. Without these "Could I get your order
    # number?" reads as the proper noun "Could I" and fails the checker — found
    # by the week-4 offline agent, missed by every week-3 known-good answer.
    "could", "would", "can", "will", "should", "may", "might", "must", "do", "does",
    "did", "is", "are", "was", "were", "have", "has", "had", "what", "which", "who",
    "how", "where", "why", "here", "absolutely", "certainly", "done", "no problem",
    "welcome", "happy", "glad", "me", "my", "mine", "yours", "their", "its",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    *_MONTHS,
}

_STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for from with without as is are
    was were be been being it its this that these those you your yours we our us they
    them their i me my he she his her can could will would should may might must do
    does did not no yes any all some each every there here what which who when where
    why how than too very just also only about into over under up down out off again
    get got have has had able ll re ve s t d m""".split()
)

_POLICY_MODALS = re.compile(
    r"\b(you can|you can't|you cannot|you may|you must|you'll need to|you need to|"
    r"you will|we will|we won't|we don't|we do not|we never|we always|is not allowed|"
    r"are not allowed|is free|are free|at no charge|any ?time|always|never|as long as|"
    r"guaranteed|no problem)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lexicon:
    """Known real entities and policy terms for a domain.

    A capitalized run that is NOT here and NOT in any grounding source is a
    strong hallucination signal — an invented venue name, for instance.
    """

    entities: frozenset[str]
    policy_terms: frozenset[str]
    #: Stems that appear in a large share of all corpus chunks ("event",
    #: "ticket", "order"). They match everything, so they prove nothing about
    #: whether a claim came from its chunk, and Rule 2 ignores them — the same
    #: reason BM25 down-weights high document-frequency terms.
    common_words: frozenset[str] = frozenset()

    @classmethod
    def empty(cls) -> Lexicon:
        return cls(frozenset(), frozenset())


@functools.lru_cache(maxsize=8)
def load_lexicon(domain: str) -> Lexicon:
    root = DOMAINS_ROOT / domain
    manifest = yaml.safe_load((root / "corpus_manifest.yaml").read_text(encoding="utf-8"))
    prompts = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))

    entities: set[str] = {v.lower() for v in manifest.get("venues", [])}
    entities.add(str(prompts.get("company", "")).lower())
    # Every multi-word capitalized run in the corpus is a known entity: venue
    # names, payment brands, product names. Single words are too noisy.
    for path in (root / "corpus").glob("*.md"):
        for run in _CAPITALIZED_RUN.findall(path.read_text(encoding="utf-8")):
            if len(run.split()) >= 2:
                entities.add(run.lower())
    entities.discard("")
    return Lexicon(
        entities=frozenset(entities),
        policy_terms=frozenset(t.lower() for t in manifest.get("policy_terms", [])),
        common_words=_common_words(root, manifest),
    )


#: A stem in at least this share of chunks is domain-generic.
COMMON_WORD_DF = 0.20


def _common_words(root: Path, manifest: dict) -> frozenset[str]:
    from retrieval.chunk import chunk_document

    chunk_words: list[set[str]] = []
    for entry in manifest["documents"]:
        text = (root / "corpus" / entry["path"]).read_text(encoding="utf-8")
        for chunk in chunk_document(text, title=entry["title"]):
            chunk_words.append(_content_words(chunk.text))
    if not chunk_words:
        return frozenset()
    counts: dict[str, int] = {}
    for words in chunk_words:
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    cutoff = COMMON_WORD_DF * len(chunk_words)
    return frozenset(word for word, n in counts.items() if n >= cutoff)


# ---------------------------------------------------------------------------
# Grounding sources
# ---------------------------------------------------------------------------
@dataclass
class GroundingSources:
    chunks: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    caller: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Index:
    """Pre-extracted facts from the grounding sources, built once per turn."""

    strict_pairs: frozenset[tuple[str, str]]   # (number, unit) from chunks
    strict_numbers: frozenset[str]             # unitless numbers from chunks
    loose_numbers: frozenset[str]              # any number from tools/caller
    dates: frozenset[str]                      # "MM-DD"
    chunk_text: str                            # canonical, joined
    tool_text: str
    caller_text: str
    chunk_words: frozenset[str]                # stemmed content words


def _stem(word: str) -> str:
    """Deliberately crude. It only has to make 'refunds' meet 'refunded'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r"[a-z][a-z'-]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _number_unit_pairs(canonical: str) -> tuple[list[tuple[str, str | None, bool]], set[str]]:
    """Every number in canonical text, with its unit and whether it was a word.

    A range "5 to 7 business days" gives the unit to BOTH ends, so an answer
    quoting "within 7 business days" or "at least 5 business days" grounds.
    """
    tokens = canonical.split()
    found: list[tuple[str, str | None, bool]] = []
    dates: set[str] = set()

    for match in _MONTH_DAY.finditer(canonical):
        dates.add(f"{_MONTHS[match.group(1)]:02d}-{int(match.group(2)):02d}")
    for match in _ISO_DATE.finditer(canonical):
        dates.add(f"{match.group(1)}-{match.group(2)}")

    for index, token in enumerate(tokens):
        match = _NUMBER_TOKEN.fullmatch(token.strip(_TOKEN_PUNCT))
        if not match:
            continue
        number = match.group(2)
        number = number.rstrip("0").rstrip(".") if "." in number else number
        unit = _unit_at(tokens, index + 1)
        if unit is None and index + 2 < len(tokens) and tokens[index + 1] in ("to", "or", "and"):
            if _NUMBER_TOKEN.fullmatch(tokens[index + 2].strip(_TOKEN_PUNCT)):
                unit = _unit_at(tokens, index + 3)
        found.append((number, unit, bool(match.group(1))))
    return found, dates


def _unit_at(tokens: list[str], index: int) -> str | None:
    if index >= len(tokens):
        return None
    word = tokens[index].strip(_TOKEN_PUNCT)
    if index + 1 < len(tokens):
        pair = (word, tokens[index + 1].strip(_TOKEN_PUNCT))
        if pair in _TWO_WORD_UNITS:
            return _TWO_WORD_UNITS[pair]
    return _UNITS.get(word)


def _build_index(sources: GroundingSources) -> _Index:
    chunk_text = canonicalize(" \n ".join(sources.chunks))
    tool_text = canonicalize(" \n ".join(sources.tool_results))
    caller_text = canonicalize(" \n ".join(sources.caller))

    chunk_numbers, chunk_dates = _number_unit_pairs(chunk_text)
    loose: set[str] = set()
    other_dates: set[str] = set()
    for text in (tool_text, caller_text):
        numbers, dates = _number_unit_pairs(text)
        loose.update(n for n, _, _ in numbers)
        other_dates.update(dates)

    return _Index(
        strict_pairs=frozenset((n, u) for n, u, _ in chunk_numbers if u),
        strict_numbers=frozenset(n for n, _, _ in chunk_numbers),
        loose_numbers=frozenset(loose),
        dates=frozenset(chunk_dates | other_dates),
        chunk_text=chunk_text,
        tool_text=tool_text,
        caller_text=caller_text,
        chunk_words=frozenset(_content_words(chunk_text)),
    )


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------
class Checker:
    """Constructed once per turn from that turn's grounding sources."""

    def __init__(
        self,
        sources: GroundingSources,
        *,
        lexicon: Lexicon | None = None,
        retrieval_sufficient: bool = True,
        claim_overlap_min: float = 0.25,
    ) -> None:
        self._index = _build_index(sources)
        self._lexicon = lexicon or Lexicon.empty()
        self._retrieval_sufficient = retrieval_sufficient
        self._claim_overlap_min = claim_overlap_min

    def check(self, segment: str) -> Verdict:
        started = time.perf_counter()
        spans = self._extract_spans(segment)

        ungrounded = [s for s in spans if not s.grounded]
        if ungrounded:
            listed = ", ".join(f"{s.type}:{s.surface!r}" for s in ungrounded)
            return self._verdict("fail_ungrounded_entity", spans, f"ungrounded {listed}", started)

        # Rule 2 still applies when the only grounded spans are names the
        # LEXICON knows. "You must wear smart dress at Lumen Hall" names a real
        # venue, which proves the venue exists — not the rule. Measured in
        # results/2026-10-13-checker-catch-rate/.
        anchored = any(s.source != "lexicon" for s in spans)
        if not anchored:
            claim = self._unsupported_claim(segment)
            if claim:
                return self._verdict("fail_unsupported_claim", spans, claim, started)
        if spans:
            return self._verdict("pass", spans, "all spans grounded", started)
        return self._verdict("no_checkable_content", spans, "nothing to check", started)

    # -- Rule 1 -------------------------------------------------------------
    def _extract_spans(self, segment: str) -> list[Span]:
        index = self._index
        spans: list[Span] = []

        for surface in _ORDER_ID.findall(segment):
            spans.append(self._literal("order_id", surface))
        for surface in _EMAIL.findall(segment):
            spans.append(self._literal("email", surface))

        canonical = canonicalize(
            _EMAIL.sub(" ", _ORDER_ID.sub(" ", segment)), mark_word_numbers=True
        )
        numbers, dates = _number_unit_pairs(canonical)

        for date in dates:
            grounded = date in index.dates
            spans.append(Span("date", date, date, grounded, "chunk" if grounded else None))

        date_days = {d.split("-")[1].lstrip("0") for d in dates}
        for number, unit, was_word in numbers:
            # A bare word-number with no unit ("which one did you mean") is a
            # pronoun as often as a quantity. Not checkable; never a failure.
            if was_word and unit is None:
                continue
            if unit is None and number in date_days:
                continue  # the day-of-month, already checked as a date
            spans.append(self._number_span(number, unit))

        spans.extend(self._entity_spans(segment))
        for term in self._lexicon.policy_terms:
            # Plurals count: "service fees" is the policy term "service fee".
            if re.search(rf"\b{re.escape(term)}(?:s|es)?\b", segment, re.IGNORECASE):
                grounded = term in index.chunk_text
                spans.append(
                    Span("policy_term", term, term, grounded, "chunk" if grounded else None)
                )
        return spans

    def _number_span(self, number: str, unit: str | None) -> Span:
        index = self._index
        kind: SpanType = (
            "currency" if unit in ("dollar", "cent")
            else "percent" if unit == "percent"
            else "duration" if unit in _DURATION_UNITS or unit == "business_day"
            else "number"
        )
        surface = f"{number} {unit}" if unit else number
        if unit and (number, unit) in index.strict_pairs:
            return Span(kind, surface, surface, True, "chunk")
        if not unit and number in index.strict_numbers:
            return Span(kind, surface, surface, True, "chunk")
        # Tool results and caller speech carry no units ({"quantity": 2}), so
        # a number from them grounds whatever unit the answer attaches.
        if number in index.loose_numbers:
            return Span(kind, surface, surface, True, "tool_or_caller")
        return Span(kind, surface, surface, False)

    def _literal(self, kind: SpanType, surface: str) -> Span:
        needle = surface.lower()
        index = self._index
        for source, text in (("tool", index.tool_text), ("caller", index.caller_text),
                             ("chunk", index.chunk_text)):
            if needle in text:
                return Span(kind, surface, needle, True, source)
        # Callers spell order ids aloud: "T K 8 8 2 1 3".
        compact = re.sub(r"[\s-]", "", needle)
        if compact in re.sub(r"[\s-]", "", index.caller_text):
            return Span(kind, surface, needle, True, "caller")
        return Span(kind, surface, needle, False)

    def _entity_spans(self, segment: str) -> list[Span]:
        spans: list[Span] = []
        sentence_starts = {m.end() for m in re.finditer(r"(?:^|[.!?]\s+)", segment)}
        for match in _CAPITALIZED_RUN.finditer(segment):
            words = match.group(0).split()
            # Trim leading non-entity words ("The Foundry" keeps "The" only if
            # the lexicon knows the whole run, checked below).
            run = match.group(0)
            lowered = run.lower()
            if lowered in self._lexicon.entities:
                spans.append(Span("proper_noun", run, lowered, True, "lexicon"))
                continue
            # Strip non-entity words from BOTH ends: "Could I" is two function
            # words, not a name.
            while words and words[0].lower().strip("'\u2019") in _NOT_ENTITIES:
                words = words[1:]
            while words and words[-1].lower().strip("'\u2019") in _NOT_ENTITIES:
                words = words[:-1]
            if not words:
                continue
            if len(words) == 1:
                word = words[0]
                at_start = match.start() in sentence_starts and words[0] == match.group(0).split()[0]
                if at_start or word.isupper() or word.lower() in _NOT_ENTITIES:
                    continue  # sentence starters and acronyms: too noisy
            run = " ".join(words)
            lowered = run.lower()
            grounded_in = next(
                (
                    source
                    for source, text in (
                        ("lexicon", " | ".join(self._lexicon.entities)),
                        ("chunk", self._index.chunk_text),
                        ("tool", self._index.tool_text),
                        ("caller", self._index.caller_text),
                    )
                    if lowered in text
                ),
                None,
            )
            spans.append(Span("proper_noun", run, lowered, grounded_in is not None, grounded_in))
        return spans

    # -- Rule 2 -------------------------------------------------------------
    def _unsupported_claim(self, segment: str) -> str | None:
        """All four conditions are required, because each alone over-fires.

        1. a policy modal ("you can", "we never", "any time" ...)
        2. nothing anchors it: no span grounded by a chunk, a tool result or
           the caller (lexicon-known names do not count — see check())
        3. low vocabulary overlap with the retrieved chunks
        4. retrieval was weak
        """
        modal = _POLICY_MODALS.search(segment)
        if not modal:
            return None
        if self._retrieval_sufficient:
            return None
        words = _content_words(segment) - self._lexicon.common_words
        if not words:
            return None  # nothing distinctive enough to judge either way
        overlap = len(words & self._index.chunk_words) / len(words)
        if overlap >= self._claim_overlap_min:
            return None
        return (
            f"policy claim {modal.group(0)!r} with weak retrieval and "
            f"{overlap:.0%} vocabulary overlap with the retrieved chunks"
        )

    @staticmethod
    def _verdict(kind: VerdictKind, spans: list[Span], reason: str, started: float) -> Verdict:
        return Verdict(kind, spans, reason, (time.perf_counter() - started) * 1000)
