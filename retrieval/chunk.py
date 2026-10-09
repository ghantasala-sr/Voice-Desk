"""Heading-aware chunking.

The retrieval unit is one `##` section, because a section answers one question.
Sliding windows are not used: overlap exists to rescue a window from cutting
mid-thought, heading sections already end at thoughts, and overlap would
duplicate facts across chunks — which inflates lexical scores and makes the
week-3 threshold calibration harder (docs/05 §2).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Section length bounds, in words.
TARGET_MIN = 60      # below this a section is probably two half-answers
TARGET_MAX = 120     # above this the model starts summarizing ACROSS facts
SPLIT_ABOVE = 140    # hard split, on paragraph boundaries
MERGE_BELOW = 25     # too small to stand alone; merge into the next sibling

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    seq: int
    heading_path: str
    text: str
    word_count: int

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        The heading path is prepended because a chunk body is often meaningless
        alone: "This is not available for resale purchases" means nothing until
        "Refunds and exchanges > Eligibility" sits in front of it.
        """
        return f"{self.heading_path}\n{self.text}"


@dataclass(frozen=True)
class Section:
    heading: str
    body: str

    @property
    def word_count(self) -> int:
        return len(_WORD.findall(self.body))


def strip_front_matter(markdown: str) -> tuple[str, str]:
    """Return (front_matter_text, body). Front matter is optional."""
    match = _FRONT_MATTER.match(markdown)
    if not match:
        return "", markdown
    return match.group(1), markdown[match.end():]


def parse_sections(markdown: str) -> list[Section]:
    """Split a document into its `##` sections, ignoring the title heading."""
    _, body = strip_front_matter(markdown)
    sections: list[Section] = []

    headings = list(_HEADING.finditer(body))
    for index, match in enumerate(headings):
        level = len(match.group(1))
        if level == 1:
            continue  # document title, not a retrieval unit
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        sections.append(
            Section(heading=match.group(2).strip(), body=body[match.end():end].strip())
        )
    return sections


def chunk_document(markdown: str, *, title: str) -> list[Chunk]:
    """Turn one document into chunks, applying the split and merge rules."""
    sections = parse_sections(markdown)
    merged = _merge_small(sections)

    chunks: list[Chunk] = []
    for section in merged:
        path = f"{title} > {section.heading}"
        for part in _split_large(section.body):
            chunks.append(
                Chunk(
                    seq=len(chunks),
                    heading_path=path,  # repeated on each part, by design
                    text=part,
                    word_count=len(_WORD.findall(part)),
                )
            )
    return chunks


def _merge_small(sections: list[Section]) -> list[Section]:
    """Fold a too-short section into the one that follows it."""
    out: list[Section] = []
    carry: Section | None = None

    for section in sections:
        if carry is not None:
            section = Section(heading=carry.heading, body=f"{carry.body}\n\n{section.body}")
            carry = None
        if section.word_count < MERGE_BELOW:
            carry = section
            continue
        out.append(section)

    if carry is not None:  # last section was short; keep it rather than lose it
        out.append(carry)
    return out


def _split_large(body: str) -> list[str]:
    """Split an over-long section into near-equal parts on paragraph breaks.

    Greedy packing rather than a single midpoint cut: a midpoint split of a
    240-word section leaves one part at 160 words, still over the limit. We
    aim for ceil(total / TARGET_MAX) parts and fill each up to that size.

    A single paragraph longer than the cap is left whole — one long chunk
    beats a cut made in the middle of a thought.
    """
    total = len(_WORD.findall(body))
    if total <= SPLIT_ABOVE:
        return [body]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if len(paragraphs) < 2:
        return [body]  # nothing to split on

    wanted = max(2, math.ceil(total / TARGET_MAX))
    per_part = total / wanted

    parts: list[str] = []
    current: list[str] = []
    running = 0

    for paragraph in paragraphs:
        words = len(_WORD.findall(paragraph))
        if current and running + words > per_part:
            parts.append("\n\n".join(current))
            current, running = [], 0
        current.append(paragraph)
        running += words

    if current:
        parts.append("\n\n".join(current))
    return parts
