"""W2-1 / W2-2 — chunking and the corpus writing rules.

The rules are not style preferences. A section that answers two questions gets
retrieved for one and confuses the model on the other; a 400-word chunk invites
the model to summarize ACROSS facts, which is where ungrounded claims come
from (docs/05 §1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from retrieval.chunk import (
    MERGE_BELOW,
    SPLIT_ABOVE,
    TARGET_MAX,
    TARGET_MIN,
    chunk_document,
    parse_sections,
    strip_front_matter,
)
from retrieval.lint_corpus import lint_domain

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "domains" / "ticketing"


# ---------------------------------------------------------------------------
# The shipped corpus
# ---------------------------------------------------------------------------
def test_shipped_corpus_passes_its_own_linter() -> None:
    problems = lint_domain("ticketing")
    assert not problems, "corpus lint failures:\n  " + "\n  ".join(problems)


def test_manifest_holds_exactly_three_documents_back() -> None:
    """The week-4 feedback cycle has nothing to discover without these."""
    manifest = yaml.safe_load((CORPUS / "corpus_manifest.yaml").read_text())
    held = [d for d in manifest["documents"] if not d["ingested"]]

    assert len(held) == 3
    assert {d["path"] for d in held} == {
        "10-group-and-vip-packages.md",
        "11-weather-and-outdoor-events.md",
        "12-age-restrictions-and-id.md",
    }
    for document in held:
        assert document.get("held_out_reason"), f"{document['path']} must say why"


def test_every_shipped_chunk_is_a_spoken_answer_budget() -> None:
    manifest = yaml.safe_load((CORPUS / "corpus_manifest.yaml").read_text())
    for entry in manifest["documents"]:
        text = (CORPUS / "corpus" / entry["path"]).read_text()
        for chunk in chunk_document(text, title=entry["title"]):
            assert TARGET_MIN - 10 <= chunk.word_count <= TARGET_MAX, (
                f"{chunk.heading_path}: {chunk.word_count} words"
            )


def test_heading_path_is_prepended_to_the_embedded_text() -> None:
    """A chunk body is often meaningless alone: "This is not available for
    resale purchases" means nothing until its heading sits in front of it."""
    text = (CORPUS / "corpus" / "01-refunds-and-exchanges.md").read_text()
    chunk = chunk_document(text, title="Refunds and exchanges")[0]

    assert chunk.embed_text.startswith(chunk.heading_path)
    assert chunk.text in chunk.embed_text
    assert chunk.heading_path.startswith("Refunds and exchanges > ")


# ---------------------------------------------------------------------------
# The chunker's rules
# ---------------------------------------------------------------------------
def test_front_matter_is_stripped() -> None:
    front, body = strip_front_matter("---\ntitle: X\n---\n## Heading\nbody\n")
    assert "title: X" in front
    assert body.startswith("## Heading")


def test_document_title_heading_is_not_a_chunk() -> None:
    sections = parse_sections("# Doc title\n\n## Real section\n\nbody text here\n")
    assert [s.heading for s in sections] == ["Real section"]


def test_small_sections_merge_forward() -> None:
    doc = (
        "## Tiny\n\ntoo short\n\n"
        "## Real\n\n" + " ".join(["word"] * 70) + "\n"
    )
    chunks = chunk_document(doc, title="T")
    assert len(chunks) == 1
    assert "too short" in chunks[0].text


def test_oversized_sections_split_on_paragraphs_keeping_the_heading() -> None:
    long_body = "\n\n".join(" ".join(["word"] * 80) for _ in range(3))
    chunks = chunk_document(f"## Long one\n\n{long_body}\n", title="T")

    assert len(chunks) >= 2, "a 240-word section must be split"
    assert all(c.heading_path == "T > Long one" for c in chunks), (
        "the heading path repeats on every part, by design"
    )
    assert all(c.word_count <= SPLIT_ABOVE for c in chunks)


def test_unsplittable_long_section_stays_whole() -> None:
    """One long chunk beats a cut made in the wrong place."""
    chunks = chunk_document("## Long\n\n" + " ".join(["word"] * 200) + "\n", title="T")
    assert len(chunks) == 1


def test_chunks_are_sequentially_numbered() -> None:
    doc = "".join(f"## S{i}\n\n{' '.join(['word'] * 70)}\n\n" for i in range(4))
    assert [c.seq for c in chunk_document(doc, title="T")] == [0, 1, 2, 3]


@pytest.mark.parametrize("words", [MERGE_BELOW - 1, MERGE_BELOW + 1])
def test_merge_threshold_is_respected(words: int) -> None:
    doc = f"## A\n\n{' '.join(['word'] * words)}\n\n## B\n\n{' '.join(['word'] * 70)}\n"
    chunks = chunk_document(doc, title="T")
    assert len(chunks) == (1 if words < MERGE_BELOW else 2)
