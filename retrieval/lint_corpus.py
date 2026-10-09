"""Enforce the corpus writing rules (docs/05 §1).

These are not style preferences. Each rule changes retrieval or speech quality:

  * One question per `##` section — the section is the retrieval unit, and a
    section answering two questions gets retrieved for one and confuses the
    model on the other.
  * 60-120 words — a SPOKEN answer budget. A 400-word chunk invites the model
    to summarize across facts, which is where ungrounded claims come from.
  * No tables, no nested lists — they chunk badly and read badly aloud.

Run: .venv/bin/python -m retrieval.lint_corpus --domain ticketing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from retrieval.chunk import TARGET_MAX, TARGET_MIN, parse_sections

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"

_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_NESTED_LIST = re.compile(r"^(?:\s{2,}|\t)[-*+]\s", re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")


def lint_domain(domain: str) -> list[str]:
    root = DOMAINS_ROOT / domain
    manifest = yaml.safe_load((root / "corpus_manifest.yaml").read_text(encoding="utf-8"))
    problems: list[str] = []

    listed = {entry["path"] for entry in manifest["documents"]}
    on_disk = {p.name for p in (root / "corpus").glob("*.md")}
    for missing in sorted(listed - on_disk):
        problems.append(f"{missing}: listed in the manifest but not on disk")
    for unlisted in sorted(on_disk - listed):
        problems.append(f"{unlisted}: on disk but not in the manifest")

    for entry in manifest["documents"]:
        path = root / "corpus" / entry["path"]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        name = entry["path"]

        if _TABLE.search(text):
            problems.append(f"{name}: contains a markdown table (chunks and reads badly aloud)")
        if _NESTED_LIST.search(text):
            problems.append(f"{name}: contains a nested list")
        if _MARKDOWN_LINK.search(text):
            problems.append(f"{name}: contains a markdown link (a URL cannot be read aloud)")

        sections = parse_sections(text)
        if not sections:
            problems.append(f"{name}: no '##' sections")

        for section in sections:
            label = f"{name} > {section.heading}"
            words = section.word_count
            if not (TARGET_MIN <= words <= TARGET_MAX):
                problems.append(f"{label}: {words} words (target {TARGET_MIN}-{TARGET_MAX})")
            if "?" in section.heading:
                problems.append(f"{label}: heading ends in a question mark; use a statement")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint a domain corpus.")
    parser.add_argument("--domain", default="ticketing")
    args = parser.parse_args()

    problems = lint_domain(args.domain)
    if problems:
        print(f"corpus lint: {len(problems)} problem(s)\n")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"corpus lint: {args.domain} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
