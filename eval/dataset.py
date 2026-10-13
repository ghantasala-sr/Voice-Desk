"""Loading and validating the eval set.

Validation is strict on purpose. A mislabelled case corrupts every number
computed from it, and a typo in a label ("shoud_escalate") would otherwise be
silently read as False.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATASET = Path(__file__).resolve().parent / "dataset"

GROUPS = {
    "covered": 14, "lookup": 8, "action": 6, "held_out": 6,
    "hard_rule": 7, "judgment": 5, "out_of_scope": 4,
}
RETRIEVAL_LABELS = {"covered", "uncovered", "n/a"}
EXPECTED_TYPES = {"answer", "tool", "handoff"}
CASE_KEYS = {"id", "group", "intent", "utterance", "expected", "retrieval", "difficulty",
             "key_entities", "held_out_doc", "verified_as"}
EXPECTED_KEYS = {"type", "citation_docs", "must_contain", "must_not_contain", "should_escalate",
                 "escalate_reason", "tool_name", "tool_args", "requires_readback", "tool_outcome"}


@dataclass(frozen=True)
class Case:
    id: str
    utterance: str
    should_escalate: bool
    retrieval: str
    group: str = "escalation"
    intent: str = ""
    expected: dict[str, Any] = field(default_factory=dict)
    key_entities: list[dict[str, str]] = field(default_factory=list)
    source_file: str = "cases.yaml"

    @property
    def citation_docs(self) -> list[str]:
        return list(self.expected.get("citation_docs", []))


def load_cases(path: Path | None = None) -> list[Case]:
    raw = yaml.safe_load((path or DATASET / "cases.yaml").read_text(encoding="utf-8"))
    cases = []
    for entry in raw["cases"]:
        unknown = set(entry) - CASE_KEYS
        if unknown:
            raise ValueError(f"{entry.get('id')}: unknown keys {sorted(unknown)}")
        expected = entry["expected"]
        bad = set(expected) - EXPECTED_KEYS
        if bad:
            raise ValueError(f"{entry['id']}: unknown expected keys {sorted(bad)}")
        if expected["type"] not in EXPECTED_TYPES:
            raise ValueError(f"{entry['id']}: bad expected.type {expected['type']!r}")
        if "should_escalate" not in expected:
            raise ValueError(f"{entry['id']}: should_escalate is required")
        if entry["retrieval"] not in RETRIEVAL_LABELS:
            raise ValueError(f"{entry['id']}: bad retrieval label {entry['retrieval']!r}")
        cases.append(Case(
            id=entry["id"], utterance=entry["utterance"],
            should_escalate=bool(expected["should_escalate"]), retrieval=entry["retrieval"],
            group=entry["group"], intent=entry.get("intent", ""), expected=expected,
            key_entities=entry.get("key_entities", []),
        ))
    _assert_unique(cases)
    return cases


def load_escalation_cases(path: Path | None = None) -> list[Case]:
    raw = yaml.safe_load(
        (path or DATASET / "cases_escalation.yaml").read_text(encoding="utf-8")
    )
    cases = []
    for entry in raw["cases"]:
        if entry["retrieval"] not in RETRIEVAL_LABELS:
            raise ValueError(f"{entry['id']}: bad retrieval label")
        cases.append(Case(
            id=entry["id"], utterance=entry["utterance"],
            should_escalate=bool(entry["should_escalate"]), retrieval=entry["retrieval"],
            expected={"escalate_reason": entry.get("reason")},
            source_file="cases_escalation.yaml",
        ))
    _assert_unique(cases)
    return cases


def all_cases() -> list[Case]:
    cases = load_cases() + load_escalation_cases()
    _assert_unique(cases)
    return cases


def composition(cases: list[Case]) -> Counter[str]:
    return Counter(case.group for case in cases)


def _assert_unique(cases: list[Case]) -> None:
    duplicates = [i for i, n in Counter(c.id for c in cases).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
