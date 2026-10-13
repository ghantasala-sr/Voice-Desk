"""W3-1 — the eval set is well-formed and has the composition the plan says."""

from __future__ import annotations

from pathlib import Path

from eval.dataset import GROUPS, all_cases, composition, load_cases, load_escalation_cases

CORPUS = Path(__file__).resolve().parent.parent / "domains" / "ticketing" / "corpus"


def test_main_set_composition_matches_the_plan() -> None:
    cases = load_cases()
    assert len(cases) == 50
    assert dict(composition(cases)) == GROUPS


def test_every_answer_case_has_a_citation_and_a_safety_check_where_it_matters() -> None:
    for case in load_cases():
        if case.group == "covered":
            assert case.citation_docs, f"{case.id}: covered cases need citation_docs"
            for doc in case.citation_docs:
                assert (CORPUS / doc).exists(), f"{case.id}: no such document {doc}"


def test_handoff_set_has_at_least_forty_positives() -> None:
    """docs/00 D13: 15 positives is a ±20-point interval. 40 is the floor."""
    positives = [c for c in all_cases() if c.should_escalate]
    negatives = [c for c in all_cases() if not c.should_escalate]
    assert len(positives) >= 40
    assert len(negatives) >= 30


def test_retrieval_labels_give_both_classes() -> None:
    labels = [c.retrieval for c in all_cases()]
    assert labels.count("covered") >= 25
    assert labels.count("uncovered") >= 20


def test_held_out_cases_point_at_documents_that_are_really_held_out() -> None:
    import yaml

    manifest = yaml.safe_load((CORPUS.parent / "corpus_manifest.yaml").read_text())
    held = {d["path"] for d in manifest["documents"] if not d["ingested"]}
    raw = yaml.safe_load((Path("eval/dataset/cases.yaml")).read_text())
    for entry in raw["cases"]:
        if entry.get("held_out_doc"):
            assert entry["held_out_doc"] in held


def test_escalation_set_loads() -> None:
    assert len(load_escalation_cases()) == 44
