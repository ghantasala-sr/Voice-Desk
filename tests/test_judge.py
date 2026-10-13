"""W3-6 — the judge's own logic, with a fake client (no API key here)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from eval.judge import RUBRIC, agreement_rate, build_user_message, judge


class FakeClient:
    """Returns scripted JSON scores and records every request."""

    def __init__(self, scores: list[dict]) -> None:
        self._scores = list(scores)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        score = self._scores.pop(0)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(score))])


def score(c, ci, s):
    return {"correctness": c, "citation": ci, "safety": s, "rationale": "x"}


def test_safety_violation_never_reaches_the_judge() -> None:
    client = FakeClient([])
    result = judge(client, case_id="c", question="q", passages=[], answer="Your refund has been approved.",
                   must_not_contain=["refund has been approved"])
    assert result.hard_fail and "must_not_contain" in result.hard_fail
    assert client.requests == []


def test_median_of_three_and_agreement() -> None:
    client = FakeClient([score(2, 2, 2), score(2, 1, 2), score(1, 2, 2)])
    result = judge(client, case_id="c", question="q", passages=["p"], answer="a")
    assert result.scores == {"correctness": 2, "citation": 2, "safety": 2}
    assert not result.agreed
    assert len(client.requests) == 3


def test_dimension_order_is_shuffled_but_the_system_prompt_is_not() -> None:
    """Shuffling lives in the user message, so the cached rubric never changes."""
    client = FakeClient([score(2, 2, 2)] * 3)
    judge(client, case_id="case-x", question="q", passages=[], answer="a")
    systems = {r["system"][0]["text"] for r in client.requests}
    users = [r["messages"][0]["content"].splitlines()[0] for r in client.requests]
    assert systems == {RUBRIC}
    assert client.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert len(set(users)) > 1, "dimension order should vary across the three runs"


def test_judge_uses_a_different_model_than_the_agent() -> None:
    from core.settings import Settings

    client = FakeClient([score(2, 2, 2)] * 3)
    judge(client, case_id="c", question="q", passages=[], answer="a")
    assert client.requests[0]["model"] != Settings().llm_model


def test_invalid_scores_are_dropped_not_trusted() -> None:
    client = FakeClient([score(2, 2, 2), {"correctness": 5, "citation": 2, "safety": 2, "rationale": "x"},
                         score(2, 2, 2)])
    result = judge(client, case_id="c", question="q", passages=[], answer="a")
    assert result.scores["correctness"] == 2
    assert not result.agreed  # only 2 of 3 runs valid


def test_agreement_rate_ignores_hard_fails() -> None:
    a = judge(FakeClient([score(2, 2, 2)] * 3), case_id="a", question="q", passages=[], answer="x")
    b = judge(FakeClient([]), case_id="b", question="q", passages=[], answer="approved",
              must_not_contain=["approved"])
    assert agreement_rate([a, b]) == 1.0


def test_user_message_carries_passages_and_order() -> None:
    msg = build_user_message("where are my tickets", ["Mobile tickets..."], "In the app.",
                             ("safety", "correctness", "citation"))
    assert msg.startswith("Score the dimensions in this order: safety, correctness, citation.")
    assert "[K1] Mobile tickets..." in msg
