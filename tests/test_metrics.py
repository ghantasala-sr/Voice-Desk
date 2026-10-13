"""W3-5 — metrics.py is the only place a number is computed, so it is tested
against known values rather than trusted."""

from __future__ import annotations

import pytest

from eval.metrics import Proportion, entity_hits, percentile, precision_recall, wer, wilson_ci


@pytest.mark.parametrize(
    "k,n,low,high",
    [(8, 10, 0.4902, 0.9433), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (50, 100, 0.4038, 0.5962)],
)
def test_wilson_against_known_values(k, n, low, high) -> None:
    lo, hi = wilson_ci(k, n)
    assert lo == pytest.approx(low, abs=1e-3)
    assert hi == pytest.approx(high, abs=1e-3)


def test_wilson_with_no_data_is_uninformative() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_percentile_matches_postgres_percentile_cont() -> None:
    values = [10, 20, 30, 40]
    assert percentile(values, 50) == 25
    assert percentile(values, 95) == pytest.approx(38.5)
    assert percentile([], 50) is None


def test_precision_recall() -> None:
    result = precision_recall([True, True, False, False], [True, False, True, False])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"]["value"] == 0.5
    assert result["recall"]["value"] == 0.5
    assert result["precision"]["n"] == 2


def test_proportion_string_carries_n() -> None:
    assert "(8/10)" in str(Proportion(8, 10))


def test_wer_ignores_number_format_and_punctuation() -> None:
    assert wer("I need two tickets.", "i need 2 tickets") == 0.0
    assert wer("five to seven business days", "5-7 business days") == 0.0
    assert wer("send two tickets", "send to tickets") == pytest.approx(1 / 3)


def test_entities_are_scored_individually() -> None:
    entities = [{"type": "order_id", "value": "TK-88213"}, {"type": "email", "value": "sam@example.com"}]
    assert entity_hits(entities, "order T K 8 8 2 1 3, sam at example dot com") == (2, 2)
    assert entity_hits(entities, "order T K 8 8 2 1 3, sam at example dot org") == (1, 2)
    assert entity_hits(entities, "order TK 88 213 email sam at example.com") == (2, 2)
    assert entity_hits([], "anything") == (0, 0)


def test_spoken_identifiers_are_not_formatting_errors() -> None:
    """The reference is written as spoken; ASR writes it as text."""
    ref = "Order is T K 8 8 2 1 3 and the email is sam at example dot com."
    assert wer(ref, "Order is TK-88213 and the email is sam@example.com.") == 0.0
    assert wer("alex dot kim at example dot com", "alex.kim@example.com") == 0.0
    # A genuine mishearing still counts.
    assert wer(ref, "Order is TK-88214 and the email is sam@example.com.") > 0


def test_normalizer_keeps_words_next_to_spelled_ids() -> None:
    assert wer("I'm T K 5 1 9 0 0", "I'm TK51900") == 0.0
    assert wer("my order is T K 7 3 3 5 5", "my order is TK 73355") == 0.0
    assert wer("omar at example dot com", "omar at example.com") == 0.0


def test_spelling_variants_are_not_errors() -> None:
    assert wer("the Harbour Pavilion show", "the Harbor Pavilion show") == 0.0
