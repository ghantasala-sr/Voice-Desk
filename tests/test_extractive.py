"""The offline agent's parsing — it drives every week-4 simulated call."""

from __future__ import annotations

from providers.llm.extractive import caller_email, parse_email, parse_order_id, parse_quantity


def test_spoken_order_ids() -> None:
    assert parse_order_id("It's order T K 8 8 2 1 3, sam") == "TK-88213"
    assert parse_order_id("tk-88213") == "TK-88213"
    assert parse_order_id("no order here") is None


def test_spoken_emails() -> None:
    assert parse_email("sam at example dot com") == "sam@example.com"
    assert parse_email("alex dot kim at example dot com") == "alex.kim@example.com"
    assert parse_email("lee plus tickets at example dot com") == "lee+tickets@example.com"


def test_the_callers_email_is_the_one_said_with_the_order() -> None:
    """Found by the simulator: a recipient and the caller's own email in one
    sentence made the agent verify with the recipient's, twice."""
    text = ("Can you transfer two tickets to friend@example.com? "
            "It's order T K 8 8 2 1 3, sam at example dot com.")
    assert caller_email(text) == "sam@example.com"


def test_quantities() -> None:
    assert parse_quantity("send two tickets") == 2
    assert parse_quantity("transfer 3 of my tickets") == 3


def test_policy_questions_about_my_tickets_are_not_lookups() -> None:
    from providers.llm.extractive import LOOKUP

    assert not LOOKUP.search("can i resell my tickets?")
    assert not LOOKUP.search("can i sell my tickets for more than i paid?")
    assert LOOKUP.search("where are my tickets?")
    assert LOOKUP.search("is my show still happening?")


async def test_a_corrected_order_number_is_the_one_verified() -> None:
    from providers.llm.base import LLMRequest
    from providers.llm.extractive import ExtractiveProvider

    msgs = [{"role": "user", "content": [{"type": "text", "text": "transfer two tickets to friend@example.com"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Could I get your order number?"}]},
            {"role": "user", "content": [{"type": "text", "text": "It's order T K 8 8 2 3 1, sam at example dot com."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I couldn't find that order number."}]},
            {"role": "user", "content": [{"type": "text", "text": "Sorry, T K 8 8 2 1 3, sam at example dot com."}]}]
    tools = [{"name": "verify_caller", "parameters": {}}]
    events = [e async for e in ExtractiveProvider().stream(
        LLMRequest(system_static="", system_dynamic="Caller verified: no", messages=msgs, tools=tools))]
    assert events[0].arguments == {"order_id": "TK-88213", "email": "sam@example.com"}
