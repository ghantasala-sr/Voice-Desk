"""An offline, deterministic agent: extractive answers and rule-based tool use.

There is no API key in the development environment, and week 4 needs an agent
to talk to hundreds of simulated callers. This provider stands in for the
model. It is honest about what it is:

  * ANSWERS are the best-matching one or two sentences from the retrieved
    <knowledge> passages, quoted with their [K] tag — grounded by construction.
    When nothing relevant was retrieved it offers a colleague, as the system
    prompt instructs a real model to.
  * TOOLS are called by rule: verify when the caller gives an order number and
    email, look up the order, propose a transfer when asked.

So a run against it measures the MACHINERY around the model — retrieval,
tools, read-back, idempotency, handoff, the gap loop — not answer quality.
Every result produced with it says so. Set VD_LLM_PROVIDER=anthropic to use
Claude through exactly the same interface.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from providers.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)

_KNOWLEDGE = re.compile(r"\[K(\d+)\] ([^\n]*)\n(.*?)(?=\n\n\[K\d+\] |\n</knowledge>)", re.S)
_STOP = set("""a an the and or but if to of in on at for with is are was be can do does i
my me you your we our it this that what how when where why will would should
could have has get there their they them so just please hi hello""".split())
_EMAIL_SPOKEN = re.compile(
    r"\b([a-z0-9]+(?:\s+(?:dot|plus)\s+[a-z0-9]+)*)\s+at\s+([a-z0-9]+)(?:\s+dot\s+|\.)(com|org|net)\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.(?:com|org|net)\b")
_ORDER_SPOKEN = re.compile(r"\bt\s*k[\s-]*((?:\d\s*){5})\b")
_QUANTITY = re.compile(r"\b(one|two|three|four|five|1|2|3|4|5)\b(?:\s+\w+)?\s+(?:of my\s+)?tickets?\b")
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

TRANSFER = re.compile(r"\b(transfer|send|give)\b.*\btickets?\b|\btickets?\b.*\b(to my|over to)\b")
# Phrases that need the caller's ORDER. Deliberately not a bare "my tickets":
# "can I resell my tickets?" is a policy question, and treating it as a lookup
# made the agent ask for an order number nobody needed — found by the week-4
# gap report.
LOOKUP = re.compile(r"\b(where are my tickets|find my tickets|check my tickets|which seats|"
                    r"seats i've got|how many tickets|is my (show|event) still|still happening|"
                    r"still on\?|my event is still)\b")
ASK_DETAILS = "Could I get your order number and the email address on the order?"
NO_ANSWER = "I don't have that information, so let me get a colleague who can help."


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP and len(w) > 2}


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


def parse_order_id(text: str) -> str | None:
    match = _ORDER_SPOKEN.search(text.lower())
    return f"TK-{re.sub(r'[^0-9]', '', match.group(1))}" if match else None


def parse_email(text: str) -> str | None:
    lowered = text.lower()
    if found := _EMAIL.search(lowered):
        return found.group(0)
    if spoken := _EMAIL_SPOKEN.search(lowered):
        local = re.sub(r"\s+dot\s+", ".", spoken.group(1))
        local = re.sub(r"\s+plus\s+", "+", local)
        return f"{local}@{spoken.group(2)}.{spoken.group(3)}"
    return None


def caller_email(text: str) -> str | None:
    """The email said WITH the order number — callers say "order X, email Y",
    so it is the first email after the order id (or, failing that, the last
    one before it). Another email in the same sentence is a recipient."""
    lowered = text.lower()
    order = _ORDER_SPOKEN.search(lowered)
    if not order:
        return None
    found = sorted(_email_spans(lowered))
    after = [email for start, email in found if start > order.start()]
    before = [email for start, email in found if start < order.start()]
    return after[0] if after else (before[-1] if before else None)


def _email_spans(lowered: str) -> list[tuple[int, str]]:
    spans = [(m.start(), m.group(0)) for m in _EMAIL.finditer(lowered)]
    for m in _EMAIL_SPOKEN.finditer(lowered):
        local = re.sub(r"\s+plus\s+", "+", re.sub(r"\s+dot\s+", ".", m.group(1)))
        spans.append((m.start(), f"{local}@{m.group(2)}.{m.group(3)}"))
    return spans


def parse_quantity(text: str) -> int | None:
    match = _QUANTITY.search(text.lower())
    if not match:
        return None
    token = match.group(1)
    return _WORDS.get(token) or int(token)


class ExtractiveProvider(LLMProvider):
    name = "extractive"
    model = "extractive-v1"

    def __init__(self, **_: Any) -> None:
        self.requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(request)
        for event in self._respond(request):
            yield event

    # -- the decision ---------------------------------------------------------
    def _respond(self, request: LLMRequest) -> list[LLMEvent]:
        messages = request.messages
        last = messages[-1] if messages else {"content": []}
        results = [b for b in last.get("content", []) if isinstance(last.get("content"), list)
                   and b.get("type") == "tool_result"]
        if results:
            return self._after_tool(request, results[-1])

        dynamic = request.system_dynamic
        if "The caller confirmed" in dynamic:
            return self._after_confirmation(dynamic)

        user_texts = [_text_of(m) for m in messages if m.get("role") == "user"]
        latest = _text_of(last)
        verified = "Caller verified: yes" in dynamic
        tools = {t["name"] for t in request.tools}

        if tools and (TRANSFER.search(latest.lower()) or LOOKUP.search(latest.lower())
                      or (not verified and parse_order_id(latest))):
            call = self._tool_for(user_texts, latest, verified)
            if call is not None:
                return [call, StreamEnd(stop_reason="tool_use")]
            return _say(ASK_DETAILS)
        return self._answer(latest, dynamic)

    def _tool_for(self, user_texts: list[str], latest: str, verified: bool) -> ToolCallDelta | None:
        # The MOST RECENT order number and verifying email: a caller who
        # corrects a misspoken digit must not be re-verified with the old one —
        # found by the simulator, where it turned one slip into a handoff.
        order = next((parse_order_id(t) for t in reversed(user_texts) if parse_order_id(t)), None)
        email = next((caller_email(t) for t in reversed(user_texts) if caller_email(t)), None)
        if not verified:
            if order and email:
                return ToolCallDelta("tu_verify", "verify_caller", {"order_id": order, "email": email})
            return None
        if TRANSFER.search(latest.lower()):
            recipient = _recipient_from_texts([latest], caller_email=email)
            quantity = parse_quantity(latest)
            if order and recipient and quantity:
                return ToolCallDelta("tu_transfer", "transfer_tickets",
                                     {"order_id": order, "quantity": quantity, "recipient_email": recipient})
            return None
        if order:
            return ToolCallDelta("tu_lookup", "lookup_order", {"order_id": order})
        return None

    def _after_tool(self, request: LLMRequest, block: dict[str, Any]) -> list[LLMEvent]:
        try:
            result = json.loads(block.get("content") or "{}")
        except json.JSONDecodeError:
            result = {}
        if "error" in result:
            if result["error"] == "unverified":
                return _say(ASK_DETAILS)
            return _say("I'm sorry, I couldn't complete that. Let me get a colleague to help.")
        if "verified" in result:
            if not result["verified"]:
                if result.get("failure_reason") == "order_not_found":
                    return _say("I couldn't find that order number. Could you read it to me again?")
                return _say("That email doesn't match the order. Could you spell the email for me?")
            # Verified: continue with what the caller originally wanted.
            conversation = " ".join(_text_of(m) for m in request.messages if m.get("role") == "user")
            order = result.get("order_id")
            if TRANSFER.search(conversation.lower()):
                recipient = _recipient(request.messages)
                quantity = parse_quantity(conversation)
                if recipient and quantity:
                    return [ToolCallDelta("tu_transfer", "transfer_tickets",
                                          {"order_id": order, "quantity": quantity,
                                           "recipient_email": recipient}),
                            StreamEnd(stop_reason="tool_use")]
            return [ToolCallDelta("tu_lookup", "lookup_order", {"order_id": order}),
                    StreamEnd(stop_reason="tool_use")]
        if "event_name" in result:
            status = result.get("event_status")
            if status == "cancelled":
                return _say(f"That event has been cancelled, so your order {result['order_id']} "
                            "will be refunded automatically to your original payment method.")
            if status == "postponed":
                return _say("That event has been postponed and your tickets stay valid for the new date.")
            seats = f"section {result['section']}" + (f", row {result['row']}" if result.get("row") else "")
            return _say(f"You have {result['quantity']} tickets for {result['event_name']} in {seats}, "
                        f"delivered by {result['delivery']}.")
        return _say("Done.")

    def _after_confirmation(self, dynamic: str) -> list[LLMEvent]:
        if '"status": "sent"' in dynamic or '"transfer_id"' in dynamic:
            return _say("Done, the tickets are on their way to your friend.")
        if '"request_id"' in dynamic:
            return _say("I've filed that refund request, and a colleague will review it.")
        return _say("I couldn't complete that one. Let me get a colleague to help.")

    def _answer(self, question: str, dynamic: str) -> list[LLMEvent]:
        passages = [(int(n), body) for n, _, body in _KNOWLEDGE.findall(dynamic)]
        if not passages:
            return _say(NO_ANSWER)
        query = _words(question)
        best: tuple[float, int, str] | None = None
        for tag, body in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")):
                words = _words(sentence)
                if not words:
                    continue
                # Cosine-style overlap, discounted by retrieval rank: [K1] is
                # the retriever's best guess and should win near-ties.
                overlap = len(query & words) / (len(query) ** 0.5 * len(words) ** 0.5 or 1)
                score = overlap / (1 + 0.25 * (tag - 1))
                if best is None or score > best[0]:
                    best = (score, tag, sentence.strip())
        if best is None or best[0] == 0:
            return _say(NO_ANSWER)
        _, tag, sentence = best
        return _say(f"{sentence} [K{tag}]")


def _recipient(messages: list[dict[str, Any]]) -> str | None:
    """The transfer recipient: an email the caller mentioned that is NOT the
    one they verified with (the one said alongside their order number)."""
    texts = [_text_of(m) for m in messages if m.get("role") == "user"]
    caller = next((caller_email(t) for t in reversed(texts) if caller_email(t)), None)
    return _recipient_from_texts(texts, caller_email=caller)


def _recipient_from_texts(texts: list[str], *, caller_email: str | None) -> str | None:
    for text in texts:
        for email in _all_emails(text):
            if email != caller_email:
                return email
    return None


def _all_emails(text: str) -> list[str]:
    return [email for _, email in sorted(_email_spans(text.lower()))]


def _say(text: str) -> list[LLMEvent]:
    return [TextDelta(text + " "), StreamEnd(stop_reason="end_turn")]
