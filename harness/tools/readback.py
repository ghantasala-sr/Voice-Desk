"""Read-back before commit. docs/07 §4.

Side-effecting tools with requires_confirmation are never executed from a
model decision alone. The read-back is RENDERED FROM THE TEMPLATE, using the
same validated arguments the executor will dispatch — never written by the
model.

That is the whole point. If the model wrote the confirmation sentence it could
misstate the quantity or the recipient, the caller would agree to the
misstatement, and the read-back would have laundered an error into consent.

Confirmation is classified deterministically. Anything that is not a clear yes
is not a yes: there is no path that executes a side effect on an ambiguous
answer.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Literal

from harness.tools.execution import Prepared

Confirmation = Literal["confirm", "deny", "unclear"]

_REASON_LABELS = {
    "cannot_attend": "not being able to attend",
    "event_cancelled": "a cancelled event",
    "event_postponed": "a postponed event",
    "duplicate_purchase": "a duplicate purchase",
    "delivery_failure": "tickets that were not delivered",
    "other": "the reason you described",
}

_AFFIRM = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|correct|right|that's right|that is right|"
    r"go ahead|please do|do it|confirm|confirmed|sounds good|perfect|absolutely|"
    r"yes please|please)\b",
    re.IGNORECASE,
)
_DENY = re.compile(
    r"\b(no|nope|wait|stop|hold on|don't|do not|cancel|not that|that's wrong|"
    r"incorrect|never mind|nevermind|actually|wrong)\b",
    re.IGNORECASE,
)


@dataclass
class PendingReadback:
    prepared: Prepared
    text: str
    turn_id: str
    tool_use_id: str
    asked: int = 1


def render(prepared: Prepared, *, event_name: str | None) -> str:
    """Fill the template from the validated arguments. Never from the model."""
    template = prepared.tool.readback_template or ""
    args = prepared.args
    values: dict[str, Any] = {
        **args,
        "s": "" if args.get("quantity") == 1 else "s",
        "event_name": event_name or "your event",
        "reason_label": _REASON_LABELS.get(str(args.get("reason_code")), "your request"),
    }
    fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    return template.format(**{f: values.get(f, "") for f in fields})


def classify(utterance: str) -> Confirmation:
    """Deterministic first, and conservative.

    A denial anywhere wins over an affirmation ("yes — wait, no"), and a
    correction ("no, three tickets") is a denial: the corrected request goes
    back through the model, which proposes a fresh call and gets a fresh
    read-back. An answer that is neither is `unclear`, never `confirm`.
    """
    text = utterance.strip().strip(".!?,").lower()
    if not text:
        return "unclear"
    if _DENY.search(text):
        return "deny"
    if _AFFIRM.match(text):
        return "confirm"
    return "unclear"
