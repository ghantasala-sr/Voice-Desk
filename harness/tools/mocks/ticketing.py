"""A mock ticketing backend with a small, deliberately awkward dataset.

The awkward orders are the point: a mixed-delivery order, a cancelled event, a
postponed event, an order already fully transferred, an email with a
plus-alias. A dataset of happy paths cannot exercise a support agent.

The mock does NOT deduplicate side effects. If the harness dispatches the same
transfer twice, two transfers happen here and `effects` shows both. That is
what makes "zero duplicate side effects" a claim about the harness rather than
about a forgiving backend.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

EVENTS: dict[str, dict[str, Any]] = {
    "EV-101": {"name": "the Saturday show at Harbour Pavilion", "venue": "Harbour Pavilion",
               "date": "2026-11-14", "status": "on_schedule"},
    "EV-102": {"name": "Lumen Hall late session", "venue": "Lumen Hall",
               "date": "2026-11-20", "status": "on_schedule"},
    "EV-103": {"name": "the Kestrel Amphitheatre summer opener", "venue": "Kestrel Amphitheatre",
               "date": "2026-10-30", "status": "cancelled"},
    "EV-104": {"name": "Northgate Arena night two", "venue": "Northgate Arena",
               "date": "2026-12-05", "status": "postponed", "new_date": "2027-01-16"},
    "EV-105": {"name": "the Foundry club night", "venue": "The Foundry",
               "date": "2026-11-08", "status": "on_schedule"},
}

ORDERS: dict[str, dict[str, Any]] = {
    "TK-88213": {"email": "sam@example.com", "event_id": "EV-101", "quantity": 4,
                 "section": "112", "row": "F", "delivery": "mobile", "transferred": 0},
    "TK-40417": {"email": "priya.n@example.com", "event_id": "EV-102", "quantity": 2,
                 "section": "GA", "row": None, "delivery": "willcall", "transferred": 0},
    "TK-51900": {"email": "lee+tickets@example.com", "event_id": "EV-103", "quantity": 2,
                 "section": "Lawn", "row": None, "delivery": "mobile", "transferred": 0},
    "TK-62044": {"email": "dana@example.com", "event_id": "EV-104", "quantity": 3,
                 "section": "204", "row": "C", "delivery": "mobile", "transferred": 0},
    "TK-73355": {"email": "omar@example.com", "event_id": "EV-101", "quantity": 2,
                 "section": "108", "row": "A", "delivery": "mobile", "transferred": 2},
    "TK-19876": {"email": "jo@example.com", "event_id": "EV-105", "quantity": 1,
                 "section": "GA", "row": None, "delivery": "print", "transferred": 0},
    "TK-24680": {"email": "mixed@example.com", "event_id": "EV-102", "quantity": 4,
                 "section": "Balcony", "row": "B", "delivery": "mixed", "transferred": 0},
}


class ToolFailure(Exception):
    """A clean, typed error from the backend: the action did NOT happen."""

    def __init__(self, error_type: str, detail: str) -> None:
        super().__init__(f"{error_type}: {detail}")
        self.error_type = error_type
        self.detail = detail


@dataclass
class Effect:
    tool: str
    idempotency_key: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class MockTicketing:
    orders: dict[str, dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(ORDERS))
    events: dict[str, dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(EVENTS))
    #: Every side effect that actually happened, in order. Never deduplicated.
    effects: list[Effect] = field(default_factory=list)
    _ticket_seq: int = 1000

    # -- dispatch -----------------------------------------------------------
    async def call(self, tool: str, args: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        handler = getattr(self, f"_{tool}", None)
        if handler is None:
            raise ToolFailure("unknown_tool", tool)
        return handler(args, idempotency_key)

    async def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        """Reconciliation read: did an action with this key actually happen?

        The companion every side-effecting endpoint must expose, because after
        a timeout nothing else can answer that question (docs/07 §3.4).
        """
        for effect in self.effects:
            if effect.idempotency_key == key:
                return effect.result
        return None

    def count_effects(self, tool: str | None = None) -> int:
        return sum(1 for e in self.effects if tool is None or e.tool == tool)

    # -- read-only ----------------------------------------------------------
    def _verify_caller(self, args: dict[str, Any], _: str) -> dict[str, Any]:
        order = self.orders.get(args["order_id"])
        if order is None:
            return {"verified": False, "failure_reason": "order_not_found"}
        if order["email"].lower() != str(args["email"]).lower():
            return {"verified": False, "failure_reason": "email_mismatch"}
        return {"verified": True, "order_id": args["order_id"]}

    def _lookup_order(self, args: dict[str, Any], _: str) -> dict[str, Any]:
        order = self.orders.get(args["order_id"])
        if order is None:
            raise ToolFailure("not_found", f"no order {args['order_id']}")
        event = self.events[order["event_id"]]
        return {
            "order_id": args["order_id"],
            "event_id": order["event_id"],
            "event_name": event["name"],
            "event_date": event["date"],
            "event_status": event["status"],
            "quantity": order["quantity"],
            "transferable": order["quantity"] - order["transferred"],
            "section": order["section"],
            "row": order["row"],
            "delivery": order["delivery"],
        }

    def _check_event_status(self, args: dict[str, Any], _: str) -> dict[str, Any]:
        event = self.events.get(args["event_id"])
        if event is None:
            raise ToolFailure("not_found", f"no event {args['event_id']}")
        return {"event_id": args["event_id"], **event}

    # -- side-effecting -----------------------------------------------------
    def _transfer_tickets(self, args: dict[str, Any], key: str) -> dict[str, Any]:
        order = self.orders.get(args["order_id"])
        if order is None:
            raise ToolFailure("not_found", f"no order {args['order_id']}")
        if order["delivery"] == "willcall":
            raise ToolFailure("not_transferable", "will-call orders cannot be transferred")
        available = order["quantity"] - order["transferred"]
        if args["quantity"] > available:
            raise ToolFailure("insufficient_tickets", f"only {available} transferable")
        order["transferred"] += args["quantity"]
        result = {"transfer_id": f"TR-{len(self.effects) + 1:05d}", "status": "sent",
                  "quantity": args["quantity"], "recipient_email": args["recipient_email"]}
        self.effects.append(Effect("transfer_tickets", key, dict(args), result))
        return result

    def _request_refund(self, args: dict[str, Any], key: str) -> dict[str, Any]:
        if args["order_id"] not in self.orders:
            raise ToolFailure("not_found", f"no order {args['order_id']}")
        result = {"request_id": f"RR-{len(self.effects) + 1:05d}", "status": "queued_for_review"}
        self.effects.append(Effect("request_refund", key, dict(args), result))
        return result

    def _create_support_ticket(self, args: dict[str, Any], key: str) -> dict[str, Any]:
        self._ticket_seq += 1
        result = {"ticket_id": f"ST-{self._ticket_seq}", "status": "open"}
        self.effects.append(Effect("create_support_ticket", key, dict(args), result))
        return result
