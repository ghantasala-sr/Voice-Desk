"""Fault profiles that make the mocks adversarial.

"Zero duplicate side effects" measured against a dictionary write you control
is near-tautological. These profiles are what make it a real claim
(docs/00 D9). The one that matters most is COMMIT-THEN-TIMEOUT: the backend
performs the write and then times out, so the harness genuinely does not know
whether the action happened and must reconcile instead of retrying blindly.

Every profile is seeded, so a failure reproduces from the seed in the report.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from harness.tools.mocks.ticketing import MockTicketing


class ToolTimeout(Exception):
    """The call did not answer in time. It MAY have committed."""


@dataclass(frozen=True)
class FaultProfile:
    name: str
    latency_ms: tuple[int, int] = (5, 15)
    timeout_rate: float = 0.0           # times out BEFORE doing anything
    garbage_rate: float = 0.0           # answers with schema-valid nonsense
    commit_then_timeout_rate: float = 0.0
    reconcile_down: bool = False        # the reconciliation read fails too
    seed: int = 0


PROFILES: dict[str, FaultProfile] = {
    "benign": FaultProfile("benign"),
    "slow": FaultProfile("slow", latency_ms=(800, 1200)),
    "flaky": FaultProfile("flaky", timeout_rate=0.3, garbage_rate=0.1, seed=7),
    "ambiguous": FaultProfile("ambiguous", commit_then_timeout_rate=1.0, seed=11),
    "blackout": FaultProfile("blackout", commit_then_timeout_rate=1.0, reconcile_down=True, seed=13),
}


#: Commit-then-timeout only means something for a WRITE. A read that "commits
#: and then times out" is just a timeout.
WRITE_TOOLS = frozenset({"transfer_tickets", "request_refund", "create_support_ticket"})


class FaultyBackend:
    """Wraps a MockTicketing and misbehaves according to a profile."""

    def __init__(
        self, backend: MockTicketing, profile: FaultProfile, *, writes: frozenset[str] = WRITE_TOOLS
    ) -> None:
        self.backend = backend
        self.profile = profile
        self._writes = writes
        self._rng = random.Random(profile.seed)

    async def call(self, tool: str, args: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        low, high = self.profile.latency_ms
        await asyncio.sleep(self._rng.uniform(low, high) / 1000)

        if self._rng.random() < self.profile.timeout_rate:
            raise ToolTimeout(f"{tool} timed out before committing")

        result = await self.backend.call(tool, args, idempotency_key=idempotency_key)

        if tool in self._writes and self._rng.random() < self.profile.commit_then_timeout_rate:
            # The write happened. The caller is about to be told otherwise.
            raise ToolTimeout(f"{tool} committed, then timed out")

        if self._rng.random() < self.profile.garbage_rate:
            return {"status": "ok", "data": None, "quantity": -1}
        return result

    async def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        if self.profile.reconcile_down:
            raise ToolTimeout("reconciliation endpoint unavailable")
        return await self.backend.get_by_idempotency_key(key)
