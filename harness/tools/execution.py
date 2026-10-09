"""The idempotency ledger and the tool executor. docs/07 §3.

The state machine, and the one rule that makes it correct:

    pending --dispatch--> dispatched --> committed
       |                       |-------> failed      (clean error upstream)
       |                       '-------> unknown     (timeout: may have committed)
       |                                    '--reconcile--> committed | failed
       '--cancel--> cancelled

    `cancelled` is reachable ONLY from `pending`.

Once a request has left the process it is resolved or reconciled, never
cancelled. Marking a dispatched call "cancelled" on a barge-in is exactly how
a duplicate side effect gets created, and it is the bug this module exists to
prevent.

Execution is two-phase so that read-back can sit between the phases:

    prepare()   validate, gate on verification, CLAIM the idempotency key
                as `pending`. Done when the read-back is spoken.
    dispatch()  `pending -> dispatched` is written BEFORE the outbound call,
                then the call. Done only after the caller says yes.

If the process dies mid-call the ledger says `dispatched`, and the startup
sweep reconciles it. A row still `pending` after a crash is provably safe to
cancel.

The enforcement point is the ledger, not this code. In Postgres it is a partial
unique index on idempotency_key WHERE side_effecting (001_init.sql): a
duplicate loses to the index deterministically, however it arrives.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Protocol

from core.clock import Clock, SystemClock
from core.ids import new_id
from harness.state import CallState, ToolExecution
from harness.tools.mocks.faults import ToolTimeout
from harness.tools.mocks.ticketing import ToolFailure
from harness.tools.registry import Registry, ToolSpec

#: Keys a successful response must carry. A response without them is
#: "schema-valid nonsense" — the garbage fault — and is never improvised into
#: an answer.
RESPONSE_KEYS: dict[str, set[str]] = {
    "verify_caller": {"verified"},
    "lookup_order": {"order_id", "event_name", "quantity"},
    "check_event_status": {"event_id", "status"},
    "transfer_tickets": {"transfer_id", "status"},
    "request_refund": {"request_id", "status"},
    "create_support_ticket": {"ticket_id", "status"},
}

RECONCILE_TIMEOUT_MS = 1500
IN_FLIGHT_POLL_MS = 50


def idempotency_key(call_id: str, tool: ToolSpec, args: dict[str, Any]) -> str:
    """Deterministic, call-scoped, over the action's IDENTITY fields only.

    * Deterministic: the same logical action collides rather than duplicates —
      whether the repeat comes from the model, a retry, or a re-attempt after
      an interruption.
    * Call-scoped: two callers transferring "the same two tickets" are two
      different actions.
    * Identity fields only: free text like reason_text is rephrased on a retry,
      and including it would let a rephrase slip past the key.
    """
    fields = tool.idempotency_fields or tuple(sorted(args))
    subset = {k: args.get(k) for k in sorted(fields)}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{call_id}|{tool.name}|{canonical}".encode()).hexdigest()


class ToolError(Exception):
    """Raised by prepare(): the call never reached the backend."""

    def __init__(self, error_type: str, detail: str) -> None:
        super().__init__(f"{error_type}: {detail}")
        self.error_type = error_type
        self.detail = detail


@dataclass
class Prepared:
    tool: ToolSpec
    execution: ToolExecution
    args: dict[str, Any]
    #: The action already committed earlier in this call. Do not dispatch;
    #: tell the caller it already went through.
    already_committed: bool = False


@dataclass(frozen=True)
class Outcome:
    execution: ToolExecution
    status: str
    response: dict[str, Any] | None
    error_type: str | None = None
    duplicate_suppressed: bool = False
    reconciled: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "committed"

    @property
    def needs_hard_handoff(self) -> bool:
        """Still unknown after reconciliation: we cannot tell the caller
        whether their action happened. Only a human should handle that."""
        return self.status == "unknown"


# ---------------------------------------------------------------------------
# Ledgers
# ---------------------------------------------------------------------------
class Ledger(Protocol):
    async def insert(self, call_id: str, turn_id: str | None, e: ToolExecution) -> bool:
        """Insert; return False if a side-effecting row with this key exists."""

    async def get_by_key(self, key: str) -> ToolExecution | None: ...
    async def update(self, e: ToolExecution) -> None: ...
    async def unresolved(self) -> list[ToolExecution]: ...


class InMemoryLedger:
    """Same uniqueness semantics as the Postgres index, for DB-free tests."""

    def __init__(self) -> None:
        self.rows: dict[str, ToolExecution] = {}          # execution_id -> row
        self._by_key: dict[str, str] = {}                 # key -> execution_id
        self._lock = asyncio.Lock()

    async def insert(self, call_id: str, turn_id: str | None, e: ToolExecution) -> bool:
        async with self._lock:
            if e.side_effecting and e.idempotency_key in self._by_key:
                return False
            self.rows[e.execution_id] = e
            if e.side_effecting:
                self._by_key[e.idempotency_key] = e.execution_id
            return True

    async def get_by_key(self, key: str) -> ToolExecution | None:
        execution_id = self._by_key.get(key)
        return self.rows.get(execution_id) if execution_id else None

    async def update(self, e: ToolExecution) -> None:
        self.rows[e.execution_id] = e

    async def unresolved(self) -> list[ToolExecution]:
        return [e for e in self.rows.values() if e.status in ("dispatched", "unknown")]

    def committed(self, tool: str | None = None) -> list[ToolExecution]:
        return [
            e for e in self.rows.values()
            if e.status == "committed" and (tool is None or e.tool_name == tool)
        ]


class PostgresLedger:
    """The real one: `tool_executions`, guarded by the partial unique index."""

    async def insert(self, call_id: str, turn_id: str | None, e: ToolExecution) -> bool:
        from core.db import fetch_one

        row = await fetch_one(
            """
            INSERT INTO tool_executions
              (execution_id, call_id, turn_id, tool_name, side_effecting,
               idempotency_key, args_canonical, status, attempt)
            VALUES (%(id)s, %(call)s, %(turn)s, %(tool)s, %(side)s,
                    %(key)s, %(args)s::jsonb, %(status)s, %(attempt)s)
            ON CONFLICT (idempotency_key) WHERE side_effecting DO NOTHING
            RETURNING execution_id
            """,
            {
                "id": e.execution_id, "call": call_id, "turn": turn_id, "tool": e.tool_name,
                "side": e.side_effecting, "key": e.idempotency_key,
                "args": json.dumps(e.args_canonical), "status": e.status, "attempt": e.attempt,
            },
        )
        return row is not None

    async def get_by_key(self, key: str) -> ToolExecution | None:
        from core.db import fetch_one

        row = await fetch_one(
            """
            SELECT execution_id, tool_name, side_effecting, idempotency_key, args_canonical,
                   status, attempt, response_payload, error_type
            FROM tool_executions WHERE idempotency_key = %(key)s AND side_effecting
            """,
            {"key": key},
        )
        return _row_to_execution(row) if row else None

    async def update(self, e: ToolExecution) -> None:
        from core.db import execute

        await execute(
            """
            UPDATE tool_executions
               SET status = %(status)s, attempt = %(attempt)s,
                   response_payload = %(response)s::jsonb, error_type = %(error)s,
                   dispatched_at = CASE WHEN %(status)s = 'dispatched' THEN now()
                                        ELSE dispatched_at END,
                   resolved_at = CASE WHEN %(status)s IN ('committed','failed','cancelled')
                                      THEN now() ELSE resolved_at END
             WHERE execution_id = %(id)s
            """,
            {
                "status": e.status, "attempt": e.attempt,
                "response": json.dumps(e.response) if e.response is not None else None,
                "error": e.error_type, "id": e.execution_id,
            },
        )

    async def unresolved(self) -> list[ToolExecution]:
        from core.db import fetch_all

        rows = await fetch_all(
            """
            SELECT execution_id, tool_name, side_effecting, idempotency_key, args_canonical,
                   status, attempt, response_payload, error_type
            FROM tool_executions WHERE status IN ('dispatched', 'unknown')
            """
        )
        return [_row_to_execution(r) for r in rows]


def _row_to_execution(row: dict[str, Any]) -> ToolExecution:
    return ToolExecution(
        execution_id=row["execution_id"], tool_name=row["tool_name"],
        side_effecting=row["side_effecting"], idempotency_key=row["idempotency_key"],
        args_canonical=row["args_canonical"], status=row["status"], attempt=row["attempt"],
        response=row["response_payload"], error_type=row["error_type"],
    )


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------
class Backend(Protocol):
    async def call(self, tool: str, args: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]: ...
    async def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None: ...


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: Registry,
        backend: Backend,
        ledger: Ledger,
        clock: Clock | None = None,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.ledger = ledger
        self.clock: Clock = clock or SystemClock()

    # -- phase 1 ------------------------------------------------------------
    async def prepare(
        self, call: CallState, turn_id: str | None, name: str, args: dict[str, Any]
    ) -> Prepared:
        tool = self.registry.get(name)
        if tool is None:
            raise ToolError("unknown_tool", name)
        errors = tool.validate(args)
        if errors:
            raise ToolError("invalid_arguments", "; ".join(errors))
        # Enforced HERE, not only in the prompt. A prompt-only disclosure rule
        # is not a control.
        if tool.requires_verification and not call.caller_verified:
            raise ToolError("unverified", f"{name} requires a verified caller")

        key = idempotency_key(call.call_id, tool, args)
        execution = ToolExecution(
            execution_id=new_id(),
            tool_name=name,
            side_effecting=tool.side_effecting,
            idempotency_key=key,
            args_canonical={k: args[k] for k in sorted(args)},
        )

        if await self.ledger.insert(call.call_id, turn_id, execution):
            call.tool_executions.append(execution)
            return Prepared(tool, execution, args)

        # Conflict: this exact action was seen before in this call.
        existing = await self.ledger.get_by_key(key)
        assert existing is not None, "insert conflicted but no row found"

        if existing.status == "committed":
            return Prepared(tool, existing, args, already_committed=True)

        if existing.status == "dispatched":
            existing = await self._await_in_flight(existing, tool)
            if existing.status == "committed":
                return Prepared(tool, existing, args, already_committed=True)

        if existing.status == "pending":
            return Prepared(tool, existing, args)

        # failed / unknown-reconciled-to-failed / cancelled: a fresh attempt.
        retry = replace(existing, status="pending", attempt=existing.attempt + 1,
                        response=None, error_type=None)
        await self.ledger.update(retry)
        _replace_in_call(call, retry)
        return Prepared(tool, retry, args)

    # -- phase 2 ------------------------------------------------------------
    async def dispatch(self, call: CallState, prepared: Prepared) -> Outcome:
        execution = prepared.execution
        if prepared.already_committed:
            return Outcome(execution, "committed", execution.response, duplicate_suppressed=True)
        if execution.status != "pending":
            raise ToolError("bad_state", f"cannot dispatch from {execution.status}")

        # Write the transition BEFORE the outbound call. If we die mid-call,
        # the ledger must say `dispatched` so the sweep can reconcile it.
        execution = replace(execution, status="dispatched", dispatched_at_ms=self.clock.now_ms())
        await self.ledger.update(execution)
        _replace_in_call(call, execution)

        try:
            response = await asyncio.wait_for(
                self.backend.call(prepared.tool.name, prepared.args,
                                  idempotency_key=execution.idempotency_key),
                timeout=prepared.tool.timeout_ms / 1000,
            )
        except ToolFailure as exc:
            return await self._resolve(call, execution, "failed", None, exc.error_type)
        except (ToolTimeout, TimeoutError):
            if not execution.side_effecting:
                return await self._resolve(call, execution, "failed", None, "timeout")
            unknown = replace(execution, status="unknown", error_type="timeout")
            await self.ledger.update(unknown)
            _replace_in_call(call, unknown)
            return await self.reconcile(call, unknown)

        if not RESPONSE_KEYS.get(execution.tool_name, set()) <= set(response) or (
            isinstance(response.get("quantity"), int) and response["quantity"] < 0
        ):
            if execution.side_effecting:
                # Nonsense back from a write: we do not know what happened.
                unknown = replace(execution, status="unknown", error_type="malformed")
                await self.ledger.update(unknown)
                _replace_in_call(call, unknown)
                return await self.reconcile(call, unknown)
            return await self._resolve(call, execution, "failed", None, "malformed")

        return await self._resolve(call, execution, "committed", response, None)

    async def run(self, call: CallState, turn_id: str | None, name: str, args: dict[str, Any]) -> Outcome:
        """prepare + dispatch, for tools that need no confirmation."""
        return await self.dispatch(call, await self.prepare(call, turn_id, name, args))

    async def cancel(self, call: CallState, prepared: Prepared) -> bool:
        """pending -> cancelled. Refuses anything that has left the process."""
        execution = prepared.execution
        if execution.status != "pending":
            return False
        cancelled = replace(execution, status="cancelled")
        await self.ledger.update(cancelled)
        _replace_in_call(call, cancelled)
        prepared.execution = cancelled
        return True

    # -- reconciliation -----------------------------------------------------
    async def reconcile(self, call: CallState, execution: ToolExecution) -> Outcome:
        """The interesting state. Ask the backend whether the action happened.

        found       -> it DID commit; say so truthfully
        not found   -> it did not; safe to retry or hand off
        read fails  -> stay unknown; HARD handoff. The agent tells the caller it
                       cannot confirm and gets a human. It does not guess and it
                       does not retry blindly.
        """
        try:
            found = await asyncio.wait_for(
                self.backend.get_by_idempotency_key(execution.idempotency_key),
                timeout=RECONCILE_TIMEOUT_MS / 1000,
            )
        except (ToolTimeout, TimeoutError, Exception):
            return Outcome(execution, "unknown", None, execution.error_type, reconciled=False)

        if found is not None:
            outcome = await self._resolve(call, execution, "committed", found, None)
        else:
            outcome = await self._resolve(call, execution, "failed", None, execution.error_type)
        return replace(outcome, reconciled=True)

    async def sweep(self, call: CallState) -> list[Outcome]:
        """Startup: reconcile anything a crash left dispatched or unknown."""
        return [await self.reconcile(call, e) for e in await self.ledger.unresolved()]

    # -- helpers ------------------------------------------------------------
    async def _resolve(
        self, call: CallState, execution: ToolExecution, status: str,
        response: dict[str, Any] | None, error_type: str | None,
    ) -> Outcome:
        resolved = replace(execution, status=status, response=response, error_type=error_type,
                           resolved_at_ms=self.clock.now_ms())
        await self.ledger.update(resolved)
        _replace_in_call(call, resolved)
        return Outcome(resolved, status, response, error_type)

    async def _await_in_flight(self, execution: ToolExecution, tool: ToolSpec) -> ToolExecution:
        """An identical call is already dispatched. Wait for it rather than
        sending a second one."""
        deadline = self.clock.now_ms() + tool.timeout_ms
        while self.clock.now_ms() < deadline:
            await asyncio.sleep(IN_FLIGHT_POLL_MS / 1000)
            current = await self.ledger.get_by_key(execution.idempotency_key)
            if current is not None and current.status != "dispatched":
                return current
        return replace(execution, status="unknown")


def _replace_in_call(call: CallState, execution: ToolExecution) -> None:
    for index, existing in enumerate(call.tool_executions):
        if existing.execution_id == execution.execution_id:
            call.tool_executions[index] = execution
            return
    call.tool_executions.append(execution)
