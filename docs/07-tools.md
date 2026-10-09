# 07 — Tools, idempotency, and read-back

The claim this subsystem exists to support is **zero duplicate side effects under
interruption**. That claim is only interesting if the system is actually hard to satisfy,
which is why the mocks are adversarial ([00-decisions.md](00-decisions.md) D9) and why the
enforcement point is a database constraint rather than application logic.

---

## 1. `tools.yaml`

```yaml
# domains/ticketing/tools.yaml
version: 1
tools:
  - name: verify_caller
    description: >
      Verify the caller's identity against an order number and the email on the
      order. Must succeed before any order details are disclosed.
    side_effecting: false
    timeout_ms: 2000
    endpoint: mock://ticketing.verify_caller
    parameters:
      type: object
      required: [order_id, email]
      properties:
        order_id: {type: string, pattern: "^TK-[0-9]{5}$"}
        email:    {type: string, format: email}
    returns:
      verified: boolean
      failure_reason: string

  - name: lookup_order
    description: Look up an order's events, seats, and delivery status.
    side_effecting: false
    requires_verification: true
    timeout_ms: 2500
    endpoint: mock://ticketing.lookup_order
    parameters:
      type: object
      required: [order_id]
      properties:
        order_id: {type: string, pattern: "^TK-[0-9]{5}$"}

  - name: check_event_status
    description: Check whether an event is on schedule, postponed, or cancelled.
    side_effecting: false
    requires_verification: false
    timeout_ms: 2000
    endpoint: mock://ticketing.check_event_status
    parameters:
      type: object
      required: [event_id]
      properties:
        event_id: {type: string}

  - name: transfer_tickets
    description: Transfer tickets from this order to another person's email.
    side_effecting: true
    requires_verification: true
    requires_confirmation: true
    timeout_ms: 4000
    endpoint: mock://ticketing.transfer_tickets
    idempotency_fields: [order_id, quantity, recipient_email]
    readback_template: >
      Transferring {quantity} ticket{s} for {event_name} to {recipient_email}.
      Should I go ahead?
    parameters:
      type: object
      required: [order_id, quantity, recipient_email]
      properties:
        order_id:        {type: string, pattern: "^TK-[0-9]{5}$"}
        quantity:        {type: integer, minimum: 1, maximum: 10}
        recipient_email: {type: string, format: email}

  - name: request_refund
    description: >
      File a refund request for review. This does NOT approve a refund and does
      not determine an amount. A human reviews every request.
    side_effecting: true
    requires_verification: true
    requires_confirmation: true
    timeout_ms: 4000
    endpoint: mock://ticketing.request_refund
    idempotency_fields: [order_id, reason_code]
    readback_template: >
      I'll file a refund request for order {order_id}, reason: {reason_text}.
      A colleague reviews it and contacts you. Shall I file it?
    parameters:
      type: object
      required: [order_id, reason_code]
      properties:
        order_id:    {type: string, pattern: "^TK-[0-9]{5}$"}
        reason_code: {type: string, enum: [cannot_attend, event_cancelled,
                                           event_postponed, duplicate_purchase,
                                           delivery_failure, other]}
        reason_text: {type: string, maxLength: 200}
    # NOTE: deliberately has no `amount` field. The model cannot name a figure
    # because the schema gives it nowhere to put one. See docs/06 §3.

  - name: create_support_ticket
    description: Open a support ticket for a human agent to follow up.
    side_effecting: true
    requires_verification: false
    requires_confirmation: false      # harmless, and used on the handoff path
    timeout_ms: 3000
    endpoint: mock://ticketing.create_support_ticket
    idempotency_fields: [call_id, category]
    parameters:
      type: object
      required: [category, summary]
      properties:
        category: {type: string, enum: [billing, access, delivery, accessibility, other]}
        summary:  {type: string, maxLength: 1000}
```

### Registry responsibilities

`harness/tools/registry.py` loads this file and, at startup:

- validates every `parameters` block is valid JSON Schema;
- asserts every `side_effecting: true` tool declares `idempotency_fields`;
- asserts every `requires_confirmation: true` tool declares a `readback_template`, and that
  every `{placeholder}` in that template resolves from the parameter set or from call state;
- compiles the provider-neutral schema list passed in `LLMRequest.tools`.

These are startup assertions, not runtime checks. A malformed tool definition should fail
`make test`, not a live call.

---

## 2. Argument validation before execution

The model's arguments are validated against the schema **before** anything is dispatched.
A failure is not an error to the caller; it is a corrective note back to the model
(one retry, per [06-harness.md](06-harness.md) §6).

Additionally, `requires_verification: true` tools are blocked by the harness when
`call.caller_verified` is false. This is enforced in `execution.py`, not just in the
prompt — a prompt-only data-disclosure rule is not a control.

---

## 3. The idempotency protocol

### 3.1 The key

```python
def idempotency_key(call_id: str, tool: ToolSpec, args: dict) -> str:
    subset = {k: args[k] for k in sorted(tool.idempotency_fields)}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{call_id}|{tool.name}|{canonical}".encode()).hexdigest()
```

Three properties, each deliberate:

- **Deterministic.** The same logical action produces the same key, so a retry — whether
  from the model repeating itself, a network retry, or a post-interruption re-attempt —
  collides rather than duplicating.
- **Scoped to the call.** Two different callers transferring the same two tickets are
  different actions. Keying globally would wrongly suppress the second.
- **Subset, not all args.** `reason_text` is free prose the model may phrase differently on
  a retry; including it would let a rephrase slip past the key. Only the fields that define
  the action's *identity* participate.

> Consequence worth stating plainly: a caller who genuinely wants to transfer two more
> tickets to the same person in the same call produces the same key and is blocked. The
> harness detects this (`committed` row with identical key), and instead of silently
> suppressing, it tells the caller the transfer already went through and offers a handoff.
> Silently doing nothing would be worse than either alternative.

### 3.2 States

```
pending ──dispatch──► dispatched ──► committed
   │                       │
   │                       ├──► failed      (clean error from upstream)
   │                       └──► unknown     (timeout — may or may not have committed)
   │                                 │
   └──cancel──► cancelled            └──reconcile──► committed | failed
```

**The one rule that makes this correct:** `cancelled` is reachable only from `pending`.
Once a request has left the process, it is never cancelled — it is resolved or reconciled.
Marking a dispatched call "cancelled" on barge-in is exactly how duplicate side effects get
created, and it is the bug this state machine exists to prevent.

### 3.3 Execution

```python
async def execute(call, turn, tool: ToolSpec, args: dict) -> ToolExecution:
```

1. Validate args; check verification gate.
2. Compute key. `INSERT ... ON CONFLICT (idempotency_key) DO NOTHING` with status
   `pending`.
3. **If the insert conflicted**, read the existing row:
   - `committed` → return the stored `response_payload`. **Do not re-dispatch.** This is
     the duplicate-suppression path, and it returns the original result so the conversation
     stays coherent.
   - `dispatched` → an identical call is in flight. Poll to `VD_TOOL_TIMEOUT_MS`, then
     treat as `unknown`.
   - `failed` / `unknown` → allowed to retry: increment `attempt`, set `pending`, continue.
   - `cancelled` → allowed to retry as a fresh attempt.
4. Transition `pending → dispatched`, stamp `dispatched_at`. **Commit this transition to
   Postgres before the outbound call.** If the process dies mid-call, the ledger says
   `dispatched`, and startup reconciliation finds it. A row still saying `pending` after a
   crash is provably safe to cancel.
5. Dispatch with timeout.
6. On success → `committed` with payload. On typed error → `failed`. On timeout →
   `unknown`.

### 3.4 Reconciliation

`unknown` is the interesting state and the reason this design is worth describing in an
interview.

Every side-effecting mock exposes a companion read operation keyed by idempotency key:

```python
async def get_by_idempotency_key(key: str) -> dict | None
```

On `unknown`, the harness queries it (with its own short timeout):

- found → the action *did* commit; update to `committed` with the real payload, emit
  `tool_reconciled`, and the conversation continues truthfully.
- not found → `failed`; safe to retry or hand off.
- reconciliation itself fails → stay `unknown`, raise a **hard** handoff signal. The agent
  tells the caller it cannot confirm whether the action completed and gets a human. It does
  not guess, and it does not retry blindly.

Startup reconciliation sweeps any `dispatched` or `unknown` rows left by a crash.

This path is what the fault-injection suite's commit-then-timeout profile exercises
([12-eval.md](12-eval.md) §7).

---

## 4. Read-back protocol

Side-effecting tools with `requires_confirmation: true` are never executed from a model
decision alone.

```
model emits transfer_tickets(...)
   │
   ├─ harness does NOT execute
   ├─ render readback_template with validated args        ← deterministic, not the LLM
   ├─ Speak(readback)                                     ← bypasses checker: template text
   ├─ set call.pending_readback = {tool, args, key, spoken_at}
   └─ end the LLM iteration; wait for the caller
```

**The read-back text is rendered from the template, not generated.** This is the whole
point. If the model wrote the confirmation sentence it could misstate the quantity or the
recipient, the caller would confirm the misstatement, and the read-back would have
laundered an error into consent. Rendering from the same validated arguments the executor
will use means the caller confirms exactly what will happen.

### Confirmation detection

`harness/tools/readback.py`, deterministic first:

```python
AFFIRM = {"yes","yeah","yep","yup","sure","ok","okay","correct","that's right",
          "go ahead","please do","do it","confirm","sounds good","right"}
DENY   = {"no","nope","wait","stop","hold on","don't","cancel","not that",
          "that's wrong","incorrect","never mind","actually"}
```

1. Normalize the transcript; if it matches an affirm phrase with nothing contradictory,
   confirm. If it matches a deny phrase, reject.
2. If it is neither (a correction like "no, three tickets", or a question), fall through to
   **one** cheap LLM classification returning `confirm | deny | unclear | correction`. A
   `correction` returns to the loop with the corrected arguments and a fresh read-back.
3. `unclear` → re-ask once, verbatim. Second `unclear` → handoff. The system does not
   execute a side effect on an ambiguous yes.
4. **Interruption during read-back is a deny** ([06-harness.md](06-harness.md) §5 step 5).
5. Silence past `VD_READBACK_TIMEOUT_MS` (6 s) → re-ask once, then handoff.

Events `readback_spoken`, `readback_confirmed`, `readback_rejected` are logged, which is how
the fault-injection suite asserts the mid-read-back interruption case produced zero commits.

---

## 5. Filler speech

A dispatched tool that has not resolved within `VD_TOOL_FILLER_MS` (700 ms) triggers one
filler line from a fixed list ("Still pulling that up…"). Fixed template, no checkable
content, bypasses the checker. At most one filler per tool call and at most two per turn;
beyond that the silence is a symptom and the budget rule should fire instead.

---

## 6. Mocks

`harness/tools/mocks/ticketing.py` holds a small seeded dataset — a dozen orders across
five events, including deliberately awkward ones: an order with mixed delivery methods, a
cancelled event, a postponed event, an order already fully transferred, and an order whose
email has a plus-alias.

`harness/tools/mocks/faults.py` wraps any mock with a profile:

```python
@dataclass
class FaultProfile:
    name: str
    latency_ms: tuple[int, int] = (20, 60)
    timeout_rate: float = 0.0
    garbage_rate: float = 0.0          # schema-valid JSON, nonsense values
    duplicate_delivery_rate: float = 0.0
    commit_then_timeout_rate: float = 0.0   # commits upstream, then times out
    seed: int = 0

PROFILES = {
    "benign":    FaultProfile("benign"),
    "slow":      FaultProfile("slow", latency_ms=(800, 1500)),
    "flaky":     FaultProfile("flaky", timeout_rate=0.3, garbage_rate=0.1),
    "ambiguous": FaultProfile("ambiguous", commit_then_timeout_rate=0.5,
                              duplicate_delivery_rate=0.3),
}
```

`commit_then_timeout` is the profile that matters. The mock performs the write, records it
under the idempotency key, and *then* raises a timeout — so the harness genuinely does not
know, and must reconcile. Without this profile the "zero duplicates" result is not worth
reporting.

All profiles are seeded, so a fault-injection failure is reproducible from the seed printed
in the test output.
