"""CallState, TurnState, and the small value objects they hold.

See docs/06-harness.md §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Trigger = Literal["transcript", "barge_in", "silence", "tool_result"]
StopReason = Literal["answered", "handoff", "budget_exceeded", "interrupted", "error"]
ExecStatus = Literal["pending", "dispatched", "committed", "failed", "cancelled", "unknown"]
Source = Literal["browser", "phone", "replay", "simulation", "text"]


@dataclass
class ToolExecution:
    """One row of the idempotency ledger. Populated in week 2 (docs/07 §3)."""

    execution_id: str
    tool_name: str
    side_effecting: bool
    idempotency_key: str
    args_canonical: dict[str, Any]
    status: ExecStatus = "pending"
    attempt: int = 1
    response: dict[str, Any] | None = None
    error_type: str | None = None
    dispatched_at_ms: int | None = None
    resolved_at_ms: int | None = None


@dataclass
class CheckerVerdict:
    """One segment's grounding verdict. Populated in week 2 (docs/08)."""

    segment_index: int
    segment_text: str
    verdict: Literal[
        "pass", "no_checkable_content", "fail_ungrounded_entity", "fail_unsupported_claim"
    ]
    spans: list[dict[str, Any]] = field(default_factory=list)
    action_taken: Literal["released", "replaced", "handoff"] = "released"
    checker_ms: int = 0


@dataclass
class TurnState:
    turn_id: str
    seq: int
    trigger: Trigger

    user_text: str = ""
    stt_confidence: float | None = None
    stt_provider: str | None = None
    entity_confidences: dict[str, float] = field(default_factory=dict)

    # Retrieval. Four signals, not two — the fused margin turned out to be
    # inverted as a sufficiency signal, so the raw retriever scores are what
    # the handoff rule thresholds on. See
    # results/2026-10-08-retrieval-signal-selection/ and docs/05 §6.
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    #: Heading paths of the retrieved chunks, for the handoff summary.
    retrieved_headings: list[str] = field(default_factory=list)
    #: Chunks the model cited with [K] tags. Self-reported, so evaluation
    #: input only — the checker never trusts it.
    cited_chunk_ids: list[str] = field(default_factory=list)
    retrieval_top_score: float | None = None       # fused; ranking provenance
    retrieval_margin: float | None = None          # fused(1)-fused(2); NOT sufficiency
    retrieval_top_vec_score: float | None = None   # primary sufficiency signal
    retrieval_top_lex_score: float | None = None
    retrieval_vec_margin: float | None = None

    llm_iterations: int = 0

    # Three distinct texts: generated ⊇ approved ⊇ spoken.
    #
    #   generated_text  everything the model produced, including segments the
    #                   checker rejected. Needed for the catch-rate metric.
    #   approved_text   what passed the checker and was handed to TTS.
    #   spoken_text     what the caller actually heard, after barge-in.
    #
    # Conflating these is the classic bug: trim context to approved_text after
    # a barge-in and the model believes it said a sentence the caller never
    # heard, then refers back to it.
    generated_text: str = ""
    approved_text: str = ""
    spoken_text: str = ""

    checker_verdicts: list[CheckerVerdict] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    handoff_signals: list[str] = field(default_factory=list)
    handoff_score: float = 0.0

    stop_reason: StopReason | None = None
    latency_budget_ms: int = 2500
    budget_exceeded: bool = False
    ack_bypass_used: bool = False

    # Per-stage latency, keyed by the names in docs/01 §5.
    stages_ms: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None

    # Bookkeeping for the segments handed to TTS this turn.
    segments: list[str] = field(default_factory=list)
    #: How much of approved_text went into history next to a tool_use block.
    #: Needed if this turn is trimmed AFTER it completed (barge-in during
    #: playout — see Harness._barge_in_after_turn).
    history_offset: int = 0
    #: The turn ran a tool, read an action back, or resolved one. Its answer
    #: came from a tool, not the knowledge base — so weak retrieval on it is
    #: expected, and repeating it is not a rephrase.
    transactional: bool = False

    def stage(self, name: str) -> int | None:
        return self.stages_ms.get(name)


@dataclass
class CallState:
    call_id: str
    domain_config: str
    source: Source

    caller_verified: bool = False
    order_id: str | None = None

    turns: list[TurnState] = field(default_factory=list)

    # The LLM conversation, trimmed on every barge-in. Provider-neutral block
    # format — see harness/context.py.
    messages: list[dict[str, Any]] = field(default_factory=list)

    tool_executions: list[ToolExecution] = field(default_factory=list)
    handoff_signals: list[str] = field(default_factory=list)
    #: An action read back to the caller and awaiting their yes. Holds a
    #: harness.tools.readback.PendingReadback; typed loosely to avoid a cycle.
    pending_readback: Any = None
    rephrase_count: int = 0
    verification_failures: int = 0
    #: Most recent lookup_order result — supplies {event_name} to read-backs.
    last_order: dict[str, Any] | None = None
    #: Set when the call is handed off: decision, structured summary, text.
    handoff: dict[str, Any] | None = None
    handoff_reason: str | None = None
    outcome: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def next_seq(self) -> int:
        return len(self.turns)

    def last_user_text(self) -> str:
        for turn in reversed(self.turns):
            if turn.user_text:
                return turn.user_text
        return ""
