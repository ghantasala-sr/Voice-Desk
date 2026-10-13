"""Handoff: hard rules, weighted soft signals, and the summary. docs/09.

The asymmetry behind every choice here: a MISSED handoff is a confident wrong
answer to a customer; an UNNECESSARY one is a mildly annoyed customer and a
minute of a human's time. The operating point is tuned for recall.

Evaluation order, cheapest first:

  1. pre_gate()     hard utterance rules, BEFORE retrieval or any model call.
                    "Get me a person" must never cost an LLM round trip.
  2. state_rules()  hard rules on call state: verification failed twice, a
                    tool whose outcome is unknown, two checker failures...
  3. score()        weighted soft signals with per-turn decay.

Why weighted rather than "any signal fires": real calls routinely have one
mediocre retrieval or one low-confidence word, and escalating on each would
push containment toward zero. Two mediocre signals together — weak retrieval
AND a caller repeating themselves — is a genuinely bad turn.

The summary's structured fields come from CallState directly, so they cannot
be wrong. v1 builds no LLM narrative; `suggested_next_step` is a template per
reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from harness.policy import load_policy
from harness.state import CallState, TurnState

RuleType = Literal["hard", "soft"]

#: Hard state rules raise these signals somewhere in the harness; seeing one
#: means "hand off now", regardless of the soft score.
HARD_SIGNALS = {
    "verification_failed": "the caller could not be verified",
    "tool_state_unknown": "an action's outcome could not be confirmed",
    "retrieval_unavailable": "the knowledge base was unavailable",
    "readback_unclear": "the caller's confirmation stayed unclear",
    # A model stuck in a tool loop does not know what to do, which is the
    # definition of a handoff. Treated as hard, although docs/09 lists a
    # soft weight: the agent has already told the caller it is getting help.
    "iteration_cap_reached": "the agent could not resolve this within its step limit",
}

_NEXT_STEPS = {
    "explicit_human_request": "The caller asked for a person. Pick up from the questions below.",
    "payment_dispute": "Payment dispute or possible fraud: route to billing; do not discuss chargeback outcomes.",
    "distress": "Possible emergency or distress. Prioritise the caller's safety first.",
    "verification_failed": "Verify the caller's identity manually before discussing the order.",
    "tool_state_unknown": "Check the order system for the action below — it may or may not have gone through. Confirm with the caller.",
    "retrieval_unavailable": "The knowledge base was down; answer from policy directly.",
    "checker_failed_twice": "The agent twice drafted details it could not ground. Answer the question below from policy.",
    "readback_unclear": "The caller did not clearly confirm the pending action. It was NOT performed.",
    "iteration_cap_reached": "The agent could not resolve this in a few steps. Pick up from the questions below.",
    "soft": "The agent was not confident it could resolve this. Pick up from the questions below.",
}


@dataclass(frozen=True)
class HandoffDecision:
    should_handoff: bool
    rule_type: RuleType | None = None
    reason: str = ""
    signals: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


NO_HANDOFF = HandoffDecision(False)


class HandoffPolicy:
    def __init__(self, domain: str) -> None:
        self.config = load_policy(domain)
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (rule["id"], re.compile(pattern, re.IGNORECASE))
            for rule in self.config.get("hard_rules", [])
            if rule.get("match") == "utterance_pattern"
            for pattern in rule.get("patterns", [])
        ]
        soft = self.config.get("soft_signals", {})
        self.weights: dict[str, float] = {name: float(cfg.get("weight", 0)) for name, cfg in soft.items()}
        self.threshold = float(self.config.get("threshold", 0.6))
        self.decay = float(self.config.get("decay_per_turn", 0.5))
        self.retrieval_threshold = float(
            soft.get("retrieval_insufficient", {}).get("threshold_top_vec_score", 0.65)
        )
        # 0.0 disables the strict level when a domain does not configure it.
        self.very_weak_threshold = float(
            soft.get("retrieval_very_weak", {}).get("threshold_top_vec_score", 0.0)
        )
        self.entity_threshold = float(soft.get("entity_confidence_low", {}).get("threshold", 0.6))
        self.rephrase_similarity = float(soft.get("rephrase_repeat", {}).get("similarity", 0.82))

    # -- 1. hard utterance rules -------------------------------------------
    def pre_gate(self, utterance: str) -> HandoffDecision:
        for rule_id, pattern in self._patterns:
            if pattern.search(utterance):
                return HandoffDecision(
                    True, "hard", rule_id, [{"signal": rule_id, "match": pattern.pattern}], 1.0
                )
        return NO_HANDOFF

    # -- 2. hard state rules ------------------------------------------------
    def state_rules(self, call: CallState) -> HandoffDecision:
        raised = [s for turn in call.turns for s in turn.handoff_signals]
        for signal in HARD_SIGNALS:
            if signal in raised:
                return HandoffDecision(True, "hard", signal, [{"signal": signal}], 1.0)
        if raised.count("checker_failure") >= 2:
            return HandoffDecision(
                True, "hard", "checker_failed_twice", [{"signal": "checker_failure", "count": 2}], 1.0
            )
        return NO_HANDOFF

    # -- signals computed from turn state ----------------------------------
    def retrieval_signal(self, turn: TurnState, *, knowledge_available: bool) -> None:
        """Raw dense cosine, NOT the fused margin. Thresholds from
        results/2026-10-13-retrieval-calibration/ and -handoff-calibration/."""
        if not knowledge_available:
            return
        score = turn.retrieval_top_vec_score or 0.0
        if score < self.retrieval_threshold:
            _raise(turn, "retrieval_insufficient")
        # Two levels, because the calibration showed one cannot do both jobs:
        # a threshold loose enough to catch most gaps flags a third of
        # answerable questions, so "weak" only contributes, and only "very
        # weak" may hand off by itself.
        if score < self.very_weak_threshold:
            _raise(turn, "retrieval_very_weak")

    def entity_confidence_signal(self, turn: TurnState, word_confidences: list[tuple[str, float]]) -> None:
        """Low confidence on a KEY entity — a digit-bearing word, an email, an
        order id — not low average confidence. A mean hides the one word that
        matters (docs/09 §3)."""
        for word, confidence in word_confidences:
            if confidence < self.entity_threshold and (any(c.isdigit() for c in word) or "@" in word):
                turn.entity_confidences[word] = confidence
                _raise(turn, "entity_confidence_low")

    def budget_signal(self, call: CallState) -> None:
        recent = call.turns[-2:]
        if len(recent) == 2 and all(t.budget_exceeded for t in recent):
            _raise(call.turns[-1], "budget_exceeded_repeat")

    # -- 3. soft scoring ----------------------------------------------------
    def score(self, call: CallState) -> HandoffDecision:
        current = call.turns[-1].seq if call.turns else 0
        contributions: list[dict[str, Any]] = []
        for turn in call.turns:
            age = current - turn.seq
            for signal in dict.fromkeys(turn.handoff_signals):   # once per turn
                weight = self.weights.get(signal, 0.0)
                if weight:
                    contributions.append({
                        "signal": signal, "turn": turn.seq, "weight": weight,
                        "value": round(weight * (self.decay ** age), 3),
                    })
        total = round(sum(c["value"] for c in contributions), 3)
        if total < self.threshold:
            return HandoffDecision(False, None, "", contributions, total)
        dominant = max(contributions, key=lambda c: c["value"])["signal"]
        return HandoffDecision(True, "soft", dominant, contributions, total)


def _raise(turn: TurnState, signal: str) -> None:
    if signal not in turn.handoff_signals:
        turn.handoff_signals.append(signal)


# ---------------------------------------------------------------------------
# The summary — what the human actually receives
# ---------------------------------------------------------------------------
def build_summary(call: CallState, decision: HandoffDecision, *, policy_refs: list[str]) -> dict[str, Any]:
    committed = [e for e in call.tool_executions if e.status == "committed" and e.side_effecting]
    failed = [e for e in call.tool_executions if e.status in ("failed", "unknown")]
    refund_filed = any(e.tool_name == "request_refund" for e in committed)
    reason_key = decision.reason if decision.rule_type == "hard" else "soft"

    return {
        "call_id": call.call_id,
        "created_at": datetime.now(UTC).isoformat(),
        "caller_verified": call.caller_verified,
        "order_id": call.order_id,
        "asked": [t.user_text for t in call.turns if t.user_text],
        "attempted": [_describe(e) for e in call.tool_executions],
        "tools_committed": [{"tool": e.tool_name, "args": e.args_canonical, "result": e.response}
                            for e in committed],
        "tools_failed": [{"tool": e.tool_name, "status": e.status, "error": e.error_type}
                         for e in failed],
        "answered": [t.spoken_text for t in call.turns if t.spoken_text],
        "handoff_reason": decision.reason,
        "rule_type": decision.rule_type,
        "signals": decision.signals,
        "score": decision.score,
        "suggested_next_step": _NEXT_STEPS.get(reason_key, _NEXT_STEPS["soft"]),
        "money_statement": (
            "A refund REQUEST was filed for review. Nothing has been approved or promised."
            if refund_filed else
            "The agent has NOT promised any refund, credit or other money decision."
        ),
        "policy_refs": policy_refs,
        "turn_count": len(call.turns),
    }


def render_summary(summary: dict[str, Any]) -> str:
    """The view a human agent reads before they say hello."""
    lines = [
        f"HANDOFF — call {summary['call_id'][:10]}…  ·  {summary['turn_count']} turns",
        f"Reason: {summary['handoff_reason'].replace('_', ' ')} ({summary['rule_type']} rule)",
        "",
        f"CALLER    {'verified' if summary['caller_verified'] else 'NOT verified'}"
        f"   order {summary['order_id'] or '—'}",
        "",
        "ASKED",
        *(f"  {i}. {q!r}" for i, q in enumerate(summary["asked"], 1)),
    ]
    if summary["attempted"]:
        lines += ["", "AGENT DID", *(f"  {a}" for a in summary["attempted"])]
    if summary["answered"]:
        lines += ["", "AGENT SAID", *(f"  {a!r}" for a in summary["answered"])]
    if summary["policy_refs"]:
        lines += ["", "RELEVANT POLICY", *(f"  {r}" for r in summary["policy_refs"])]
    lines += [
        "", "SUGGESTED NEXT STEP", f"  {summary['suggested_next_step']}",
        f"  {summary['money_statement']}",
    ]
    return "\n".join(lines)


def _describe(execution: Any) -> str:
    mark = {"committed": "✓", "failed": "✗", "unknown": "?", "cancelled": "x"}.get(execution.status, "·")
    args = ", ".join(f"{k}={v}" for k, v in execution.args_canonical.items() if k != "summary")
    return f"{mark} {execution.tool_name}({args})  {execution.status}"
