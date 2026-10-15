"""The turn loop.

    trigger -> work -> check -> memory -> stop

One turn, in order (docs/01 §3):

  0. assemble TurnState, stamp t0, open a span
  1. PRE-GATE     hard handoff rules on the raw transcript. No model call.
  1b. READ-BACK   if an action is awaiting confirmation, resolve it first.
  2. SIGNALS      rephrase, low-confidence entities.
  3. RETRIEVE     hybrid search.
  4. DECIDE       hard state rules + soft score. Hand off here if warranted —
                  still before any model call.
  5. ITERATE      stream the LLM; segment; check; speak; run tools; repeat up
                  to max_tool_iterations LLM calls.
  6. STOP         exactly one stop_reason. Decide again on what was learned.
  7. PERSIST      emit the finished turn for the runner to store.

The invariant that outranks everything else: a failure becomes a handoff
signal, never an improvised answer. There is no path in this file where an
exception handler produces caller-facing text that did not come from a fixed
template.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from core.clock import Clock, Stopwatch, SystemClock
from core.ids import new_id
from core.settings import Settings
from core.tracing import current_trace_id, span
from harness import context as ctx
from harness import prompts
from harness.checker import Checker, GroundingSources, Lexicon, load_lexicon
from harness.events import (
    BargeIn,
    CancelSpeech,
    Emit,
    EndCall,
    EventStream,
    InEvent,
    OutCommand,
    Shutdown,
    SilenceTimeout,
    Speak,
    TranscriptIn,
    TurnCompleted,
)
from harness.handoff import HandoffDecision, HandoffPolicy, build_summary, render_summary
from harness.normalize import strip_citations, to_spoken
from harness.policy import claim_overlap_min, is_social, retrieval_sufficiency_threshold
from harness.retriever import NullRetriever, Retriever, empty_result
from harness.segmenter import Segmenter
from harness.state import CallState, CheckerVerdict, TurnState
from harness.tools.execution import Outcome, ToolError, ToolExecutor
from harness.tools.readback import PendingReadback, classify, render
from harness.tools.registry import Registry
from providers.llm.base import (
    LLMProvider,
    LLMRequest,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)
from retrieval.search import RetrievalResult

MAX_SILENCE_REPROMPTS = 1
MAX_READBACK_ASKS = 2

# Do not spend a retry unless there is enough budget left for it to plausibly
# help. A second slow failure is worse for the caller than an honest handoff.
LLM_RETRY_MIN_BUDGET_MS = 1200


class Harness:
    """Owns the conversation. Knows nothing about audio."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings,
        domain: str | None = None,
        clock: Clock | None = None,
        retriever: Retriever | None = None,
        executor: ToolExecutor | None = None,
        similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.domain = domain or settings.domain
        self.clock: Clock = clock or SystemClock()
        # Ports, not implementations: the harness stays testable without
        # Postgres, without an embedding model, and without a real backend.
        self.retriever: Retriever = retriever or NullRetriever()
        self.executor = executor
        self.registry = executor.registry if executor else Registry.empty()
        #: Embedding similarity for the rephrase signal. Without it that
        #: signal is off — word overlap cannot see a rephrase, by definition.
        self.similarity = similarity
        self.policy = HandoffPolicy(self.domain)
        self._silence_reprompts = 0
        self._ack_index = 0
        #: Tool calls abandoned by a barge-in, still resolving their ledger
        #: rows. Held so they are not garbage-collected mid-flight.
        self._background: set[asyncio.Task[Any]] = set()

    # -----------------------------------------------------------------------
    # The run loop
    # -----------------------------------------------------------------------
    async def run(self, call: CallState, stream: EventStream) -> AsyncIterator[OutCommand]:
        """Drive a call until Shutdown. Yields commands for the runner."""
        with span("call", call_id=call.call_id, domain=call.domain_config, source=call.source):
            yield Emit("call_started", {"domain": call.domain_config, "source": call.source})

            while True:
                event = await stream.get()

                if isinstance(event, Shutdown):
                    # A call that answered at least once and was never handed
                    # off is contained — the business metric's numerator.
                    # "dropped" is reserved for calls that never completed a
                    # turn.
                    answered = any(t.stop_reason == "answered" for t in call.turns)
                    call.outcome = call.outcome or ("contained" if answered else "dropped")
                    yield Emit("call_ended", {"outcome": call.outcome, "reason": event.reason})
                    yield EndCall(event.reason)
                    return

                if isinstance(event, TranscriptIn):
                    if not event.text.strip():
                        continue
                    self._silence_reprompts = 0
                    async for command in self._run_turn(call, event, stream):
                        yield command
                    if call.outcome in ("handoff", "error"):
                        yield Emit("call_ended", {"outcome": call.outcome})
                        yield EndCall(call.outcome)
                        return

                elif isinstance(event, SilenceTimeout):
                    async for command in self._handle_silence(call):
                        yield command
                    if call.outcome == "dropped":
                        yield EndCall("silence")
                        return

                elif isinstance(event, BargeIn):
                    async for command in self._barge_in_after_turn(call, event):
                        yield command

    # -----------------------------------------------------------------------
    # One turn
    # -----------------------------------------------------------------------
    async def _run_turn(
        self, call: CallState, event: TranscriptIn, stream: EventStream
    ) -> AsyncIterator[OutCommand]:
        turn = TurnState(
            turn_id=new_id(),
            seq=call.next_seq,
            trigger="transcript",
            user_text=event.text,
            stt_confidence=event.confidence,
            stt_provider=event.provider or None,
            latency_budget_ms=self.settings.turn_budget_ms,
        )
        call.turns.append(turn)
        ctx.append_user(call.messages, event.text)

        watcher = _BargeInWatcher(stream)
        watcher.start()
        stopwatch = Stopwatch(self.clock)
        state = _GenerationState()
        run = _TurnRun(knowledge=empty_result(event.text))
        handing_off = self.settings.handoff_enabled

        try:
            with span(
                "turn", call_id=call.call_id, turn_id=turn.turn_id, trigger=turn.trigger
            ) as turn_span:
                turn.trace_id = current_trace_id()
                yield Emit("turn_started", {"user_text": event.text}, turn.turn_id)
                # The transcript the harness acted on — what the gap report
                # and the STT-confidence analysis read. Partials never reach
                # the harness, so there is no stt_partial event (docs/03).
                yield Emit("stt_final", {"text": event.text, "confidence": event.confidence,
                                         "provider": event.provider}, turn.turn_id)

                turn.social = is_social(event.text)

                # --- 1. pre-gate: hard utterance rules, no model call -------
                decision = self.policy.pre_gate(event.text) if handing_off else None
                if decision and decision.should_handoff:
                    async for command in self._hand_off(call, turn, decision, run, state, stopwatch):
                        yield command
                    run.done = True

                # --- 1b. an action is waiting for the caller's yes ----------
                if not run.done and call.pending_readback is not None:
                    async for command in self._resolve_readback(call, turn, run, state, stopwatch):
                        yield command

                # --- 2. signals from the utterance itself ------------------
                if not run.done and handing_off:
                    self._rephrase_signal(call, turn)
                    self.policy.entity_confidence_signal(turn, event.word_confidences)

                # --- 3. retrieve -------------------------------------------
                if not run.done:
                    with stopwatch.stage("retrieval"):
                        run.knowledge = await self._retrieve(call, turn)
                    yield Emit(
                        "retrieval_completed",
                        {
                            "query": run.knowledge.query_text,
                            "chunk_ids": turn.retrieved_chunk_ids,
                            "top_score": turn.retrieval_top_score,
                            "top_vec_score": turn.retrieval_top_vec_score,
                            "elapsed_ms": run.knowledge.elapsed_ms,
                        },
                        turn.turn_id,
                    )

                # --- 4. decide, still before any model call ----------------
                if not run.done and handing_off:
                    decision = self._decide(call)
                    if decision.should_handoff:
                        async for command in self._hand_off(call, turn, decision, run, state, stopwatch):
                            yield command
                        run.done = True

                # --- 5. iterate --------------------------------------------
                if not run.done:
                    async for command in self._generate(call, turn, stopwatch, watcher, run, state):
                        yield command

                # --- 6. stop -----------------------------------------------
                if watcher.triggered:
                    turn_span.mark_cancelled()
                    async for command in self._finish_interrupted(call, turn, watcher, run):
                        yield command
                elif turn.stop_reason is None:
                    self._finish_normally(call, turn, run, state)
                    if handing_off and turn.stop_reason != "error":
                        # Weak retrieval only means something on a turn that
                        # answered FROM KNOWLEDGE. "yes" or "send two tickets"
                        # retrieves weakly by nature — its answer comes from a
                        # tool — and counting those would, with decay, hand
                        # off in the middle of a transfer that was going fine.
                        turn.transactional = bool(
                            run.tool_texts or run.readback_started or run.action_note
                        )
                        # Nor on a greeting: "Hi there." has nothing to
                        # retrieve, and must not hand the caller to a human.
                        if not turn.transactional and not turn.social:
                            self.policy.retrieval_signal(
                                turn,
                                knowledge_available=not isinstance(self.retriever, NullRetriever),
                            )
                        self.policy.budget_signal(call)
                        decision = self._decide(call)
                        if decision.should_handoff:
                            async for command in self._hand_off(
                                call, turn, decision, run, state, stopwatch
                            ):
                                yield command

                self._record_stages(turn, event, stopwatch)
                turn_span.set(
                    stop_reason=turn.stop_reason,
                    llm_iterations=turn.llm_iterations,
                    handoff_signals=list(turn.handoff_signals),
                )
        finally:
            # Always stop the watcher and return the events it absorbed, even
            # if the body raised.
            stream.pushback(await watcher.stop())

        # --- 7. persist ---------------------------------------------------
        assert turn.stop_reason is not None, "every turn must end with exactly one stop_reason"
        yield Emit(
            "turn_completed",
            {
                "stop_reason": turn.stop_reason,
                "spoken_chars": len(turn.spoken_text),
                "signals": list(turn.handoff_signals),
                "stages_ms": turn.stages_ms,
            },
            turn.turn_id,
        )
        yield TurnCompleted(turn)

    async def _retrieve(self, call: CallState, turn: TurnState) -> RetrievalResult:
        """Hybrid search, scoped to this call's domain.

        The previous user utterance is passed so a follow-up ("what about
        parking?") inherits its subject (docs/05 §5). A retrieval FAILURE is a
        hard handoff signal: answering without grounding is the one thing the
        system must not do.
        """
        previous = next(
            (t.user_text for t in reversed(call.turns[:-1]) if t.user_text), None
        )
        with span("retrieval", call_id=call.call_id, turn_id=turn.turn_id) as retrieval_span:
            try:
                result = await self.retriever.fetch(turn.user_text, previous=previous)
            except Exception as exc:
                turn.handoff_signals.append("retrieval_unavailable")
                retrieval_span.set(error=f"{type(exc).__name__}: {exc}")
                return empty_result(turn.user_text)

            turn.retrieved_chunk_ids = result.chunk_ids()
            turn.retrieved_headings = [c.heading_path for c in result.chunks]
            turn.retrieval_top_score = result.top_score
            turn.retrieval_margin = result.margin
            turn.retrieval_top_vec_score = result.top_vec_score
            turn.retrieval_top_lex_score = result.top_lex_score
            turn.retrieval_vec_margin = result.vec_margin

            retrieval_span.set(
                top_score=result.top_score,
                top_vec_score=result.top_vec_score,
                margin=result.margin,
                n_chunks=len(result.chunks),
            )
        return result

    def _record_stages(
        self, turn: TurnState, event: TranscriptIn, stopwatch: Stopwatch
    ) -> None:
        """Copy measured stages onto the turn.

        [A] endpoint and [B] stt_final are measured upstream and arrive on the
        event. [C]-[F] are measured here. [G] tts_ttfb and [H] playout are
        measured by the transport and written onto the row afterwards by the
        runner — the harness cannot see them, and inventing them would be
        worse than leaving them NULL.
        """
        turn.stages_ms.update(stopwatch.stages)
        if event.endpoint_ms is not None:
            turn.stages_ms["endpoint"] = event.endpoint_ms
        if event.stt_final_ms is not None:
            turn.stages_ms["stt_final"] = event.stt_final_ms

        first_speak = turn.stages_ms.get("first_speak")
        if first_speak is not None:
            # first_audio counts an acknowledgment as audio; first_content is
            # the honest headline because it does not.
            turn.stages_ms.setdefault("first_audio", first_speak)

    # -----------------------------------------------------------------------
    # Handoff
    # -----------------------------------------------------------------------
    def _decide(self, call: CallState) -> HandoffDecision:
        hard = self.policy.state_rules(call)
        return hard if hard.should_handoff else self.policy.score(call)

    def _rephrase_signal(self, call: CallState, turn: TurnState) -> None:
        """The same question again — after a full answer — is the strongest
        cheap sign the agent is failing. Embedding similarity, because a
        rephrase uses new words."""
        if self.similarity is None:
            return
        # Only against turns the caller HEARD IN FULL. Repeating a request
        # after cutting the agent off, or after declining a read-back, is the
        # caller re-issuing it — not evidence that the agent is failing.
        previous = [
            t.user_text for t in call.turns[:-1]
            if t.user_text and t.stop_reason == "answered" and not t.transactional
        ]
        if any(
            self.similarity(p, turn.user_text) >= self.policy.rephrase_similarity
            for p in previous
        ):
            call.rephrase_count += 1
            turn.handoff_signals.append("rephrase_repeat")

    async def _hand_off(
        self,
        call: CallState,
        turn: TurnState,
        decision: HandoffDecision,
        run: _TurnRun,
        state: _GenerationState,
        stopwatch: Stopwatch,
    ) -> AsyncIterator[OutCommand]:
        """End the call into a human's hands, with a summary they can use."""
        # Nothing waiting for confirmation survives a handoff.
        if call.pending_readback is not None and self.executor is not None:
            await self.executor.cancel(call, call.pending_readback.prepared)
            call.pending_readback = None

        # Policy the call actually touched, across ALL turns: a handoff that
        # pre-gates on "get me a manager" retrieves nothing itself, but the
        # human picking up a refund complaint needs the refund policy.
        refs = list(dict.fromkeys(
            heading for t in call.turns for heading in t.retrieved_headings[:2]
        ))[:4]
        summary = build_summary(call, decision, policy_refs=refs)
        call.handoff = {
            "turn_id": turn.turn_id,
            "reason": decision.reason,
            "rule_type": decision.rule_type,
            "score": decision.score,
            "signals": decision.signals,
            "summary": summary,
            "summary_text": render_summary(summary),
        }
        call.handoff_reason = decision.reason
        call.outcome = "handoff"

        line = prompts.fallback_line(self.domain, "handoff")
        yield self._speak(turn, line, state, stopwatch)
        if call.messages and call.messages[-1]["role"] == "assistant":
            call.messages[-1]["content"].append({"type": "text", "text": line})
        else:
            call.messages.append(ctx.assistant_text(line))
        turn.spoken_text = turn.approved_text
        turn.stop_reason = "handoff"
        turn.handoff_score = decision.score
        yield Emit(
            "handoff_triggered",
            {
                "reason": decision.reason,
                "rule_type": decision.rule_type,
                "score": decision.score,
                "signals": [s.get("signal") for s in decision.signals],
            },
            turn.turn_id,
        )

    # -----------------------------------------------------------------------
    # Read-back: an action awaiting the caller's yes
    # -----------------------------------------------------------------------
    async def _resolve_readback(
        self,
        call: CallState,
        turn: TurnState,
        run: _TurnRun,
        state: _GenerationState,
        stopwatch: Stopwatch,
    ) -> AsyncIterator[OutCommand]:
        pending: PendingReadback = call.pending_readback
        assert self.executor is not None
        tool = pending.prepared.tool.name
        decision = classify(turn.user_text)

        if decision == "confirm":
            call.pending_readback = None
            yield Emit("readback_confirmed", {"tool": tool}, turn.turn_id)
            yield Emit("tool_dispatched", {"tool": tool, "attempt": pending.prepared.execution.attempt},
                       turn.turn_id)
            outcome = await self.executor.dispatch(call, pending.prepared)
            for command in self._tool_outcome_events(turn, tool, outcome):
                yield command
            self._apply_tool_outcome(call, turn, tool, outcome)
            payload = self._payload(outcome)
            run.tool_texts.append(json.dumps(payload))
            run.action_note = (
                f"The caller confirmed {tool} and it has ALREADY been performed. "
                f"Result: {json.dumps(payload)}. Tell them the outcome in one sentence."
            )
            run.withheld_tools.add(tool)
            return

        if decision == "deny":
            await self.executor.cancel(call, pending.prepared)
            call.pending_readback = None
            yield Emit("readback_rejected", {"tool": tool, "reason": "caller_declined"}, turn.turn_id)
            yield Emit("tool_cancelled", {"tool": tool}, turn.turn_id)
            run.action_note = (
                f"The caller did NOT confirm the {tool} you read back, so it was not performed. "
                "Respond to what they said; if they corrected a detail, propose it again."
            )
            return

        # Unclear. Re-ask once, verbatim. Never execute on an ambiguous yes.
        if pending.asked < MAX_READBACK_ASKS:
            pending.asked += 1
            yield self._speak(turn, prompts.fallback_line(self.domain, "readback_unclear"),
                              state, stopwatch, countable=False)
            yield self._speak(turn, pending.text, state, stopwatch)
            yield Emit("readback_spoken", {"tool": tool, "asked": pending.asked}, turn.turn_id)
            turn.spoken_text = turn.approved_text
            call.messages.append(ctx.assistant_text(turn.approved_text))
            turn.stop_reason = "answered"
            run.done = True
            return

        await self.executor.cancel(call, pending.prepared)
        call.pending_readback = None
        yield Emit("readback_rejected", {"tool": tool, "reason": "unclear"}, turn.turn_id)
        turn.handoff_signals.append("readback_unclear")

    async def _start_readback(
        self,
        call: CallState,
        turn: TurnState,
        tool_call: ToolCallDelta,
        run: _TurnRun,
        state: _GenerationState,
        stopwatch: Stopwatch,
    ) -> AsyncIterator[OutCommand]:
        """The model proposed an action that changes an order. Do NOT run it.

        Claim the idempotency key as `pending`, read the details back from the
        template, and end the turn. Dispatch happens only on the caller's yes.
        """
        assert self.executor is not None
        prepared = await self.executor.prepare(
            call, turn.turn_id, tool_call.name, tool_call.arguments
        )
        if prepared.already_committed:
            # The exact same action already went through in this call. Say so
            # rather than silently doing nothing, or doing it twice.
            yield self._speak(turn, prompts.fallback_line(self.domain, "already_done"),
                              state, stopwatch)
            yield Emit("tool_committed", {"tool": tool_call.name, "duplicate_suppressed": True},
                       turn.turn_id)
            run.stop_after_tools = True
            return

        order = call.last_order or {}
        event_name = order.get("event_name") if order.get("order_id") == tool_call.arguments.get(
            "order_id") else None
        text = to_spoken(render(prepared, event_name=event_name))
        yield self._speak(turn, text, state, stopwatch)
        call.pending_readback = PendingReadback(
            prepared=prepared, text=text, turn_id=turn.turn_id, tool_use_id=tool_call.call_id
        )
        run.readback_started = True
        yield Emit(
            "readback_spoken",
            {"tool": tool_call.name, "args": tool_call.arguments, "text": text},
            turn.turn_id,
        )

    # -----------------------------------------------------------------------
    # Generation: stream -> segment -> check -> speak -> tools -> repeat
    # -----------------------------------------------------------------------
    async def _generate(
        self,
        call: CallState,
        turn: TurnState,
        stopwatch: Stopwatch,
        watcher: _BargeInWatcher,
        run: _TurnRun,
        state: _GenerationState,
    ) -> AsyncIterator[OutCommand]:
        segmenter = Segmenter()

        while True:
            if turn.llm_iterations >= self.settings.max_tool_iterations:
                # A model still asking for tools after this many calls does
                # not know what to do. Do not call it again.
                turn.handoff_signals.append("iteration_cap_reached")
                yield Emit("handoff_signal_raised", {"signal": "iteration_cap_reached"},
                           turn.turn_id)
                return

            checker = self._checker_for(call, run)
            tool_calls: list[ToolCallDelta] = []
            async for command in self._stream_once(
                call, turn, stopwatch, watcher, run, state, checker, segmenter, tool_calls
            ):
                yield command
            if watcher.triggered or state.halted or state.failed:
                break

            # Speak whatever the model said before it stopped for a tool.
            tail = segmenter.flush()
            if tail:
                async for command in self._release(call, turn, tail, state, stopwatch, checker, run):
                    yield command
            if state.halted or not tool_calls:
                break

            async for command in self._run_tools(call, turn, stopwatch, watcher, run, state, tool_calls):
                yield command
            if watcher.triggered or run.readback_started or run.stop_after_tools:
                break

        if state.checker_ms:
            stopwatch.stages["checker"] = round(state.checker_ms)

    async def _stream_once(
        self,
        call: CallState,
        turn: TurnState,
        stopwatch: Stopwatch,
        watcher: _BargeInWatcher,
        run: _TurnRun,
        state: _GenerationState,
        checker: Checker | None,
        segmenter: Segmenter,
        tool_calls: list[ToolCallDelta],
    ) -> AsyncIterator[OutCommand]:
        """One model response, with at most one retry on a provider error."""
        for attempt in (1, 2):
            turn.llm_iterations += 1
            request = self._build_request(call, attempt, run)

            with span(
                "llm_iteration", call_id=call.call_id, turn_id=turn.turn_id,
                iteration=turn.llm_iterations, attempt=attempt,
            ) as llm_span:
                llm_span.set(model=self.llm.model)
                yield Emit("llm_iteration_started",
                           {"iteration": turn.llm_iterations, "attempt": attempt}, turn.turn_id)
                stream = _stream_until(
                    self.llm.stream(request),
                    interrupt=watcher.flag,
                    stopwatch=stopwatch,
                    deadlines={
                        # Fires only if nothing has been said yet. A short
                        # fixed acknowledgment so the caller is not left in
                        # silence while the first real sentence generates.
                        "ack": self.settings.ack_after_ms,
                        "budget": turn.latency_budget_ms,
                    },
                )
                try:
                    async for item in stream:
                        if isinstance(item, _Tick):
                            async for command in self._on_deadline(item.name, turn, state, stopwatch):
                                yield command
                            continue

                        if isinstance(item, TextDelta):
                            if state.first_delta_ms is None:
                                state.first_delta_ms = stopwatch.mark("llm_ttft")
                                llm_span.set(ttft_ms=state.first_delta_ms)
                            turn.generated_text += item.text
                            for segment in segmenter.feed(item.text):
                                async for command in self._release(
                                    call, turn, segment, state, stopwatch, checker, run
                                ):
                                    yield command
                                if state.halted:
                                    break
                            if state.halted:
                                break

                        elif isinstance(item, ToolCallDelta):
                            tool_calls.append(item)
                            yield Emit(
                                "llm_tool_call",
                                {"name": item.name, "arguments": item.arguments},
                                turn.turn_id,
                            )

                        elif isinstance(item, StreamEnd):
                            state.stream_end = item
                            llm_span.set(
                                input_tokens=item.input_tokens,
                                output_tokens=item.output_tokens,
                                cache_read_tokens=item.cache_read_tokens,
                            )
                except Exception as exc:
                    # A provider that RAISES instead of returning an error
                    # event takes the same path as one that reports it: retry
                    # once, then a fixed line and a handoff signal. A bug in a
                    # provider must not take the whole call down with it.
                    state.stream_end = StreamEnd(
                        stop_reason="error", error=f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    # Explicit close: breaking out of `async for` does not
                    # close the generator promptly, and an unclosed stream is
                    # an upstream connection still running.
                    await stream.aclose()

            if watcher.triggered or state.halted:
                return

            end = state.stream_end
            if end is None or end.stop_reason != "error":
                if end is not None:
                    yield Emit("llm_text_completed",
                               {"stop_reason": end.stop_reason, "chars": len(turn.generated_text),
                                "output_tokens": end.output_tokens}, turn.turn_id)
                return  # clean finish

            remaining = turn.latency_budget_ms - stopwatch.elapsed_ms()
            yield Emit(
                "error",
                {"stage": "llm", "attempt": attempt, "detail": end.error,
                 "remaining_budget_ms": remaining},
                turn.turn_id,
            )
            tool_calls.clear()
            # Retry once while the budget allows — or, budget or not, while the
            # caller has heard nothing but filler (an acknowledgment, the
            # "give me a second" line). A single Vertex stall used to end the
            # call with the provider_error line; a second attempt is cheaper
            # than a handoff when nothing has been promised yet.
            nothing_said = not turn.approved_text
            if attempt == 2 or (remaining < LLM_RETRY_MIN_BUDGET_MS and not nothing_said):
                turn.handoff_signals.append("llm_failure")
                yield self._speak(
                    turn, prompts.fallback_line(self.domain, "provider_error"),
                    state, stopwatch, countable=False,
                )
                state.failed = True
                return
            state.stream_end = None  # clear before the retry

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------
    async def _run_tools(
        self,
        call: CallState,
        turn: TurnState,
        stopwatch: Stopwatch,
        watcher: _BargeInWatcher,
        run: _TurnRun,
        state: _GenerationState,
        tool_calls: list[ToolCallDelta],
    ) -> AsyncIterator[OutCommand]:
        """Execute this round's tool calls and write them into history.

        History gets an assistant message (text spoken this round + tool_use
        blocks) and a user message of tool_results, so the next model call
        can see what happened. A confirmation-required action is the
        exception: it is never executed here, and its tool_use never enters
        history — the read-back text does instead, and the outcome reaches
        the model later as an action note.
        """
        uses: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        def commit_round() -> None:
            if not uses:
                return
            text = turn.approved_text[run.history_offset:].strip()
            content = ([{"type": "text", "text": text}] if text else []) + uses
            call.messages.append({"role": "assistant", "content": content})
            call.messages.append({"role": "user", "content": list(results)})
            run.history_offset = len(turn.approved_text)
            uses.clear()
            results.clear()

        for tool_call in tool_calls:
            spec = self.registry.get(tool_call.name)
            if self.executor is not None and spec is not None and spec.requires_confirmation:
                commit_round()
                try:
                    async for command in self._start_readback(
                        call, turn, tool_call, run, state, stopwatch
                    ):
                        yield command
                except ToolError as exc:
                    # Invalid arguments or unverified: back to the model to fix.
                    yield Emit("tool_failed", {"tool": tool_call.name, "error": exc.error_type,
                                               "detail": exc.detail}, turn.turn_id)
                    uses.append(_tool_use_block(tool_call))
                    results.append(_tool_result_block(
                        tool_call, {"error": exc.error_type, "detail": exc.detail}))
                    continue
                return  # a read-back ends the turn; later calls are dropped

            box: list[dict[str, Any]] = []
            async for command in self._run_one_tool(call, turn, tool_call, state, stopwatch, watcher, box):
                yield command
            if watcher.triggered:
                # Interrupted mid-tool: the result is dropped from context and
                # nothing about this round enters history, so no tool_result
                # can be orphaned.
                return
            uses.append(_tool_use_block(tool_call))
            results.append(_tool_result_block(tool_call, box[0]))
            run.tool_texts.append(json.dumps(box[0]))

        commit_round()

    async def _run_one_tool(
        self,
        call: CallState,
        turn: TurnState,
        tool_call: ToolCallDelta,
        state: _GenerationState,
        stopwatch: Stopwatch,
        watcher: _BargeInWatcher,
        box: list[dict[str, Any]],
    ) -> AsyncIterator[OutCommand]:
        name = tool_call.name
        if self.executor is None:
            box.append({"error": "tools_unavailable"})
            return
        yield Emit("tool_dispatched", {"tool": name, "args": tool_call.arguments}, turn.turn_id)

        task = asyncio.ensure_future(
            self.executor.run(call, turn.turn_id, name, tool_call.arguments)
        )
        interrupt = asyncio.ensure_future(watcher.flag.wait())
        filler = asyncio.ensure_future(asyncio.sleep(self.settings.tool_filler_ms / 1000))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {task, interrupt, filler}, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    break
                if interrupt in done:
                    # A dispatched call is NEVER cancelled: it resolves or
                    # reconciles in the background, and its result is dropped.
                    self._background.add(task)
                    task.add_done_callback(self._background.discard)
                    return
                if filler in done:
                    lines = prompts.load_domain(self.domain).get("fillers", [])
                    if lines:
                        yield self._speak(turn, lines[0], state, stopwatch, countable=False)
                        yield Emit("filler_spoken", {"tool": name}, turn.turn_id)
                    filler = asyncio.ensure_future(asyncio.sleep(3600))  # once
        finally:
            await _cancel(interrupt)
            await _cancel(filler)

        try:
            outcome = task.result()
        except ToolError as exc:
            yield Emit("tool_failed", {"tool": name, "error": exc.error_type, "detail": exc.detail},
                       turn.turn_id)
            if exc.error_type == "unknown_tool":
                turn.handoff_signals.append("tool_failure")
            box.append({"error": exc.error_type, "detail": exc.detail})
            return

        for command in self._tool_outcome_events(turn, name, outcome):
            yield command
        self._apply_tool_outcome(call, turn, name, outcome)
        box.append(self._payload(outcome))

    def _tool_outcome_events(self, turn: TurnState, name: str, outcome: Outcome) -> list[Emit]:
        payload = {"tool": name, "status": outcome.status, "attempt": outcome.execution.attempt,
                   "error": outcome.error_type,
                   "duplicate_suppressed": outcome.duplicate_suppressed}
        events = []
        if outcome.reconciled:
            events.append(Emit("tool_reconciled", payload, turn.turn_id))
        kind = {"committed": "tool_committed", "failed": "tool_failed",
                "unknown": "tool_unknown"}.get(outcome.status, "tool_failed")
        events.append(Emit(kind, payload, turn.turn_id))
        return events

    def _apply_tool_outcome(self, call: CallState, turn: TurnState, name: str, outcome: Outcome) -> None:
        """Typed tool failures become handoff signals — never improvised answers."""
        if outcome.needs_hard_handoff:
            turn.handoff_signals.append("tool_state_unknown")
            return
        if not outcome.ok:
            turn.handoff_signals.append("tool_failure")
            return
        response = outcome.response or {}
        if name == "verify_caller":
            if response.get("verified"):
                call.caller_verified = True
                call.order_id = response.get("order_id")
            else:
                call.verification_failures += 1
                if call.verification_failures >= self.settings.max_verification_failures:
                    turn.handoff_signals.append("verification_failed")
        elif name == "lookup_order":
            call.last_order = response

    @staticmethod
    def _payload(outcome: Outcome) -> dict[str, Any]:
        if outcome.ok:
            payload = dict(outcome.response or {})
            if outcome.duplicate_suppressed:
                payload["note"] = "already completed earlier in this call; not repeated"
            return payload
        return {"error": outcome.error_type or outcome.status, "status": outcome.status}

    # -----------------------------------------------------------------------
    # Releasing a segment to the caller
    # -----------------------------------------------------------------------
    async def _release(
        self,
        call: CallState,
        turn: TurnState,
        segment: str,
        state: _GenerationState,
        stopwatch: Stopwatch,
        checker: Checker | None,
        run: _TurnRun,
    ) -> AsyncIterator[OutCommand]:
        """The path every model-written segment takes to the caller's ear.

            strip [K] citations -> check -> normalize to spoken form -> speak

        A segment that fails the checker is never spoken. A fixed safe line is
        spoken instead, a handoff signal is raised, and generation for the
        turn stops — the rest of an answer that has just been caught making
        something up is not worth trusting either.

        Segments already spoken cannot be retracted. That is inherent to
        streaming, and it is why generated_text is kept separately: the
        metrics must show what the caller heard before the catch.
        """
        knowledge = run.knowledge
        clean, tags = strip_citations(segment)
        for tag in tags:
            if 1 <= tag <= len(knowledge.chunks):
                chunk_id = knowledge.chunks[tag - 1].chunk_id
                if chunk_id not in turn.cited_chunk_ids:
                    turn.cited_chunk_ids.append(chunk_id)
        if not clean:
            return

        if checker is None:
            yield self._speak(turn, to_spoken(clean), state, stopwatch)
            return

        verdict = checker.check(clean)
        state.checker_ms += verdict.elapsed_ms
        action = "released" if verdict.passed else "replaced"
        turn.checker_verdicts.append(
            CheckerVerdict(
                segment_index=len(turn.checker_verdicts),
                segment_text=clean,
                verdict=verdict.verdict,
                spans=[
                    {"type": s.type, "surface": s.surface, "grounded": s.grounded,
                     "source": s.source}
                    for s in verdict.spans
                ],
                action_taken=action,
                checker_ms=round(verdict.elapsed_ms),
            )
        )
        yield Emit(
            "checker_verdict",
            {
                "segment_index": len(turn.checker_verdicts) - 1,
                "verdict": verdict.verdict,
                "action": action,
                "reason": verdict.reason,
                "spans": len(verdict.spans),
            },
            turn.turn_id,
        )

        if verdict.passed:
            yield self._speak(turn, to_spoken(clean), state, stopwatch)
            return

        kind = (
            "ungrounded_entity"
            if verdict.verdict == "fail_ungrounded_entity"
            else "unsupported_claim"
        )
        yield self._speak(
            turn, prompts.safe_line(self.domain, kind), state, stopwatch, countable=False
        )
        turn.handoff_signals.append("checker_failure")
        call.handoff_signals.append("checker_failure")
        state.halted = True
        yield Emit(
            "handoff_signal_raised",
            {"signal": "checker_failure", "verdict": verdict.verdict, "reason": verdict.reason},
            turn.turn_id,
        )

    def _checker_for(self, call: CallState, run: _TurnRun) -> Checker | None:
        if not self.settings.checker_enabled:
            return None
        try:
            lexicon = load_lexicon(self.domain)
        except FileNotFoundError:
            lexicon = Lexicon.empty()
        return Checker(
            GroundingSources(
                chunks=[f"{c.heading_path}\n{c.text}" for c in run.knowledge.chunks],
                # Tool results are evidence: "you have 2 tickets in section
                # 112" is grounded by the lookup that returned it.
                tool_results=list(run.tool_texts),
                caller=[t.user_text for t in call.turns if t.user_text],
            ),
            lexicon=lexicon,
            retrieval_sufficient=(
                run.knowledge.top_vec_score >= retrieval_sufficiency_threshold(self.domain)
            ),
            claim_overlap_min=claim_overlap_min(self.domain),
        )

    async def _on_deadline(
        self, name: str, turn: TurnState, state: _GenerationState, stopwatch: Stopwatch
    ) -> AsyncIterator[OutCommand]:
        if state.spoke_anything:
            return

        if name == "ack" and turn.social:
            return  # "Let me check that for you." is not a reply to "hi"

        if name == "ack":
            # Fixed strings with no checkable content by construction, so they
            # bypass the checker. Latency is reported both with and without
            # this: the honest headline is first_content, since an
            # acknowledgment front-loads filler rather than information.
            lines = prompts.acknowledgments(self.domain)
            if not lines:
                return
            line = lines[self._ack_index % len(lines)]
            self._ack_index += 1
            turn.ack_bypass_used = True
            yield self._speak(turn, line, state, stopwatch, countable=False)
            return

        if name == "budget" and not turn.budget_exceeded:
            # The budget elapsed with no audio produced. Speak a fixed
            # fallback so the caller is not left in silence, and keep going —
            # the real answer may still arrive.
            turn.budget_exceeded = True
            yield Emit(
                "budget_exceeded",
                {"elapsed_ms": stopwatch.elapsed_ms(), "budget_ms": turn.latency_budget_ms},
                turn.turn_id,
            )
            yield self._speak(
                turn,
                prompts.fallback_line(self.domain, "budget_exceeded"),
                state,
                stopwatch,
                countable=False,
            )

    def _speak(
        self,
        turn: TurnState,
        text: str,
        state: _GenerationState,
        stopwatch: Stopwatch,
        *,
        countable: bool = True,
    ) -> Speak:
        """Hand one segment to TTS, closing the timing stages it completes.

        `countable=False` marks fixed filler lines — fallbacks, safe lines,
        acknowledgments. They are spoken but kept out of `approved_text`,
        because approved_text is what gets written back into conversation
        history, and a filler is not something the agent said about the
        caller's problem.

        approved_text holds the SPOKEN form ("twenty-five dollars", not
        "$25"). It has to: after a barge-in, the transport reports how many
        characters of what TTS received were played, and those characters are
        only meaningful against the same text.

        Two first-audio stages:
          first_speak    any audio, including an acknowledgment
          first_content  the first countable segment — the honest number
        """
        if state.first_speak_ms is None:
            state.first_speak_ms = stopwatch.mark("first_speak")
        if countable and state.first_content_ms is None:
            state.first_content_ms = stopwatch.mark("first_content")
            if state.first_delta_ms is not None:
                # Stage [E]: how long we waited for a complete, checked
                # segment. The latency the checker's existence costs, reported
                # on its own rather than buried in the LLM stage.
                stopwatch.stages["sentence_buffer"] = (
                    state.first_content_ms - state.first_delta_ms
                )

        segment_id = len(turn.segments)
        turn.segments.append(text)
        if countable:
            turn.approved_text = f"{turn.approved_text} {text}".strip()
        state.spoke_anything = True
        return Speak(text=text, segment_id=segment_id, context_id=turn.turn_id)

    def _build_request(self, call: CallState, attempt: int, run: _TurnRun) -> LLMRequest:
        # Retry with a shorter context: if the first call failed on size or a
        # transient upstream issue, a smaller request is the cheapest thing
        # that might work. Never cut between a tool_use and its tool_result.
        messages = call.messages
        if attempt > 1:
            start = max(0, len(messages) - 4)
            while start > 0 and any(
                b.get("type") == "tool_result" for b in messages[start].get("content", [])
            ):
                start -= 1
            messages = messages[start:]

        return LLMRequest(
            system_static=prompts.static_system_prompt(self.domain),
            system_dynamic=prompts.dynamic_system_prompt(
                domain=self.domain,
                caller_verified=call.caller_verified,
                knowledge_block=run.knowledge.knowledge_block(),
                action_note=run.action_note,
            ),
            messages=messages,
            tools=[t for t in self.registry.schemas() if t["name"] not in run.withheld_tools],
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            timeout_ms=self.settings.llm_timeout_ms,
        )

    # -----------------------------------------------------------------------
    # Stop paths — exactly one stop_reason per turn
    # -----------------------------------------------------------------------
    def _finish_normally(
        self, call: CallState, turn: TurnState, run: _TurnRun, state: _GenerationState
    ) -> None:
        turn.history_offset = run.history_offset
        tail = turn.approved_text[run.history_offset:].strip()
        if tail:
            call.messages.append(ctx.assistant_text(tail))
        # Nothing interrupted, so the caller heard everything we approved.
        turn.spoken_text = turn.approved_text

        if state.failed or "llm_failure" in turn.handoff_signals:
            turn.stop_reason = "error"
            call.outcome = "error"
        elif turn.budget_exceeded:
            turn.stop_reason = "budget_exceeded"
        else:
            turn.stop_reason = "answered"

    async def _finish_interrupted(
        self, call: CallState, turn: TurnState, watcher: _BargeInWatcher, run: _TurnRun
    ) -> AsyncIterator[OutCommand]:
        """Barge-in. Order matters: silence the caller's speaker FIRST.

        Everything after CancelSpeech is bookkeeping the caller cannot hear,
        so it can take a few more milliseconds without anyone noticing.
        Getting this order backwards is the common implementation bug, and it
        is audible (docs/10 §3).
        """
        yield CancelSpeech(turn.turn_id)

        # An interruption during a read-back is a NO, not a timeout. That is
        # overwhelmingly what it is, and executing on a maybe is not allowed.
        pending = call.pending_readback
        if pending is not None and pending.turn_id == turn.turn_id and self.executor is not None:
            await self.executor.cancel(call, pending.prepared)
            call.pending_readback = None
            yield Emit("readback_rejected", {"tool": pending.prepared.tool.name,
                                             "reason": "interrupted"}, turn.turn_id)
            yield Emit("tool_cancelled", {"tool": pending.prepared.tool.name}, turn.turn_id)

        played = watcher.event.played_chars if watcher.event else None
        report = ctx.trim_on_interrupt(
            call,
            turn,
            played_chars=played,
            playout_trim_ms=self.settings.playout_trim_ms,
            history_offset=run.history_offset,
        )
        ctx.assert_invariant(call.messages)

        turn.stop_reason = "interrupted"
        yield Emit(
            "barge_in",
            {"played_chars": played, "spoken_chars": len(turn.spoken_text)},
            turn.turn_id,
        )
        yield Emit(
            "context_trimmed",
            {
                "messages_before": report.messages_before,
                "messages_after": report.messages_after,
                "chars_dropped": report.chars_dropped,
                "orphan_tool_results_dropped": report.orphan_tool_results_dropped,
                "assistant_message_removed": report.assistant_message_removed,
            },
            turn.turn_id,
        )

    async def _barge_in_after_turn(
        self, call: CallState, event: BargeIn
    ) -> AsyncIterator[OutCommand]:
        """The caller cut in AFTER the turn finished generating.

        This is the common case in live audio, not an edge case. A turn
        completes when its text is handed to TTS, not when playback ends: the
        model finishes in about a second, and TTS keeps speaking for several
        more. A barge-in in that window means the caller did NOT hear the end
        of the last turn, and history must be rewritten to say so — exactly
        as for an interruption mid-generation.

        It needs a played-character count to act on. Without one there is no
        way to tell "cut in mid-sentence" from "started talking after the
        agent finished", and trimming on a guess would erase speech the
        caller heard.
        """
        last = call.turns[-1] if call.turns else None
        pending = call.pending_readback

        # A read-back is a question awaiting an answer. Being talked over
        # while asking it is a NO — never executed on a maybe — whether or
        # not we know how far playback got.
        if pending is not None and self.executor is not None:
            await self.executor.cancel(call, pending.prepared)
            call.pending_readback = None
            yield Emit("readback_rejected", {"tool": pending.prepared.tool.name,
                                             "reason": "interrupted"}, pending.turn_id)
            yield Emit("tool_cancelled", {"tool": pending.prepared.tool.name}, pending.turn_id)

        played = event.played_chars
        still_playing = (
            last is not None
            and played is not None
            and last.stop_reason in ("answered", "budget_exceeded")
            and played < len(last.approved_text)
        )
        if not still_playing:
            yield Emit("barge_in", {"while_idle": True, "played_chars": played})
            return

        assert last is not None
        yield CancelSpeech(last.turn_id)
        report = ctx.trim_on_interrupt(
            call, last, played_chars=played,
            playout_trim_ms=self.settings.playout_trim_ms,
            history_offset=last.history_offset,
        )
        ctx.assert_invariant(call.messages)
        last.stop_reason = "interrupted"
        yield Emit(
            "barge_in",
            {"during_playout": True, "played_chars": played, "spoken_chars": len(last.spoken_text)},
            last.turn_id,
        )
        yield Emit(
            "context_trimmed",
            {"messages_before": report.messages_before, "messages_after": report.messages_after,
             "chars_dropped": report.chars_dropped,
             "assistant_message_removed": report.assistant_message_removed},
            last.turn_id,
        )
        # Re-persist: the stored row said "answered" with the full text.
        yield TurnCompleted(last)

    async def _handle_silence(self, call: CallState) -> AsyncIterator[OutCommand]:
        self._silence_reprompts += 1
        if self._silence_reprompts <= MAX_SILENCE_REPROMPTS:
            yield Emit("silence_reprompt", {"count": self._silence_reprompts})
            yield Speak(
                text=prompts.fallback_line(self.domain, "silence_reprompt"),
                segment_id=0,
                context_id=f"silence-{self._silence_reprompts}",
            )
            return

        call.outcome = "dropped"
        yield Emit("call_ended", {"outcome": "dropped", "reason": "silence"})


def _tool_use_block(tool_call: ToolCallDelta) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_call.call_id, "name": tool_call.name,
            "input": tool_call.arguments}


def _tool_result_block(tool_call: ToolCallDelta, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_call.call_id,
            "content": json.dumps(payload)}


# ---------------------------------------------------------------------------
# Generation bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class _GenerationState:
    """Timing and control flags for one turn's speech."""

    first_delta_ms: int | None = None
    first_speak_ms: int | None = None
    first_content_ms: int | None = None
    spoke_anything: bool = False
    #: The checker rejected a segment: generation for the turn stops.
    halted: bool = False
    #: The model failed twice: a fixed line was spoken, the turn errors out.
    failed: bool = False
    checker_ms: float = 0.0
    stream_end: StreamEnd | None = None


@dataclass
class _TurnRun:
    """Per-turn working state that is not worth persisting."""

    knowledge: RetrievalResult
    #: One line for the dynamic prompt — e.g. the outcome of a confirmed action.
    action_note: str | None = None
    #: Tool results this turn, as text. Grounding sources for the checker.
    tool_texts: list[str] = field(default_factory=list)
    #: How much of approved_text is already in history next to a tool_use.
    history_offset: int = 0
    readback_started: bool = False
    #: Stop after this round of tools — e.g. an action was already done.
    stop_after_tools: bool = False
    #: The turn is complete before any generation (a re-asked read-back).
    done: bool = False
    #: Tools withheld from the model this turn. After a confirmed action the
    #: model only has to REPORT the outcome; offering the same tool again
    #: invited a second call (Gemini made one every time). The ledger would
    #: suppress it, but the caller then heard "already gone through".
    withheld_tools: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Barge-in plumbing
# ---------------------------------------------------------------------------
class _BargeInWatcher:
    """Drains the event stream during a turn, looking for a BargeIn.

    Anything else it pulls off is held in `deferred` and pushed back when the
    turn ends, so nothing is lost and ordering is preserved.
    """

    def __init__(self, stream: EventStream) -> None:
        self._stream = stream
        self._task: asyncio.Task[None] | None = None
        self.flag = asyncio.Event()
        self.event: BargeIn | None = None
        self.deferred: list[InEvent] = []

    @property
    def triggered(self) -> bool:
        return self.flag.is_set()

    def start(self) -> None:
        self._task = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        while True:
            event = await self._stream.get()
            if isinstance(event, BargeIn):
                self.event = event
                self.flag.set()
                return
            self.deferred.append(event)

    async def stop(self) -> list[InEvent]:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self.deferred


@dataclass(frozen=True)
class _Tick:
    """A named deadline elapsed ("ack" or "budget"). Each fires at most once."""

    name: str


async def _stream_until(
    source: AsyncIterator[Any],
    *,
    interrupt: asyncio.Event,
    stopwatch: Stopwatch,
    deadlines: dict[str, int],
) -> AsyncIterator[Any]:
    """Yield from `source`, waking early for a barge-in or a deadline.

    A plain `async for` would block on the next token. Barge-in has to be
    detected while we are waiting for the FIRST token — exactly when a caller
    is most likely to interrupt — so each step races the next event, the
    interrupt flag, and every pending deadline timer.

    The pending `__anext__` task is deliberately kept across a deadline tick
    rather than cancelled and recreated. Cancelling an in-flight `__anext__`
    and then re-entering the generator raises "already running"; holding the
    task is both correct and cheaper.
    """
    iterator = source.__aiter__()
    next_task: asyncio.Future[Any] | None = None
    timers: dict[asyncio.Future[None], str] = {
        asyncio.ensure_future(asyncio.sleep(max(0, ms - stopwatch.elapsed_ms()) / 1000)): name
        for name, ms in deadlines.items()
    }

    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(iterator.__anext__())
            interrupt_task = asyncio.ensure_future(interrupt.wait())

            waiting: set[asyncio.Future[Any]] = {next_task, interrupt_task, *timers}
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

            if interrupt_task in done:
                # Abandoning the stream entirely, so cancelling the pending
                # __anext__ is safe — `finally` closes the generator next.
                await _cancel(next_task)
                return

            await _cancel(interrupt_task)

            fired = [timer for timer in done if timer in timers]
            if fired:
                for timer in sorted(fired, key=lambda t: deadlines[timers[t]]):
                    yield _Tick(timers.pop(timer))
                if next_task not in done:
                    continue  # next_task is preserved

            if next_task in done:
                try:
                    value = next_task.result()
                except StopAsyncIteration:
                    return
                finally:
                    next_task = None
                yield value
    finally:
        for timer in timers:
            await _cancel(timer)
        await _cancel(next_task)
        await _aclose(source)


async def _cancel(task: asyncio.Future[Any] | None) -> None:
    """Cancel and await a task, swallowing its outcome.

    Awaiting matters: an un-awaited cancelled task logs a warning, and an
    un-retrieved exception logs a louder one.
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, StopAsyncIteration, Exception):
        pass


async def _aclose(source: Any) -> None:
    """Close an async generator so the upstream connection really goes away.

    A leaked stream shows up as mysterious latency three turns later.
    """
    close = getattr(source, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
