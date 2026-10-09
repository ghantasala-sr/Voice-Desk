"""Persisting calls and turns.

State is flushed per TURN, not at the end of the call. A dropped WebSocket
must not lose the conversation — and the browser client's resume path
(docs/10 §4) reloads CallState from these rows.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from harness.state import CallState, TurnState


class CallStore(Protocol):
    async def create_call(self, call: CallState) -> None: ...
    async def save_turn(self, call: CallState, turn: TurnState) -> None: ...
    async def finish_call(self, call: CallState, duration_ms: int) -> None: ...
    async def update_turn_audio_timing(
        self,
        turn_id: str,
        *,
        tts_ttfb_ms: int | None,
        playout_ms: int | None,
        first_audio_ms: int | None,
        voice_to_voice_ms: int | None,
    ) -> None: ...


class NullStore:
    """Keeps everything in memory. Used by harness tests and the text driver."""

    def __init__(self) -> None:
        self.calls: list[CallState] = []
        self.turns: list[TurnState] = []

    async def create_call(self, call: CallState) -> None:
        self.calls.append(call)

    async def save_turn(self, call: CallState, turn: TurnState) -> None:
        self.turns.append(turn)

    async def finish_call(self, call: CallState, duration_ms: int) -> None:
        return None

    async def update_turn_audio_timing(self, turn_id: str, **kwargs: Any) -> None:
        return None


class PostgresStore:
    async def create_call(self, call: CallState) -> None:
        from core.db import execute

        await execute(
            """
            INSERT INTO calls (call_id, domain_config, source, config_snapshot)
            VALUES (%(call_id)s, %(domain)s, %(source)s, %(snapshot)s::jsonb)
            ON CONFLICT (call_id) DO NOTHING
            """,
            {
                "call_id": call.call_id,
                "domain": call.domain_config,
                "source": call.source,
                "snapshot": json.dumps(call.config_snapshot),
            },
        )

    async def save_turn(self, call: CallState, turn: TurnState) -> None:
        from core.db import execute

        stages = turn.stages_ms
        await execute(
            """
            INSERT INTO turns (
              turn_id, call_id, seq, trigger,
              user_text, stt_confidence, stt_provider,
              retrieved_chunk_ids, retrieval_top_score, retrieval_margin,
              retrieval_top_vec_score, retrieval_top_lex_score, retrieval_vec_margin,
              llm_iterations, generated_text, approved_text, spoken_text,
              stop_reason, handoff_signals, handoff_score,
              endpoint_ms, stt_final_ms, retrieval_ms, llm_ttft_ms,
              sentence_buffer_ms, checker_ms, tts_ttfb_ms, playout_ms,
              first_audio_ms, voice_to_voice_ms, ack_bypass_used, trace_id,
              first_content_ms, cited_chunk_ids
            ) VALUES (
              %(turn_id)s, %(call_id)s, %(seq)s, %(trigger)s,
              %(user_text)s, %(stt_confidence)s, %(stt_provider)s,
              %(chunk_ids)s, %(top_score)s, %(margin)s,
              %(top_vec)s, %(top_lex)s, %(vec_margin)s,
              %(llm_iterations)s, %(generated)s, %(approved)s, %(spoken)s,
              %(stop_reason)s, %(signals)s, %(handoff_score)s,
              %(endpoint)s, %(stt_final)s, %(retrieval)s, %(llm_ttft)s,
              %(buffer)s, %(checker)s, %(tts_ttfb)s, %(playout)s,
              %(first_audio)s, %(v2v)s, %(ack)s, %(trace_id)s,
              %(first_content)s, %(cited)s
            )
            ON CONFLICT (turn_id) DO UPDATE SET
              stop_reason = EXCLUDED.stop_reason,
              spoken_text = EXCLUDED.spoken_text,
              approved_text = EXCLUDED.approved_text
            """,
            {
                "turn_id": turn.turn_id,
                "call_id": call.call_id,
                "seq": turn.seq,
                "trigger": turn.trigger,
                "user_text": turn.user_text,
                "stt_confidence": turn.stt_confidence,
                "stt_provider": turn.stt_provider,
                "chunk_ids": turn.retrieved_chunk_ids,
                "top_score": turn.retrieval_top_score,
                "margin": turn.retrieval_margin,
                "top_vec": turn.retrieval_top_vec_score,
                "top_lex": turn.retrieval_top_lex_score,
                "vec_margin": turn.retrieval_vec_margin,
                "llm_iterations": turn.llm_iterations,
                "generated": turn.generated_text,
                "approved": turn.approved_text,
                "spoken": turn.spoken_text,
                "stop_reason": turn.stop_reason,
                "signals": turn.handoff_signals,
                "handoff_score": turn.handoff_score,
                "endpoint": stages.get("endpoint"),
                "stt_final": stages.get("stt_final"),
                "retrieval": stages.get("retrieval"),
                "llm_ttft": stages.get("llm_ttft"),
                "buffer": stages.get("sentence_buffer"),
                "checker": stages.get("checker"),
                "tts_ttfb": stages.get("tts_ttfb"),
                "playout": stages.get("playout"),
                "first_audio": stages.get("first_audio"),
                "v2v": stages.get("voice_to_voice"),
                "ack": turn.ack_bypass_used,
                "trace_id": turn.trace_id,
                "first_content": stages.get("first_content"),
                "cited": turn.cited_chunk_ids,
            },
        )
        await self._save_verdicts(call, turn)
        await execute(
            "UPDATE calls SET turn_count = %(n)s WHERE call_id = %(call_id)s",
            {"n": len(call.turns), "call_id": call.call_id},
        )

    async def _save_verdicts(self, call: CallState, turn: TurnState) -> None:
        """One row per checker verdict — catch rate is computed per verdict and
        grouped by hallucination class, so these cannot live in a JSONB blob."""
        if not turn.checker_verdicts:
            return
        from core.db import pool

        async with pool().connection() as conn, conn.cursor() as cur:
            await cur.execute("DELETE FROM checker_verdicts WHERE turn_id = %s", (turn.turn_id,))
            await cur.executemany(
                """
                INSERT INTO checker_verdicts
                  (call_id, turn_id, segment_index, segment_text, verdict,
                   checkable_spans, action_taken, checker_ms)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                [
                    (
                        call.call_id, turn.turn_id, v.segment_index, v.segment_text,
                        v.verdict, json.dumps(v.spans), v.action_taken, v.checker_ms,
                    )
                    for v in turn.checker_verdicts
                ],
            )

    async def finish_call(self, call: CallState, duration_ms: int) -> None:
        from core.db import execute

        await execute(
            """
            UPDATE calls
               SET outcome = %(outcome)s,
                   handoff_reason = %(handoff_reason)s,
                   caller_verified = %(verified)s,
                   order_id = %(order_id)s,
                   rephrase_count = %(rephrase)s,
                   ended_at = now(),
                   duration_ms = %(duration)s
             WHERE call_id = %(call_id)s
            """,
            {
                "outcome": call.outcome or "dropped",
                "handoff_reason": call.handoff_reason,
                "verified": call.caller_verified,
                "order_id": call.order_id,
                "rephrase": call.rephrase_count,
                "duration": duration_ms,
                "call_id": call.call_id,
            },
        )
        if call.handoff:
            await self._save_handoff(call)

    async def _save_handoff(self, call: CallState) -> None:
        from core.db import execute
        from core.ids import new_id

        handoff = call.handoff
        assert handoff is not None
        await execute(
            """
            INSERT INTO handoffs
              (handoff_id, call_id, turn_id, reason, rule_type, signals, score,
               summary, summary_text)
            VALUES (%(id)s, %(call)s, %(turn)s, %(reason)s, %(rule)s, %(signals)s::jsonb,
                    %(score)s, %(summary)s::jsonb, %(text)s)
            """,
            {
                "id": new_id(), "call": call.call_id, "turn": handoff["turn_id"],
                "reason": handoff["reason"], "rule": handoff["rule_type"],
                "signals": json.dumps(handoff["signals"]), "score": handoff["score"],
                "summary": json.dumps(handoff["summary"], default=str),
                "text": handoff["summary_text"],
            },
        )

    async def update_turn_audio_timing(
        self,
        turn_id: str,
        *,
        tts_ttfb_ms: int | None = None,
        playout_ms: int | None = None,
        first_audio_ms: int | None = None,
        voice_to_voice_ms: int | None = None,
    ) -> None:
        """Written by the transport once real audio timing is known.

        The harness cannot see TTS time-to-first-byte or playout start, so it
        leaves those NULL rather than inventing them.
        """
        from core.db import execute

        await execute(
            """
            UPDATE turns
               SET tts_ttfb_ms       = COALESCE(%(tts)s, tts_ttfb_ms),
                   playout_ms        = COALESCE(%(playout)s, playout_ms),
                   first_audio_ms    = COALESCE(%(first_audio)s, first_audio_ms),
                   voice_to_voice_ms = COALESCE(%(v2v)s, voice_to_voice_ms)
             WHERE turn_id = %(turn_id)s
            """,
            {
                "tts": tts_ttfb_ms,
                "playout": playout_ms,
                "first_audio": first_audio_ms,
                "v2v": voice_to_voice_ms,
                "turn_id": turn_id,
            },
        )
