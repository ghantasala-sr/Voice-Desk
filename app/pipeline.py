"""Pipecat wiring — the ONLY module in this repo that imports pipecat.

Everything here is audio: transport, VAD, streaming STT and TTS, and the
interruption event. Nothing here decides what to say, when to call a tool, or
whether to hand off.

Requires the optional audio extra and three API keys:

    .venv/bin/pip install -e '.[audio]'
    # ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY in .env

`make chat` needs none of that — the harness runs without an audio stack,
which is the whole point of the boundary.

NOTE ON VERSIONS: Pipecat's service and serializer APIs move between releases.
The imports below are isolated in `_build_services()` precisely so that a
version bump is a change to one function, not a change to the call flow.
Verify them against the version pinned in uv.lock when you first install.
"""

from __future__ import annotations

import asyncio
from typing import Any

from analytics.events import PostgresEventWriter
from analytics.store import NullStore, PostgresStore
from app.bridge import PlayoutTracker, TurnTimings, transcript_message, ws_message
from core.clock import SystemClock
from core.ids import new_id
from core.logging import bind_call, get_logger
from core.settings import Settings
from harness.events import (
    BargeIn,
    CancelSpeech,
    Emit,
    EndCall,
    EventStream,
    Shutdown,
    Speak,
    TranscriptIn,
    TurnCompleted,
)
from harness.loop import Harness
from harness.state import CallState
from providers import build_llm, config_snapshot

log = get_logger(__name__)

# The browser sends 16 kHz mono PCM16 in 20 ms frames (640 bytes).
INPUT_SAMPLE_RATE = 16_000


async def run_call(websocket: Any, *, settings: Settings) -> None:
    """Run one browser call end to end."""
    call = CallState(
        call_id=new_id(),
        domain_config=settings.domain,
        source="browser",
        config_snapshot=config_snapshot(settings),
    )
    bind_call(call.call_id)

    store: Any = PostgresStore()
    writer: Any = PostgresEventWriter()
    try:
        await store.create_call(call)
    except Exception as exc:
        log.warning("call_not_recorded", detail=str(exc))
        store, writer = NullStore(), _NullWriter()

    from harness.assembly import build_stack

    stack = await build_stack(
        settings, domain=settings.domain, retrieval="real", tools=True,
        persist_ledger=isinstance(store, PostgresStore),
    )
    for line in stack.banner:
        log.info("stack", detail=line)

    stream = EventStream()
    harness = Harness(
        llm=build_llm(settings), settings=settings, clock=SystemClock(),
        retriever=stack.retriever, executor=stack.executor, similarity=stack.similarity,
    )
    session = _Session(
        websocket=websocket, call=call, stream=stream, store=store, writer=writer
    )

    await websocket.send_text(ws_message("call_started", call_id=call.call_id))

    pipeline_task = asyncio.create_task(_run_audio(session, settings))
    try:
        async for command in harness.run(call, stream):
            await session.handle(command)
    finally:
        pipeline_task.cancel()
        await writer.flush()
        await store.finish_call(call, duration_ms=session.elapsed_ms())


# ---------------------------------------------------------------------------
# Harness command -> browser / TTS
# ---------------------------------------------------------------------------
class _Session:
    """Forwards harness commands outward and audio events inward."""

    def __init__(self, *, websocket: Any, call: CallState, stream: EventStream,
                 store: Any, writer: Any) -> None:
        self.websocket = websocket
        self.call = call
        self.stream = stream
        self.store = store
        self.writer = writer
        self.clock = SystemClock()
        self._started_ms = self.clock.now_ms()

        self.playout = PlayoutTracker()
        self.timings = TurnTimings()
        self.current_turn_id: str | None = None
        self.tts: Any = None  # set by _run_audio

    def elapsed_ms(self) -> int:
        return self.clock.now_ms() - self._started_ms

    async def handle(self, command: Any) -> None:
        if isinstance(command, Speak):
            if command.context_id != self.current_turn_id:
                self.current_turn_id = command.context_id
                self.playout.reset()
            self.playout.segment_started(command.segment_id, command.text)
            await self.websocket.send_text(
                transcript_message("agent", command.text, segment_id=command.segment_id)
            )
            if self.tts is not None:
                await self.tts.speak(command.text, context_id=command.context_id)

        elif isinstance(command, CancelSpeech):
            # Flush the transport buffer FIRST. The caller's experience is
            # decided by when sound stops leaving their speaker; upstream
            # cleanup can take another few milliseconds unnoticed.
            if self.tts is not None:
                await self.tts.cancel(command.context_id)
            await self.websocket.send_text(ws_message("cancel_speech"))

        elif isinstance(command, Emit):
            await self.writer.emit(
                self.call.call_id, command.event_type, command.payload, command.turn_id
            )
            await self.websocket.send_text(
                ws_message("event", name=command.event_type, payload=command.payload)
            )

        elif isinstance(command, TurnCompleted):
            turn = command.turn
            await self.store.save_turn(self.call, turn)
            # Stages [G] and [H] are measured out here, not in the harness.
            await self.store.update_turn_audio_timing(
                turn.turn_id,
                tts_ttfb_ms=self.timings.tts_ttfb_ms,
                playout_ms=self.timings.playout_ms,
                first_audio_ms=self.timings.first_audio_ms,
                voice_to_voice_ms=self.timings.voice_to_voice_ms(),
            )
            await self.websocket.send_text(
                ws_message(
                    "metrics",
                    stages=turn.stages_ms,
                    stop_reason=turn.stop_reason,
                    voice_to_voice_ms=self.timings.voice_to_voice_ms(),
                )
            )
            self.timings = TurnTimings()

        elif isinstance(command, EndCall):
            await self.websocket.send_text(
                ws_message("call_ended", outcome=self.call.outcome or command.reason)
            )

    # -- inbound, called by the Pipecat shim --------------------------------
    def on_user_started_speaking(self) -> None:
        """Barge-in. `played_chars` is the honest estimate of what was heard."""
        self.stream.put(BargeIn(played_chars=self.playout.played_chars()))

    def on_user_stopped_speaking(self) -> None:
        self.timings.user_stopped_ms = self.clock.now_ms()

    def on_final_transcript(self, text: str, confidence: float,
                            words: list[tuple[str, float]], provider: str) -> None:
        endpoint_ms = int(self.call.config_snapshot["vad"]["stop_secs"] * 1000)
        self.stream.put(
            TranscriptIn(
                text=text,
                confidence=confidence,
                word_confidences=words,
                provider=provider,
                endpoint_ms=endpoint_ms,
            )
        )

    def on_audio_out(self, num_bytes: int) -> None:
        self.playout.audio_delivered(num_bytes)
        if self.timings.first_audio_ms is None:
            self.timings.first_audio_ms = self.clock.now_ms()

    def on_segment_drained(self, segment_id: int) -> None:
        self.playout.segment_completed(segment_id)

    def on_closed(self) -> None:
        self.stream.put(Shutdown("transport_closed"))


class _NullWriter:
    async def emit(self, *args: Any, **kwargs: Any) -> None: ...
    async def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# The Pipecat half
# ---------------------------------------------------------------------------
async def _run_audio(session: _Session, settings: Settings) -> None:
    """Build and run the Pipecat pipeline, feeding `session` as events arrive.

    Kept to one function so a Pipecat version bump touches nothing else.
    """
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as exc:
        log.error("audio_extra_missing", detail=str(exc))
        await session.websocket.send_text(
            ws_message(
                "error",
                message="Audio dependencies are not installed. "
                "Run: pip install -e '.[audio]'  — or use `make chat` for text mode.",
            )
        )
        session.on_closed()
        return

    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=settings.vad_confidence,
            start_secs=settings.vad_start_secs,
            stop_secs=settings.vad_stop_secs,
        )
    )
    transport = FastAPIWebsocketTransport(
        websocket=session.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            vad_analyzer=vad,
        ),
    )
    stt = DeepgramSTTService(
        api_key=settings.require("deepgram_api_key"),
        live_options={"model": "nova-3", "smart_format": True, "numerals": True},
    )
    tts = CartesiaTTSService(
        api_key=settings.require("cartesia_api_key"),
        voice_id=settings.cartesia_voice_id,
    )
    session.tts = _TTSAdapter(tts)

    adapter = _HarnessFrameProcessor(session)
    task = PipelineTask(
        Pipeline([transport.input(), stt, adapter, tts, transport.output()]),
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
    )
    await PipelineRunner().run(task)


class _TTSAdapter:
    """Thin wrapper so `_Session` does not need Pipecat's frame vocabulary."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def speak(self, text: str, *, context_id: str) -> None:
        from pipecat.frames.frames import TTSSpeakFrame

        await self._service.queue_frame(TTSSpeakFrame(text))

    async def cancel(self, context_id: str) -> None:
        await self._service.interrupt()


def _HarnessFrameProcessor(session: _Session) -> Any:
    """Pipecat FrameProcessor that forwards audio events into the harness.

    Defined inside a function so the pipecat import stays lazy — importing
    this module must not require the audio extra.
    """
    from pipecat.frames.frames import (
        Frame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class _Processor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, UserStartedSpeakingFrame):
                session.on_user_started_speaking()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                session.on_user_stopped_speaking()
            elif isinstance(frame, TranscriptionFrame):
                session.on_final_transcript(
                    frame.text, confidence=1.0, words=[], provider="deepgram"
                )

            # Audio frames flow on untouched; the harness produces speech via
            # Speak commands, not by passing frames through.
            await self.push_frame(frame, direction)

    return _Processor()
