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
from app.bridge import AudioTimeline, PlayoutTracker, transcript_message, ws_message
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
from providers import build_llm, config_snapshot, offline_fallback

log = get_logger(__name__)

# The browser sends 16 kHz mono PCM16 in 20 ms frames (640 bytes), and its
# AudioContext plays back at the same rate (app/static/client.js).
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 16_000
#: Hang-up waits for this much transport silence — longer than the gap
#: between two queued segments, shorter than anyone would notice.
QUIET_MS = 700
#: The browser's playback worklet holds a little audio of its own.
CLIENT_BUFFER_S = 0.4


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
        llm=build_llm(offline_fallback(settings)), settings=settings, clock=SystemClock(),
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
        if session.tts is not None:
            session.tts.close()
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

        self.playout = PlayoutTracker(sample_rate=OUTPUT_SAMPLE_RATE)
        self._queued_segments: list[int] = []
        self._playout_end_ms = 0
        self._bot_speaking = False
        self._bot_changed_ms = 0
        self.timeline = AudioTimeline()
        self.current_turn_id: str | None = None
        self.tts: Any = None  # set by _run_audio

    def elapsed_ms(self) -> int:
        return self.clock.now_ms() - self._started_ms

    async def handle(self, command: Any) -> None:
        if isinstance(command, Speak):
            if command.context_id != self.current_turn_id:
                self.current_turn_id = command.context_id
                self.playout.reset()
                self._queued_segments.clear()
            self.playout.segment_started(command.segment_id, command.text)
            self._queued_segments.append(command.segment_id)
            self.timeline.speak(command.context_id, self.clock.now_ms())
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
            self._queued_segments.clear()
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
            await self.websocket.send_text(
                ws_message("metrics", stages=turn.stages_ms, stop_reason=turn.stop_reason,
                           voice_to_voice_ms=None)
            )
            # Stages [G] and [H] are measured out here, not in the harness —
            # and usually AFTER this point, when the turn's audio starts.
            self.timeline.completed(turn.turn_id)
            await self._write_audio_timing(turn.turn_id)

        elif isinstance(command, EndCall):
            await self.drain_speech()
            await self.websocket.send_text(
                ws_message("call_ended", outcome=self.call.outcome or command.reason)
            )

    # -- inbound, called by the Pipecat shim --------------------------------
    def on_user_started_speaking(self) -> None:
        """Barge-in. `played_chars` is the honest estimate of what was heard.

        Only while the agent is SPEAKING (audio playing or queued). Found live:
        VAD fired once more on a short "Hi there." while the model was still
        thinking; the turn was cancelled with nothing said and the caller got
        silence. Before the agent speaks there is nothing to interrupt — if the
        caller says more, it arrives as the next transcript."""
        self.timeline.user_started(self.clock.now_ms())
        if not (self._bot_speaking or (self.tts is not None and self.tts.busy())):
            log.info("barge_in_ignored", reason="agent_not_speaking")
            return
        self.stream.put(BargeIn(played_chars=self.playout.played_chars()))

    def on_user_stopped_speaking(self) -> None:
        self.timeline.user_stopped(self.clock.now_ms())

    async def _write_audio_timing(self, turn_id: str) -> None:
        timings = self.timeline.take_ready(turn_id)
        if timings is None:
            return
        try:
            await self.store.update_turn_audio_timing(turn_id, **timings)
            await self.websocket.send_text(ws_message("metrics", audio=timings,
                                                      voice_to_voice_ms=timings["voice_to_voice_ms"]))
        except Exception as exc:  # a closed socket must not lose the DB write's error
            log.warning("audio_timing_not_sent", detail=str(exc))

    def on_final_transcript(self, text: str, confidence: float,
                            words: list[tuple[str, float]], provider: str) -> None:
        endpoint_ms = int(self.call.config_snapshot["vad"]["stop_secs"] * 1000)
        self.timeline.transcript(self.clock.now_ms())
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
        # The browser plays in real time from the first byte, so the end of
        # playout advances by each chunk's duration.
        now = self.clock.now_ms()
        self._playout_end_ms = max(self._playout_end_ms, now) + num_bytes * 1000 // (OUTPUT_SAMPLE_RATE * 2)
        self.playout.audio_delivered(num_bytes)
        started = self.timeline.audio(now)
        if started is not None:
            # Called from the serializer (sync): write the row from a task.
            asyncio.get_running_loop().create_task(self._write_audio_timing(started))

    def on_segment_drained(self, segment_id: int) -> None:
        self.playout.segment_completed(segment_id)

    def on_tts_segment_done(self) -> None:
        if self._queued_segments:
            self.on_segment_drained(self._queued_segments.pop(0))
        if self.tts is not None:
            self.tts.segment_done()

    def on_bot_speaking(self, speaking: bool) -> None:
        self._bot_speaking = speaking
        self._bot_changed_ms = self.clock.now_ms()

    async def drain_speech(self, limit_s: float = 20.0) -> None:
        """Before hanging up: let queued speech generate, then let it finish
        PLAYING — the transport has been quiet for QUIET_MS, plus the client's
        own playback buffer. Cancelling at once cut the handoff line, so the
        caller never heard that they were being passed to a person."""
        if self.tts is None:
            return
        deadline = self.clock.now_ms() + int(limit_s * 1000)
        await self.tts.wait_idle(limit_s)
        while self.clock.now_ms() < deadline:
            quiet_for = self.clock.now_ms() - self._bot_changed_ms
            if not self._bot_speaking and quiet_for >= QUIET_MS:
                break
            await asyncio.sleep(0.05)
        remaining = (self._playout_end_ms - self.clock.now_ms()) / 1000
        await asyncio.sleep(max(0.0, min(remaining, 2.0)) + CLIENT_BUFFER_S)

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
    Written for Pipecat 1.x: the FastAPI transport lives in
    `transports.websocket.fastapi`, VAD is its own processor rather than a
    transport parameter, and the transport needs a serializer for the wire
    format — ours is raw 16 kHz PCM16 in binary frames plus JSON control text.
    """
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as exc:
        log.error("audio_extra_missing", detail=str(exc))
        await session.websocket.send_text(
            ws_message(
                "error",
                message="Audio dependencies are not installed. "
                "Run: pip install 'pipecat-ai[deepgram,cartesia,silero,websocket]' — "
                "or use the Talk tab / `make chat` for text mode.",
            )
        )
        session.on_closed()
        return

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        sample_rate=INPUT_SAMPLE_RATE,
        params=VADParams(
            confidence=settings.vad_confidence,
            start_secs=settings.vad_start_secs,
            stop_secs=settings.vad_stop_secs,
        ),
    ))
    transport = FastAPIWebsocketTransport(
        websocket=session.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            add_wav_header=False,
            serializer=_raw_pcm_serializer(session),
        ),
    )
    stt = DeepgramSTTService(
        api_key=settings.require("deepgram_api_key"),
        sample_rate=INPUT_SAMPLE_RATE,
        live_options=LiveOptions(model="nova-3", smart_format=True, numerals=True,
                                 interim_results=True, language="en-US"),
    )
    tts = CartesiaTTSService(
        api_key=settings.require("cartesia_api_key"),
        voice_id=settings.require("cartesia_voice_id"),
        sample_rate=OUTPUT_SAMPLE_RATE,
    )

    adapter = _HarnessFrameProcessor(session)
    session.tts = _TTSAdapter(adapter)
    task = PipelineTask(
        Pipeline([transport.input(), vad, stt, adapter, tts,
                  _PlayoutObserver(session), transport.output()]),
        params=PipelineParams(audio_in_sample_rate=INPUT_SAMPLE_RATE,
                              audio_out_sample_rate=OUTPUT_SAMPLE_RATE, enable_metrics=True),
    )

    @transport.event_handler("on_client_disconnected")
    async def _gone(_transport: Any, _client: Any) -> None:
        session.on_closed()
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


class _TTSAdapter:
    """Thin wrapper so `_Session` does not need Pipecat's frame vocabulary.

    Speech and cancellation are pushed as frames from the harness adapter, so
    they travel the pipeline in order: a cancel reaches the TTS service AND the
    output transport, which is what actually stops sound leaving the speaker.

    Segments go to TTS ONE AT A TIME. Each TTSSpeakFrame is its own Cartesia
    request, the harness can release four segments within milliseconds, and the
    account's concurrency limit is 2 — the extra segments failed and were
    silent. The next segment is released when the previous one has finished
    generating (TTSStoppedFrame, seen by _PlayoutObserver), or after a timeout
    so a lost frame can never wedge the call.
    """

    SEGMENT_TIMEOUT_S = 8.0

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._done = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        from pipecat.frames.frames import TTSSpeakFrame
        from pipecat.processors.frame_processor import FrameDirection

        while True:
            text = await self._queue.get()
            self._done.clear()
            await self._processor.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
            try:
                await asyncio.wait_for(self._done.wait(), self.SEGMENT_TIMEOUT_S)
            except TimeoutError:
                log.warning("tts_segment_timeout", text=text[:60])
            if self._queue.empty():
                self._idle.set()

    async def speak(self, text: str, *, context_id: str) -> None:
        self._idle.clear()
        await self._queue.put(text)

    def segment_done(self) -> None:
        self._done.set()

    def busy(self) -> bool:
        return not self._idle.is_set()

    async def wait_idle(self, limit_s: float) -> None:
        try:
            await asyncio.wait_for(self._idle.wait(), limit_s)
        except TimeoutError:
            pass

    async def cancel(self, context_id: str) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
        self._done.set()
        self._idle.set()
        await self._processor.broadcast_interruption()

    def close(self) -> None:
        self._worker.cancel()


def _raw_pcm_serializer(session: _Session) -> Any:
    """The browser's wire format: binary PCM16 frames; text frames are JSON
    control messages that belong to the app, not to Pipecat."""
    from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
    from pipecat.serializers.base_serializer import FrameSerializer

    class _RawPCM(FrameSerializer):
        async def serialize(self, frame: Any) -> bytes | None:
            if isinstance(frame, OutputAudioRawFrame):
                session.on_audio_out(len(frame.audio))
                return bytes(frame.audio)
            return None

        async def deserialize(self, data: str | bytes) -> Any:
            if isinstance(data, (bytes, bytearray)):
                return InputAudioRawFrame(audio=bytes(data), sample_rate=INPUT_SAMPLE_RATE,
                                          num_channels=1)
            return None  # {"type": "start"} and other control text

    return _RawPCM()


def _PlayoutObserver(session: _Session) -> Any:
    """Sits after TTS, just before the output transport.

    Downstream, a TTSStoppedFrame means one segment's audio is fully produced;
    segments are spoken in order, so they drain FIFO. Upstream, the transport
    reports Bot{Started,Stopped}SpeakingFrame — the only reliable sign that
    audio has finished PLAYING, because the transport paces audio in real
    time and most of a long line is still unsent when generation ends."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        ErrorFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import FrameProcessor

    class _Observer(FrameProcessor):
        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSStoppedFrame):
                session.on_tts_segment_done()
            elif isinstance(frame, BotStartedSpeakingFrame):
                session.on_bot_speaking(True)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                session.on_bot_speaking(False)
            elif isinstance(frame, ErrorFrame):
                log.warning("tts_error", detail=str(getattr(frame, "error", frame))[:200])
                session.on_tts_segment_done()  # never wait on a segment that failed
            await self.push_frame(frame, direction)

    return _Observer()


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
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class _Processor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
                session.on_user_started_speaking()
            elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
                session.on_user_stopped_speaking()
            elif isinstance(frame, TranscriptionFrame):
                session.on_final_transcript(
                    frame.text, confidence=1.0, words=[], provider="deepgram"
                )

            # Audio frames flow on untouched; the harness produces speech via
            # Speak commands, not by passing frames through.
            await self.push_frame(frame, direction)

    return _Processor()
