"""Run the harness with typed input and no audio.

This is the harness's primary test surface, not an afterthought. Because
`harness/` cannot import pipecat, the whole conversational system runs here
deterministically in milliseconds — which is what lets the interruption and
idempotency suites run on every commit instead of only before a demo.

Three ways to use it:

    make chat                                     interactive REPL
    make script FILE=tests/scripts/basic.yaml     scripted, with assertions
    TextDriver(...)                                from a test

Script format:

    llm_fixture: tests/fixtures/llm/two_sentences.yaml
    turns:
      - user: "hello"
      - user: "what about parking?"
        interrupt_at_ms: 150      # barge-in 150ms into the agent's reply
        played_chars: 20          # what the transport says the caller heard
    expect:
      stop_reasons: [answered, interrupted]
      context_invariant: ok
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from analytics.events import EventWriter, NullEventWriter
from analytics.store import CallStore, NullStore
from core.ids import new_id
from core.logging import configure_logging
from core.settings import Settings, get_settings
from harness import context as ctx
from harness.events import (
    BargeIn,
    Emit,
    EndCall,
    EventStream,
    OutCommand,
    Shutdown,
    Speak,
    TranscriptIn,
    TurnCompleted,
)
from harness.loop import Harness
from harness.retriever import Retriever
from harness.state import CallState, TurnState
from harness.tools.execution import ToolExecutor
from providers import build_llm, config_snapshot, effective_model, offline_fallback
from providers.llm.base import LLMProvider


@dataclass
class TurnResult:
    """Everything one `say()` produced."""

    turn: TurnState | None
    spoken: list[str] = field(default_factory=list)
    commands: list[OutCommand] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    ended: bool = False

    @property
    def stop_reason(self) -> str | None:
        return self.turn.stop_reason if self.turn else None

    @property
    def text(self) -> str:
        return " ".join(self.spoken)


class TextDriver:
    """Drives a Harness with typed utterances."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings | None = None,
        domain: str | None = None,
        source: str = "text",
        retriever: Retriever | None = None,
        store: CallStore | None = None,
        writer: EventWriter | None = None,
        executor: ToolExecutor | None = None,
        similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.harness = Harness(
            llm=llm, settings=self.settings, domain=domain, retriever=retriever,
            executor=executor, similarity=similarity,
        )
        self.call = CallState(
            call_id=new_id(),
            domain_config=domain or self.settings.domain,
            source=source,  # type: ignore[arg-type]
            config_snapshot=config_snapshot(self.settings),
        )
        # In-memory by default so tests never need a database. The CLI passes
        # Postgres-backed ones when the database is up, which is what makes a
        # `make chat` session visible in the live console.
        self.store: Any = store or NullStore()
        self.writer: Any = writer or NullEventWriter()
        self._stream = EventStream()
        self._out: asyncio.Queue[OutCommand] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ended = False
        self._started: float | None = None

    # -- lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> TextDriver:
        self._started = asyncio.get_running_loop().time()
        await self.store.create_call(self.call)
        self._task = asyncio.create_task(self._pump())
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _pump(self) -> None:
        """Consume the harness's command stream and persist as it goes."""
        async for command in self.harness.run(self.call, self._stream):
            if isinstance(command, Emit):
                await self.writer.emit(
                    self.call.call_id, command.event_type, command.payload, command.turn_id
                )
            elif isinstance(command, TurnCompleted):
                await self.store.save_turn(self.call, command.turn)
                # Flush per turn, not per call: the console tails call_events,
                # and a turn should appear there as soon as it finishes.
                await self.writer.flush()
            await self._out.put(command)
        await self.writer.flush()

    async def close(self) -> None:
        if self._ended or self._task is None:
            return
        self._stream.put(Shutdown("closed"))
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except TimeoutError:  # pragma: no cover
            self._task.cancel()
        self._ended = True
        await self._finish()

    async def _finish(self) -> None:
        started = self._started or asyncio.get_running_loop().time()
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        await self.store.finish_call(self.call, duration_ms)

    # -- driving ------------------------------------------------------------
    async def say(
        self,
        text: str,
        *,
        confidence: float = 1.0,
        word_confidences: list[tuple[str, float]] | None = None,
        endpoint_ms: int | None = None,
        interrupt_at_ms: int | None = None,
        played_chars: int | None = None,
        timeout: float = 15.0,
    ) -> TurnResult:
        """Speak one utterance and wait for the turn to finish.

        `interrupt_at_ms` schedules a barge-in that many milliseconds after the
        utterance lands — the deterministic equivalent of a caller cutting the
        agent off mid-sentence.
        """
        self._stream.put(
            TranscriptIn(
                text=text, confidence=confidence, provider="text", endpoint_ms=endpoint_ms,
                word_confidences=word_confidences or [],
            )
        )
        if interrupt_at_ms is not None:
            asyncio.get_running_loop().call_later(
                interrupt_at_ms / 1000,
                self._stream.put,
                BargeIn(played_chars=played_chars),
            )
        return await self._collect(timeout)

    async def barge_in(self, played_chars: int | None) -> TurnResult:
        """The caller cuts in while the LAST turn's audio is still playing.

        In live audio this is the common barge-in: generation finished, TTS
        did not. `played_chars` is what the transport reports was delivered.
        """
        return await self.send(BargeIn(played_chars=played_chars))

    async def send(self, event: Any, *, quiet_for: float = 0.15) -> TurnResult:
        """Deliver a raw event and collect whatever it produces.

        Unlike `say()`, this does not wait for a TurnCompleted — some events
        (SilenceTimeout, a stray BargeIn) never produce a turn. It collects
        until the command stream goes quiet for `quiet_for` seconds.
        """
        self._stream.put(event)
        result = TurnResult(turn=None)
        while True:
            try:
                command = await asyncio.wait_for(self._out.get(), timeout=quiet_for)
            except TimeoutError:
                return result
            result.commands.append(command)
            if isinstance(command, Speak):
                result.spoken.append(command.text)
            elif isinstance(command, Emit):
                result.events.append(command.event_type)
            elif isinstance(command, TurnCompleted):
                result.turn = command.turn
            elif isinstance(command, EndCall):
                result.ended = True
                self._ended = True
                await self._finish()
                return result

    async def _collect(self, timeout: float) -> TurnResult:
        result = TurnResult(turn=None)
        deadline = asyncio.get_running_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("turn did not complete within the timeout")
            command = await asyncio.wait_for(self._out.get(), timeout=remaining)
            result.commands.append(command)

            if isinstance(command, Speak):
                result.spoken.append(command.text)
            elif isinstance(command, Emit):
                result.events.append(command.event_type)
            elif isinstance(command, TurnCompleted):
                result.turn = command.turn
                return result
            elif isinstance(command, EndCall):
                result.ended = True
                self._ended = True
                await self._finish()
                return result

    def assert_invariant(self) -> None:
        ctx.assert_invariant(self.call.messages)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
async def open_recording(settings: Settings, *, enabled: bool) -> tuple[Any, Any, str]:
    """Postgres-backed store and writer if the database is up, else in-memory.

    Recording is what makes a text session show up in the live console and in
    the week-4 analytics. Falling back silently would leave you wondering why
    the console is empty, so the banner always says which it is.
    """
    from analytics.events import PostgresEventWriter
    from analytics.store import PostgresStore
    from core import db

    if not enabled:
        return None, None, "recording off"
    if not db.is_reachable(settings.database_url):
        return None, None, "NOT RECORDING (Postgres unreachable — run `make up`)"
    await db.open_pool(settings)
    return PostgresStore(), PostgresEventWriter(), "recording to Postgres"


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------
async def run_script(path: Path, *, record: bool = True, live: bool = False) -> int:
    """Run a scripted conversation and check its expectations.

        llm_fixture: tests/fixtures/llm/x.yaml
        stack:
          retrieval: real            # real | none | [list of chunk texts]
          tools: true
          verified: true             # start the call already verified
          fault_profile: ambiguous   # harness/tools/mocks/faults.py
        turns:
          - user: "send two tickets to sam@example.com"
          - barge_in: 15             # cut in while that reply is still playing
          - user: "yes"
            interrupt_at_ms: 120     # or: cut in mid-generation
            played_chars: 20
        expect:
          stop_reasons: [answered, interrupted, answered]
          outcome: contained
          effects: {transfer_tickets: 1}
          context_invariant: ok
    """
    from core import db
    from harness.assembly import build_stack

    spec = yaml.safe_load(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    settings = get_settings()
    domain = spec.get("domain") or settings.domain
    stack_spec = spec.get("stack", {})

    # --live ignores the fixture and runs the configured model, so the same
    # beats can be checked against a real LLM. A fixture-backed run says
    # nothing about any model; it tests the harness.
    fixture = None if live else spec.get("llm_fixture")
    if live:
        settings = offline_fallback(settings)
        print(f"  [live model: {settings.llm_provider} · {effective_model(settings)}]")
    llm = (
        build_llm(settings.model_copy(update={"llm_provider": "replay"}), fixture=fixture)
        if fixture
        else build_llm(settings)
    )
    store, writer, banner = await open_recording(settings, enabled=record)
    stack = await build_stack(
        settings,
        domain=domain,
        retrieval=stack_spec.get("retrieval", "none"),
        tools=bool(stack_spec.get("tools", False)),
        persist_ledger=store is not None,
        fault_profile=stack_spec.get("fault_profile"),
    )
    print(f"  [{banner} · {' · '.join(stack.banner)}]")

    failures: list[str] = []
    stop_reasons: list[str | None] = []

    async with TextDriver(
        llm=llm, settings=settings, domain=domain, store=store, writer=writer,
        retriever=stack.retriever, executor=stack.executor, similarity=stack.similarity,
    ) as driver:
        if stack_spec.get("verified"):
            driver.call.caller_verified = True
            driver.call.order_id = stack_spec.get("order_id", "TK-88213")

        for step in spec["turns"]:
            if "barge_in" in step:
                result = await driver.barge_in(step["barge_in"])
                print(f"  ⟂ caller cuts in ({step['barge_in']} chars heard)")
                if driver.call.turns:
                    stop_reasons[-1] = driver.call.turns[-1].stop_reason
                continue
            result = await driver.say(
                step["user"],
                interrupt_at_ms=step.get("interrupt_at_ms"),
                played_chars=step.get("played_chars"),
            )
            stop_reasons.append(result.stop_reason)
            print(f"  user  > {step['user']}")
            for line in result.spoken:
                print(f"  agent > {line}")
            print(f"  [stop_reason={result.stop_reason}]")
            if result.ended or driver.call.outcome in ("handoff", "error"):
                break

        expect = spec.get("expect", {})
        if "stop_reasons" in expect and stop_reasons != expect["stop_reasons"]:
            failures.append(f"stop_reasons: expected {expect['stop_reasons']}, got {stop_reasons}")
        if expect.get("context_invariant") == "ok":
            try:
                driver.assert_invariant()
            except AssertionError as exc:
                failures.append(f"context invariant: {exc}")
        for tool, count in (expect.get("effects") or {}).items():
            actual = stack.backend.count_effects(tool) if stack.backend else 0
            if actual != count:
                failures.append(f"effects[{tool}]: expected {count}, got {actual}")

    outcome = driver.call.outcome
    if "outcome" in expect and outcome != expect["outcome"]:
        failures.append(f"outcome: expected {expect['outcome']}, got {outcome}")
    if driver.call.handoff:
        print("\n" + "\n".join("  | " + line for line in driver.call.handoff["summary_text"].splitlines()))

    await db.close_pool()
    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nOK  (call {driver.call.call_id}, outcome {outcome})")
    return 0


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
async def run_interactive(
    domain: str | None, *, use_retrieval: bool = True, record: bool = True
) -> int:
    from core import db

    settings = get_settings()
    llm = build_llm(settings)
    domain = domain or settings.domain

    from harness.assembly import build_stack

    print(f"voicedesk text mode — domain={domain}, "
          f"llm={settings.llm_provider}:{settings.llm_model}")
    store, writer, record_banner = await open_recording(settings, enabled=record)
    stack = await build_stack(
        settings, domain=domain, retrieval="real" if use_retrieval else "none",
        tools=True, persist_ledger=store is not None,
    )
    print(" · ".join([record_banner, *stack.banner]))
    print("Ctrl-D or 'quit' to exit.\n")

    async with TextDriver(
        llm=llm, settings=settings, domain=domain, retriever=stack.retriever,
        store=store, writer=writer, executor=stack.executor, similarity=stack.similarity,
    ) as driver:
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, lambda: input("you > "))
            except EOFError:
                break
            if line.strip().lower() in {"quit", "exit"}:
                break
            if not line.strip():
                continue

            result = await driver.say(line)
            for segment in result.spoken:
                print(f"agent > {segment}")
            stages = result.turn.stages_ms if result.turn else {}
            print(f"        [{result.stop_reason} | {_format_stages(stages)}]\n")
            if result.ended:
                break
    await db.close_pool()
    return 0


def _format_stages(stages: dict[str, Any]) -> str:
    order = ["endpoint", "stt_final", "retrieval", "llm_ttft", "sentence_buffer", "first_audio"]
    parts = [f"{name}={stages[name]}ms" for name in order if name in stages]
    return " ".join(parts) or "no timings"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the harness with typed input.")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--no-retrieval", action="store_true", help="run ungrounded, without the KB"
    )
    parser.add_argument(
        "--no-record", action="store_true", help="do not write the call to Postgres"
    )
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument("--live", action="store_true",
                        help="with --script: ignore llm_fixture, use the configured model")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    if args.script:
        return asyncio.run(run_script(args.script, record=not args.no_record, live=args.live))
    return asyncio.run(
        run_interactive(
            args.domain, use_retrieval=not args.no_retrieval, record=not args.no_record
        )
    )


if __name__ == "__main__":
    sys.exit(main())
