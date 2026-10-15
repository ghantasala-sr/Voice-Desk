"""W4-1 — drive a synthetic caller population through the REAL harness.

Each simulated call is a persona with a goal, talking to the production stack:
real hybrid retrieval, the grounding checker, the tool executor and
idempotency ledger, read-back, handoff. Every call is recorded to Postgres
exactly like a real one (source='simulation'), so the gap report cannot tell
the difference — and must not need to.

The CALLER is rule-driven from personas.yaml and intents.yaml: it opens,
gives details when asked, says yes to a correct read-back, rephrases or gives
up when an answer does not help, sometimes cuts in, and asks for a person
when its patience runs out. "Did the answer help?" is judged from the hidden
answer key: the cited passage came from one of the intent's gold documents,
or the transaction actually completed.

The AGENT is whatever LLM provider is configured. Without an API key that is
the offline extractive provider (providers/llm/extractive.py), so a run
measures the machinery around the model, not the model's answers. Every
result says which agent produced it.

The answer key for each call goes into a `simulation_label` event. The gap
report never reads that event; it is used only to SCORE the gap report.

Run: .venv/bin/python -m eval.caller_sim.simulate --n 300 --label baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from analytics.events import PostgresEventWriter
from analytics.store import PostgresStore
from core import db
from core.ids import new_id
from core.settings import Settings, get_settings
from eval.metrics import Proportion
from eval.text_driver import TextDriver
from harness.retriever import PostgresRetriever
from harness.tools.execution import PostgresLedger, ToolExecutor
from harness.tools.mocks.faults import PROFILES, FaultyBackend
from harness.tools.mocks.ticketing import MockTicketing
from harness.tools.registry import load_registry
from providers import build_llm, offline_fallback
from retrieval.embed import get_embedder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_TURNS = 6

READBACK = "Should I go ahead"
ASKS_DETAILS = ("order number", "email address on the order", "read it to me again")
EMAIL_MISMATCH = "doesn't match the order"
SIGNOFF = "Thanks, that's everything."


# ---------------------------------------------------------------------------
# Spoken forms, as a caller says them
# ---------------------------------------------------------------------------
def spoken_order(order_id: str, *, misstate: bool = False) -> str:
    digits = list(order_id.split("-")[1])
    if misstate:
        digits[-1], digits[-2] = digits[-2], digits[-1]  # transposes two digits
    return "T K " + " ".join(digits)


def spoken_email(email: str) -> str:
    local, domain = email.split("@")
    local = local.replace(".", " dot ").replace("+", " plus ")
    return f"{local} at {domain.replace('.', ' dot ')}"


# ---------------------------------------------------------------------------
# The caller
# ---------------------------------------------------------------------------
@dataclass
class Caller:
    persona: dict[str, Any]
    intent: dict[str, Any]
    rng: random.Random
    misstated: bool = False
    unhelpful: int = 0
    used: set[str] = field(default_factory=set)

    def opening(self) -> str:
        pool = self.intent["openings"]
        if self.intent.get("vague") and self.rng.random() < self.persona["vague"]:
            pool = self.intent["vague"]
        text = self._pick(pool)
        if self.persona["id"] == "detailed":
            text = f"Hi there, I bought tickets a while ago and I've got a question. {text}"
        if self.intent.get("transactional") and self.rng.random() < self.persona["details_upfront"]:
            text = f"{text} {self.details()}"
        return text

    def details(self) -> str:
        order = self.intent["order"]
        misstate = bool(self.persona.get("misstate_once")) and not self.misstated
        self.misstated = self.misstated or misstate
        return (f"It's order {spoken_order(order['order_id'], misstate=misstate)}, "
                f"{spoken_email(order['email'])}.")

    def rephrase(self) -> str | None:
        fresh = [u for u in self.intent["openings"] if u not in self.used]
        return self._pick(fresh) if fresh else None

    def wants_human(self) -> bool:
        return self.unhelpful >= self.persona["human_after"]

    def _pick(self, pool: list[str]) -> str:
        choice = self.rng.choice(pool)
        self.used.add(choice)
        return choice


def helped(intent: dict[str, Any], cited_docs: set[str], backend: MockTicketing,
           reply: str) -> bool:
    """The HIDDEN judgment: did this turn actually solve the caller's problem?"""
    kind = intent.get("transactional")
    if kind == "transfer":
        return backend.count_effects("transfer_tickets") > 0
    if kind == "lookup":
        return any(s in reply for s in ("You have", "has been cancelled", "has been postponed"))
    return bool(cited_docs & set(intent.get("gold_docs", [])))


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------
async def simulate_call(index: int, *, run_id: str, seed: int, personas: list[dict[str, Any]],
                        intents: list[dict[str, Any]], settings: Settings,
                        retriever: PostgresRetriever, similarity: Any,
                        chunk_docs: dict[str, str]) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{index}")
    persona = rng.choices(personas, weights=[p["weight"] for p in personas])[0]
    intent = rng.choices(intents, weights=[i["weight"] for i in intents])[0]
    caller = Caller(persona, intent, rng)

    backend = MockTicketing()
    executor = ToolExecutor(registry=load_registry(settings.domain),
                            backend=FaultyBackend(backend, PROFILES["benign"]),
                            ledger=PostgresLedger())
    writer = PostgresEventWriter()
    driver = TextDriver(llm=build_llm(settings), settings=settings, source="simulation",
                        store=PostgresStore(), writer=writer, retriever=retriever,
                        executor=executor, similarity=similarity)
    driver.call.config_snapshot["simulation_run"] = run_id
    driver.call.config_snapshot["agent"] = settings.llm_provider

    utterance: str | None = caller.opening()
    satisfied = False
    interruptions = 0
    turns = 0
    async with driver:
        while utterance is not None and turns < MAX_TURNS:
            turns += 1
            result = await driver.say(utterance)
            if result.ended or driver.call.outcome in ("handoff", "error"):
                break
            reply = " ".join(result.spoken)
            turn = result.turn
            cited = {chunk_docs.get(c, "") for c in (turn.cited_chunk_ids if turn else [])}

            # A caller whose problem is solved has no reason to cut in.
            if READBACK not in reply and helped(intent, cited, backend, reply):
                satisfied = True
                break

            # Cutting in while the reply is still playing.
            if rng.random() < persona["interrupts"] and turn and turn.approved_text:
                interruptions += 1
                await driver.barge_in(played_chars=rng.randint(5, max(6, len(turn.approved_text) // 2)))
                if READBACK in reply:
                    # A read-back talked over is a NO; the caller asks again.
                    utterance = caller.opening() if intent.get("transactional") else caller.rephrase()
                    continue

            if READBACK in reply:
                utterance = "Yes, go ahead."
                continue
            if any(s in reply for s in ASKS_DETAILS) or EMAIL_MISMATCH in reply:
                utterance = caller.details() if intent.get("order") else caller.rephrase()
                continue
            caller.unhelpful += 1
            if caller.wants_human():
                utterance = "Can I talk to a real person, please?"
            elif rng.random() < persona["rephrase"]:
                utterance = caller.rephrase()
            else:
                utterance = None  # gives up and hangs up, unhelped

        label = {
            "persona": persona["id"], "intent": intent["id"], "covered": intent["covered"],
            "satisfied": satisfied, "interruptions": interruptions, "turns": turns,
            "seed": seed, "index": index, "run": run_id,
        }
        # The answer key. Never read by the gap report — only used to score it.
        await writer.emit(driver.call.call_id, "simulation_label", label)
        await writer.flush()

    return {**label, "call_id": driver.call.call_id, "outcome": driver.call.outcome,
            "handoff_reason": driver.call.handoff_reason,
            "effects": backend.count_effects()}


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["outcome"] != "error"]
    contained = [r for r in scored if r["outcome"] == "contained"]
    # Wrong-answer proxy: the call ended "contained" but the caller was NOT
    # helped — the agent answered, the answer did not solve the problem. The
    # denominator is contained calls (docs/11 §6): a handed-off call gave no
    # wrong answer.
    wrong = [r for r in contained if not r["satisfied"]]
    by_intent: dict[str, dict[str, Any]] = {}
    for intent in sorted({r["intent"] for r in rows}):
        sub = [r for r in scored if r["intent"] == intent]
        by_intent[intent] = {
            "n": len(sub),
            "contained": sum(r["outcome"] == "contained" for r in sub),
            "handoff": sum(r["outcome"] == "handoff" for r in sub),
            "satisfied": sum(r["satisfied"] for r in sub),
        }
    return {
        "n_calls": len(rows),
        "containment": Proportion(len(contained), len(scored)).as_dict(),
        "handoff_rate": Proportion(sum(r["outcome"] == "handoff" for r in scored), len(scored)).as_dict(),
        "wrong_answer_rate": Proportion(len(wrong), len(contained)).as_dict(),
        "resolved_rate": Proportion(sum(r["satisfied"] for r in scored), len(scored)).as_dict(),
        "interruptions": sum(r["interruptions"] for r in rows),
        "by_intent": by_intent,
        "by_persona": {p: sum(1 for r in rows if r["persona"] == p) for p in sorted({r["persona"] for r in rows})},
    }


async def run(n: int, label: str, seed: int, settings: Settings) -> dict[str, Any]:
    personas = yaml.safe_load((HERE / "personas.yaml").read_text())["personas"]
    intents = yaml.safe_load((HERE / "intents.yaml").read_text())["intents"]
    await db.open_pool(settings)
    embedder = get_embedder(settings)
    retriever = PostgresRetriever(domain=settings.domain, embedder=embedder, settings=settings)

    def similarity(a: str, b: str) -> float:
        left, right = embedder.embed_passages([a, b])
        return sum(x * y for x, y in zip(left, right, strict=True))

    chunk_rows = await db.fetch_all(
        "SELECT c.chunk_id, d.path FROM chunks c JOIN documents d USING (doc_id)")
    chunk_docs = {r["chunk_id"]: r["path"] for r in chunk_rows}
    corpus = await db.fetch_all(
        "SELECT path FROM documents WHERE ingested ORDER BY path")
    run_id = f"{label}-{new_id()[:8]}"

    rows = []
    for index in range(n):
        rows.append(await simulate_call(index, run_id=run_id, seed=seed, personas=personas,
                                        intents=intents, settings=settings, retriever=retriever,
                                        similarity=similarity, chunk_docs=chunk_docs))
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{n}", flush=True)

    duplicates = await db.fetch_all(
        "SELECT idempotency_key FROM tool_executions WHERE side_effecting AND status='committed'"
        " GROUP BY 1 HAVING count(*) > 1")
    await db.close_pool()
    return {"run": run_id, "label": label, "seed": seed, "agent": settings.llm_provider,
            "ingested_documents": [r["path"] for r in corpus],
            "duplicate_side_effects": len(duplicates),
            **summarize(rows), "_rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate a caller population.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    # No key: the offline agent, and every result says so.
    settings = offline_fallback(get_settings())
    result = asyncio.run(run(args.n, args.label, args.seed, settings))

    out = ROOT / "results" / f"{date.today().isoformat()}-simulation-{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    rows = result.pop("_rows")
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    c, h, w = result["containment"], result["handoff_rate"], result["wrong_answer_rate"]
    print(f"\nrun {result['run']}  agent={result['agent']}  calls={result['n_calls']}")
    print(f"  containment  {c['value']:.3f} {c['ci95']}  ({c['successes']}/{c['n']})")
    print(f"  handoff      {h['value']:.3f}")
    print(f"  wrong answer {w['value']:.3f} {w['ci95']}  (of contained)")
    print(f"  interruptions {result['interruptions']}   duplicate side effects {result['duplicate_side_effects']}")
    for intent, s in result["by_intent"].items():
        print(f"    {intent:22} n={s['n']:3}  contained={s['contained']:3}  handoff={s['handoff']:3}  helped={s['satisfied']:3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
