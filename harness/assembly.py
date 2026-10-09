"""Assemble the production stack: retriever, tool executor, similarity.

One factory, used by the text driver, the scripts, and the live pipeline, so
the three cannot drift into different configurations. Every degradation is
reported in `banner` rather than happening silently — an agent that is
quietly ungrounded, or quietly not recording its tool calls, is worse than one
that refuses to start.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core import db
from core.settings import Settings
from harness.retriever import FakeRetriever, PostgresRetriever, Retriever
from harness.tools.execution import InMemoryLedger, Ledger, PostgresLedger, ToolExecutor
from harness.tools.mocks.faults import PROFILES, FaultyBackend
from harness.tools.mocks.ticketing import MockTicketing
from harness.tools.registry import load_registry


@dataclass
class Stack:
    retriever: Retriever | None = None
    executor: ToolExecutor | None = None
    similarity: Callable[[str, str], float] | None = None
    backend: MockTicketing | None = None
    banner: list[str] = field(default_factory=list)


async def build_stack(
    settings: Settings,
    *,
    domain: str,
    retrieval: str | list[str] = "real",
    tools: bool = True,
    persist_ledger: bool = False,
    fault_profile: str | None = None,
) -> Stack:
    """`retrieval` is "real", "none", or a list of chunk texts for a script."""
    stack = Stack()
    embedder = None

    if isinstance(retrieval, list):
        stack.retriever = FakeRetriever.with_chunks(retrieval, heading="Script knowledge")
        stack.banner.append(f"retrieval: {len(retrieval)} scripted chunks")
    elif retrieval == "real":
        if not db.is_reachable(settings.database_url):
            stack.banner.append("NO RETRIEVAL (Postgres unreachable — run `make up && make ingest`)")
        else:
            await db.open_pool(settings)
            row = await db.fetch_one(
                "SELECT count(*) AS n FROM chunks WHERE domain_config = %(d)s", {"d": domain}
            )
            if not row or not row["n"]:
                stack.banner.append("NO RETRIEVAL (no chunks ingested — run `make ingest`)")
            else:
                from retrieval.embed import get_embedder

                embedder = get_embedder(settings)
                stack.retriever = PostgresRetriever(
                    domain=domain, embedder=embedder, settings=settings
                )
                stack.banner.append(f"retrieval: {row['n']} chunks")
    else:
        stack.banner.append("retrieval off")

    if embedder is not None:
        def similarity(a: str, b: str) -> float:
            left, right = embedder.embed_passages([a, b])
            return sum(x * y for x, y in zip(left, right, strict=True))  # unit vectors

        stack.similarity = similarity
    else:
        stack.banner.append("rephrase signal off (no embedder)")

    if tools:
        profile = PROFILES[fault_profile or settings.tool_fault_profile]
        ledger: Ledger
        if persist_ledger and db.is_reachable(settings.database_url):
            await db.open_pool(settings)
            ledger = PostgresLedger()
            where = "Postgres"
        else:
            ledger = InMemoryLedger()
            where = "memory"
        stack.backend = MockTicketing()
        stack.executor = ToolExecutor(
            registry=load_registry(domain),
            backend=FaultyBackend(stack.backend, profile),
            ledger=ledger,
        )
        stack.banner.append(f"tools: mock backend, profile={profile.name}, ledger in {where}")
    return stack
