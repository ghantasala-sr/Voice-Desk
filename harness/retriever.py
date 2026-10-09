"""The harness's view of retrieval.

The harness takes a Retriever, it does not import a database. That keeps the
turn loop testable with no Postgres and no 2 GB embedding model — the same
reason it takes an LLMProvider rather than an Anthropic client.

`NullRetriever` is the week-1 behaviour: no knowledge, no grounding signals.
`PostgresRetriever` is the real one, built in app/ and in the text driver.
`FakeRetriever` lets a test hand the loop exact chunks and exact scores, which
is how checker and handoff behaviour get tested deterministically.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.settings import Settings
from retrieval.embed import Embedder
from retrieval.search import RetrievalResult, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    async def fetch(
        self, utterance: str, *, previous: str | None = None
    ) -> RetrievalResult: ...


def empty_result(query: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunks=[],
        top_score=0.0,
        margin=0.0,
        top_vec_score=0.0,
        top_lex_score=0.0,
        vec_margin=0.0,
        query_text=query,
        elapsed_ms=0,
    )


class NullRetriever:
    """No knowledge base. The agent can only use fixed template lines."""

    async def fetch(self, utterance: str, *, previous: str | None = None) -> RetrievalResult:
        return empty_result(utterance)


class FakeRetriever:
    """Returns a fixed result. For tests that need exact scores."""

    def __init__(self, result: RetrievalResult | None = None) -> None:
        self.result = result if result is not None else empty_result()
        self.calls: list[tuple[str, str | None]] = []

    @classmethod
    def with_chunks(
        cls,
        texts: list[str],
        *,
        heading: str = "Test doc > Section",
        top_vec_score: float = 0.80,
    ) -> FakeRetriever:
        chunks = [
            RetrievedChunk(
                chunk_id=f"c{index}",
                doc_id="d0",
                doc_path="test.md",
                heading_path=heading,
                text=text,
                fused_score=0.032 - index * 0.001,
                lex_score=0.8,
                vec_score=top_vec_score - index * 0.05,
                lex_rank=index + 1,
                vec_rank=index + 1,
            )
            for index, text in enumerate(texts)
        ]
        return cls(
            RetrievalResult(
                chunks=chunks,
                top_score=chunks[0].fused_score if chunks else 0.0,
                margin=0.001,
                top_vec_score=top_vec_score if chunks else 0.0,
                top_lex_score=0.8 if chunks else 0.0,
                vec_margin=0.05,
                query_text="",
                elapsed_ms=1,
            )
        )

    async def fetch(self, utterance: str, *, previous: str | None = None) -> RetrievalResult:
        self.calls.append((utterance, previous))
        return self.result


class PostgresRetriever:
    """The real one: hybrid search over the ingested corpus."""

    def __init__(self, *, domain: str, embedder: Embedder, settings: Settings) -> None:
        self._domain = domain
        self._embedder = embedder
        self._settings = settings

    async def fetch(self, utterance: str, *, previous: str | None = None) -> RetrievalResult:
        from retrieval.search import search

        return await search(
            utterance,
            domain=self._domain,
            embedder=self._embedder,
            settings=self._settings,
            previous_utterance=previous,
        )
