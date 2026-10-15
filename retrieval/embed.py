"""Embeddings.

Two details here are easy to get wrong and expensive to debug (docs/05 §3):

  * ASYMMETRIC PREFIXING. BGE expects queries to carry a prefix and passages
    not to. Getting it backwards, or applying it to both, measurably degrades
    retrieval while looking like it works. `embed_query` and `embed_passage`
    are separate functions precisely so the asymmetry cannot be accidentally
    collapsed into one call.
  * UNIT NORMALIZATION at ingest, so cosine distance is a dot product and
    scores are comparable across queries.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from core.settings import Settings

#: Tied to BAAI/bge-small-en-v1.5 and to `vector(384)` in 002_corpus.sql.
#: Changing the model is a migration, not a config flip — ingest.py asserts
#: this matches the column width and refuses to write wrong-width vectors.
EMBED_DIM = 384

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_query(self, text: str) -> list[float]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """The real one. Needs `make retrieval` (sentence-transformers + torch)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        # The accessor was renamed; support both so a version bump is not a
        # crash at ingest time.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(getter())

    def embed_query(self, text: str) -> list[float]:
        # Prefix: queries only.
        vector = self._model.encode(
            BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # No prefix: passages only.
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [[float(x) for x in vector] for vector in vectors]


class HashEmbedder:
    """A deterministic stand-in with NO semantic meaning.

    For tests and for exercising the SQL without a 2 GB model download. Dense
    recall under this embedder is effectively random — which is fine, because
    what it is testing is the plumbing: dimensions, batching, the RRF
    arithmetic, and the margin computation.

    It must never produce a number that reaches `results/`. `ingest.py` refuses
    to write a corpus hash when this is active, so a run embedded with it
    cannot be mistaken for a real one.
    """

    name = "hash-test-only"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        # Hash each token into a bucket; a bag-of-words sketch. Shares the
        # lexical signal only, which is enough to keep tests meaningful.
        buckets = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            buckets[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
        return [x / norm for x in buckets]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(BGE_QUERY_PREFIX + text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


_EMBEDDERS: dict[str, Embedder] = {}


def get_embedder(settings: Settings) -> Embedder:
    """The real embedder, or the hash stand-in if it is not installed.

    One instance per model name for the life of the process: loading BGE takes
    a second or more and blocks the event loop, and the web app builds a stack
    per chat connection."""
    if settings.embed_model == "hash-test-only":
        return HashEmbedder()
    if settings.embed_model in _EMBEDDERS:
        return _EMBEDDERS[settings.embed_model]
    try:
        embedder = _EMBEDDERS[settings.embed_model] = SentenceTransformerEmbedder(settings.embed_model)
        return embedder
    except ImportError as exc:
        raise RuntimeError(
            f"{settings.embed_model} needs sentence-transformers. Run `make retrieval`, "
            "or set VD_EMBED_MODEL=hash-test-only to exercise the plumbing only."
        ) from exc


def is_real(embedder: Embedder) -> bool:
    return not isinstance(embedder, HashEmbedder)
