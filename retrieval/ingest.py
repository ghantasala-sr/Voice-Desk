"""Corpus -> chunks -> embeddings -> Postgres.

Idempotent and safe to re-run: documents are hashed, and a document whose
content has not changed is skipped entirely.

ALL documents are recorded, including held-out ones with ingested = FALSE and
no chunks (docs/00 D4). That distinction is what lets the week-4 gap report say
"a document exists for this topic but is not indexed", which is a different and
much more actionable finding than "no document covers this topic".

Run: make ingest
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import db
from core.ids import new_id
from core.logging import configure_logging, get_logger
from core.settings import Settings, get_settings
from retrieval.chunk import chunk_document
from retrieval.embed import EMBED_DIM, Embedder, get_embedder, is_real

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"
log = get_logger(__name__)


@dataclass
class IngestReport:
    documents_total: int
    documents_ingested: int
    documents_held_out: int
    documents_skipped: int
    chunks_written: int
    corpus_hash: str
    embedder: str

    def render(self) -> str:
        lines = [
            f"  documents   {self.documents_total} total"
            f" / {self.documents_ingested} ingested"
            f" / {self.documents_held_out} held out"
            f" / {self.documents_skipped} unchanged",
            f"  chunks      {self.chunks_written} written",
            f"  embedder    {self.embedder}",
            f"  corpus hash {self.corpus_hash}",
        ]
        return "\n".join(lines)


def load_manifest(domain: str) -> dict[str, Any]:
    path = DOMAINS_ROOT / domain / "corpus_manifest.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def corpus_hash(entries: list[tuple[str, str]]) -> str:
    """sha256 over the sorted (path, content_hash) of INGESTED documents only.

    This is how a before/after comparison proves the corpus actually changed
    (docs/14 §5). Held-out documents are excluded by construction: ingesting
    one must change the hash.
    """
    payload = json.dumps(sorted(entries), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def ingest(
    domain: str, *, settings: Settings, force: bool = False,
    overrides: dict[str, bool] | None = None,
) -> IngestReport:
    """`overrides` flips a document's `ingested` flag for this run only — how
    the week-4 feedback cycle applies its one change without editing the
    manifest that defines the documented baseline."""
    manifest = load_manifest(domain)
    for entry in manifest["documents"]:
        if overrides and entry["path"] in overrides:
            entry["ingested"] = overrides[entry["path"]]
    corpus_dir = DOMAINS_ROOT / domain / "corpus"
    embedder = get_embedder(settings)

    if embedder.dim != EMBED_DIM:
        raise RuntimeError(
            f"{embedder.name} produces {embedder.dim}-dim vectors but the `chunks.embedding` "
            f"column is vector({EMBED_DIM}). Changing the model is a MIGRATION, not a config "
            "flip — otherwise this writes silently-wrong vectors."
        )

    ingested = held_out = skipped = chunks_written = 0
    hash_entries: list[tuple[str, str]] = []

    for entry in manifest["documents"]:
        path = corpus_dir / entry["path"]
        text = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_id = f"{domain}:{entry['path']}"

        existing = await db.fetch_one(
            "SELECT content_hash, ingested FROM documents"
            " WHERE domain_config = %(domain)s AND path = %(path)s",
            {"domain": domain, "path": entry["path"]},
        )

        await db.execute(
            """
            INSERT INTO documents
              (doc_id, domain_config, path, title, topics, content_hash, ingested, ingested_at)
            VALUES
              (%(doc_id)s, %(domain)s, %(path)s, %(title)s, %(topics)s,
               %(hash)s, %(ingested)s, CASE WHEN %(ingested)s THEN now() END)
            ON CONFLICT (domain_config, path) DO UPDATE SET
              title = EXCLUDED.title, topics = EXCLUDED.topics,
              content_hash = EXCLUDED.content_hash, ingested = EXCLUDED.ingested,
              ingested_at = EXCLUDED.ingested_at
            """,
            {
                "doc_id": doc_id, "domain": domain, "path": entry["path"],
                "title": entry["title"], "topics": entry.get("topics", []),
                "hash": content_hash, "ingested": entry["ingested"],
            },
        )

        if not entry["ingested"]:
            # Recorded, deliberately unindexed. Drop any chunks from a previous
            # run so toggling `ingested` back and forth stays consistent.
            await db.execute("DELETE FROM chunks WHERE doc_id = %(id)s", {"id": doc_id})
            held_out += 1
            continue

        ingested += 1
        hash_entries.append((entry["path"], content_hash))

        unchanged = (
            existing is not None
            and existing["content_hash"] == content_hash
            and existing["ingested"]
        )
        if unchanged and not force:
            skipped += 1
            continue

        chunks_written += await _write_chunks(
            doc_id=doc_id, domain=domain, title=entry["title"], text=text, embedder=embedder
        )

    digest = corpus_hash(hash_entries)
    if not is_real(embedder):
        # A corpus embedded with the hash stand-in must never be mistaken for
        # a real one in a results directory.
        digest = f"UNUSABLE-hash-embedder-{digest}"

    return IngestReport(
        documents_total=len(manifest["documents"]),
        documents_ingested=ingested,
        documents_held_out=held_out,
        documents_skipped=skipped,
        chunks_written=chunks_written,
        corpus_hash=digest,
        embedder=embedder.name,
    )


async def _write_chunks(
    *, doc_id: str, domain: str, title: str, text: str, embedder: Embedder
) -> int:
    chunks = chunk_document(text, title=title)
    vectors = embedder.embed_passages([c.embed_text for c in chunks])

    await db.execute("DELETE FROM chunks WHERE doc_id = %(id)s", {"id": doc_id})

    async with db.pool().connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO chunks
              (chunk_id, doc_id, domain_config, seq, heading_path, text,
               embed_text, word_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (
                    new_id(), doc_id, domain, chunk.seq, chunk.heading_path,
                    chunk.text, chunk.embed_text, chunk.word_count,
                    "[" + ",".join(f"{x:.6f}" for x in vector) + "]",
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return len(chunks)


async def _main(domain: str, force: bool) -> int:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    await db.open_pool(settings)
    try:
        report = await ingest(domain, settings=settings, force=force)
    finally:
        await db.close_pool()
    print(f"ingest: {domain}")
    print(report.render())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the knowledge base.")
    parser.add_argument("--domain", default="ticketing")
    parser.add_argument("--force", action="store_true", help="re-embed everything")
    args = parser.parse_args()
    return asyncio.run(_main(args.domain, args.force))


if __name__ == "__main__":
    sys.exit(main())
