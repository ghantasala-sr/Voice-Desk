"""Postgres access: one async connection pool, opened at app start.

No ORM. The schema is small and the week-2 retrieval query is a hand-written
CTE doing reciprocal rank fusion — the SQL is the interesting part, and an ORM
would put indirection over it (docs/02 §2).

psycopg is imported lazily so that harness unit tests, which never touch the
database, run without it installed.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from core.settings import Settings

if TYPE_CHECKING:  # pragma: no cover
    from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None


async def open_pool(settings: Settings) -> AsyncConnectionPool:
    """Open the process-wide pool. Idempotent."""
    global _pool
    if _pool is not None:
        return _pool

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    )
    await _pool.open(wait=True, timeout=5)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("database pool is not open — call open_pool() at startup")
    return _pool


async def execute(sql: str, params: dict[str, Any] | None = None) -> None:
    async with pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params or {})


async def fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    async with pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params or {})
        return await cur.fetchone()


async def fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    async with pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params or {})
        return await cur.fetchall()


@functools.lru_cache(maxsize=4)
def is_reachable(database_url: str, *, timeout: int = 1) -> bool:
    """Is OUR database actually usable? Used to skip database-marked tests.

    A real connect, not a socket probe: a port can be open while the database,
    role, or password is wrong, and a test that is going to fail on auth
    should be skipped rather than run. Cached, because it is called at test
    collection time.
    """
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=timeout) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
