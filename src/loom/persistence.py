"""Checkpointer factory for LangGraph state persistence.

Three tiers selected by the ``LOOM_PERSISTENCE`` env var:

  memory   — ``MemorySaver`` (default): in-process RAM only.
             Checkpoints are lost when the process exits.
             Good for unit tests and quick experiments.

  sqlite   — ``SqliteSaver``: file-backed SQLite database.
             Persists across CLI invocations with zero infrastructure.
             Good for local development and demos.
             Set ``SQLITE_PATH`` to control the database file location.

  postgres — ``AsyncPostgresSaver``: production-grade Postgres pool.
             Set ``POSTGRES_DSN`` (or individual POSTGRES_* vars).

Usage
-----
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        ...
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from loom.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """Async context manager that yields the appropriate checkpointer.

    The backend is chosen from ``settings.loom_persistence``:
    ``"memory"`` | ``"sqlite"`` | ``"postgres"``.
    The legacy ``LOOM_USE_POSTGRES=true`` flag is also honoured.
    """
    backend = settings.loom_persistence.lower()

    if backend == "postgres":
        async with _postgres_checkpointer() as cp:
            yield cp
    elif backend == "sqlite":
        async with _sqlite_checkpointer() as cp:
            yield cp
    else:
        logger.info("persistence | backend=memory (process-local, not persistent)")
        yield MemorySaver()


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _sqlite_checkpointer() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """Yield a ``SqliteSaver`` backed by a local SQLite file."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Install 'langgraph-checkpoint-sqlite' to use SQLite persistence."
        ) from exc

    path = settings.sqlite_path
    logger.info("persistence | backend=sqlite | path=%s", path)

    async with AsyncSqliteSaver.from_conn_string(path) as cp:
        await cp.setup()
        logger.info("persistence | SQLite checkpointer ready")
        yield cp


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _postgres_checkpointer() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """Yield an ``AsyncPostgresSaver`` connected to Postgres.

    Uses a manually-configured ``AsyncConnectionPool`` so we can set
    ``max_idle`` to recycle connections before Neon's serverless compute
    auto-suspends them (default: 5 min = 300 s).  Without this, the pool
    holds stale connections that Neon has already killed, causing
    "the connection is closed" / "terminating connection due to
    administrator command" errors on the next request.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415
        from psycopg_pool import AsyncConnectionPool  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Install 'langgraph-checkpoint-postgres' and 'psycopg[binary,pool]' "
            "to use Postgres persistence."
        ) from exc

    dsn = settings.effective_postgres_dsn
    logger.info("persistence | backend=postgres | dsn=%s", _redact_dsn(dsn))

    # min_size=0: no persistent idle connections — nothing for Neon to kill.
    # max_idle=240 s: close connections idle > 4 min (before Neon's 5-min suspend).
    # check: ping each connection before use so stale ones are replaced immediately.
    # prepare_threshold=0: disable prepared statements (breaks on recycled connections).
    async def _check(conn) -> None:  # type: ignore[type-arg]
        await conn.execute("SELECT 1")

    async with AsyncConnectionPool(
        dsn,
        min_size=0,
        max_size=10,
        max_idle=240,
        check=_check,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    ) as pool:
        await pool.open(wait=True)
        cp = AsyncPostgresSaver(pool)
        await cp.setup()
        logger.info("persistence | Postgres checkpointer ready")
        yield cp


def _redact_dsn(dsn: str) -> str:
    return re.sub(r"(?<=:)[^:@]+(?=@)", "***", dsn)
