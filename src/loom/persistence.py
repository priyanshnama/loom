"""Checkpointer factory for LangGraph state persistence.

Two tiers selected by the ``LOOM_PERSISTENCE`` env var:

  memory  — ``MemorySaver`` (default): in-process RAM only.
  sqlite  — ``AsyncSqliteSaver``: file-backed SQLite.

Usage
-----
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        ...
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from loom.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Async context manager that yields the appropriate checkpointer."""
    backend = settings.loom_persistence.lower()

    if backend == "sqlite":
        async with _sqlite_checkpointer() as cp:
            yield cp
    else:
        logger.info("persistence | backend=memory (process-local, not persistent)")
        yield MemorySaver()


@asynccontextmanager
async def _sqlite_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
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
        yield cp
