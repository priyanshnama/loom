"""Session management utilities.

Provides ``reset_thread`` which deletes all LangGraph checkpoints stored for a
given ``thread_id``, effectively clearing the conversation history.

Supports all three persistence backends (memory, SQLite, Postgres) without the
caller needing to know which one is active.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# langgraph-checkpoint-sqlite uses: checkpoints, writes
# langgraph-checkpoint-postgres uses: checkpoints, checkpoint_blobs, checkpoint_writes
_SQLITE_TABLES = ("checkpoints", "writes")
_POSTGRES_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


async def reset_thread(checkpointer: BaseCheckpointSaver, thread_id: str) -> None:
    """Delete every checkpoint stored under ``thread_id``.

    Args:
        checkpointer: The active checkpointer yielded by ``get_checkpointer()``.
        thread_id: The session identifier to clear.
    """
    if isinstance(checkpointer, MemorySaver):
        _reset_memory(checkpointer, thread_id)
    else:
        backend = type(checkpointer).__name__
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

            if isinstance(checkpointer, AsyncSqliteSaver):
                await _reset_sqlite(checkpointer, thread_id)
                logger.info("session | reset thread_id=%s (sqlite)", thread_id)
                return
        except ImportError:
            pass

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

            if isinstance(checkpointer, AsyncPostgresSaver):
                await _reset_postgres(checkpointer, thread_id)
                logger.info("session | reset thread_id=%s (postgres)", thread_id)
                return
        except ImportError:
            pass

        raise NotImplementedError(f"reset_thread not supported for {backend}")


# ---------------------------------------------------------------------------
# Backend-specific implementations
# ---------------------------------------------------------------------------


def _reset_memory(checkpointer: MemorySaver, thread_id: str) -> None:
    """Clear all in-memory checkpoints for the thread."""
    # MemorySaver stores: storage[(thread_id, ns, checkpoint_id)] = checkpoint
    keys = [k for k in list(checkpointer.storage.keys()) if k[0] == thread_id]
    for k in keys:
        del checkpointer.storage[k]
    # Clear write buffer too.
    if hasattr(checkpointer, "writes"):
        wkeys = [k for k in list(checkpointer.writes.keys()) if k[0] == thread_id]
        for k in wkeys:
            del checkpointer.writes[k]
    logger.info("session | reset thread_id=%s (memory) — %d entries removed", thread_id, len(keys))


async def _reset_sqlite(checkpointer: object, thread_id: str) -> None:
    """Delete all rows for ``thread_id`` from the SQLite checkpoint tables."""
    conn = checkpointer.conn  # type: ignore[attr-defined]
    for table in _SQLITE_TABLES:
        await conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))  # noqa: S608
    await conn.commit()


async def _reset_postgres(checkpointer: object, thread_id: str) -> None:
    """Delete all rows for ``thread_id`` from the Postgres checkpoint tables."""
    pool = checkpointer.conn  # type: ignore[attr-defined]
    async with pool.connection() as conn:
        for table in _POSTGRES_TABLES:
            await conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s",  # noqa: S608
                (thread_id,),
            )
