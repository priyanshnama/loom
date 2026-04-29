"""Session management utilities.

Provides ``reset_thread`` which deletes all LangGraph checkpoints stored for a
given ``thread_id``, effectively clearing the conversation history.

Supports memory and SQLite backends.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_SQLITE_TABLES = ("checkpoints", "writes")


async def reset_thread(checkpointer: BaseCheckpointSaver, thread_id: str) -> None:
    """Delete every checkpoint stored under ``thread_id``."""
    if isinstance(checkpointer, MemorySaver):
        _reset_memory(checkpointer, thread_id)
        logger.info("session | reset thread_id=%s (memory)", thread_id)
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

        if isinstance(checkpointer, AsyncSqliteSaver):
            await _reset_sqlite(checkpointer, thread_id)
            logger.info("session | reset thread_id=%s (sqlite)", thread_id)
            return
    except ImportError:
        pass

    raise NotImplementedError(f"reset_thread not supported for {type(checkpointer).__name__}")


def _reset_memory(checkpointer: MemorySaver, thread_id: str) -> None:
    keys = [k for k in list(checkpointer.storage.keys()) if k[0] == thread_id]
    for k in keys:
        del checkpointer.storage[k]
    if hasattr(checkpointer, "writes"):
        for k in [k for k in list(checkpointer.writes.keys()) if k[0] == thread_id]:
            del checkpointer.writes[k]
    logger.info("session | reset thread_id=%s (memory) — %d entries removed", thread_id, len(keys))


async def _reset_sqlite(checkpointer: object, thread_id: str) -> None:
    conn = checkpointer.conn  # type: ignore[attr-defined]
    for table in _SQLITE_TABLES:
        await conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))  # noqa: S608
    await conn.commit()
