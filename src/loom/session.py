"""Session management utilities.

Provides ``reset_thread`` which deletes all LangGraph checkpoints stored for a
given ``thread_id``, effectively clearing the conversation history.

Supports memory, SQLite, and Neo4j backends.
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

    # SQLite
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

        if isinstance(checkpointer, AsyncSqliteSaver):
            await _reset_sqlite(checkpointer, thread_id)
            logger.info("session | reset thread_id=%s (sqlite)", thread_id)
            return
    except ImportError:
        pass

    # Neo4j
    from loom.persistence import Neo4jCheckpointer  # noqa: PLC0415

    if isinstance(checkpointer, Neo4jCheckpointer):
        await _reset_neo4j(checkpointer, thread_id)
        logger.info("session | reset thread_id=%s (neo4j)", thread_id)
        return

    raise NotImplementedError(f"reset_thread not supported for {type(checkpointer).__name__}")


# ---------------------------------------------------------------------------
# Backend-specific implementations
# ---------------------------------------------------------------------------


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


async def _reset_neo4j(checkpointer: object, thread_id: str) -> None:
    from loom.persistence import Neo4jCheckpointer  # noqa: PLC0415

    cp: Neo4jCheckpointer = checkpointer  # type: ignore[assignment]
    async with cp._driver.session(database=cp._db) as session:
        await session.run(
            """
            MATCH (c:Checkpoint {thread_id: $tid})
            OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:CheckpointWrite)
            DETACH DELETE c, w
            """,
            tid=thread_id,
        )
