"""Checkpointer factory for LangGraph state persistence.

Three tiers selected by the ``LOOM_PERSISTENCE`` env var:

  memory  — ``MemorySaver`` (default): in-process RAM only.
  sqlite  — ``AsyncSqliteSaver``: file-backed SQLite.
  neo4j   — ``Neo4jCheckpointer``: custom Neo4j-backed saver.

Usage
-----
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        ...
"""

from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from loom.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Async context manager that yields the appropriate checkpointer."""
    backend = settings.loom_persistence.lower()

    if backend == "neo4j":
        async with _neo4j_checkpointer() as cp:
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


# ---------------------------------------------------------------------------
# Neo4j checkpointer
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _neo4j_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    try:
        import neo4j  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Install 'neo4j' to use Neo4j persistence.") from exc

    driver = neo4j.AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    logger.info("persistence | backend=neo4j | uri=%s", settings.neo4j_uri)
    cp = Neo4jCheckpointer(driver, settings.neo4j_database)
    await cp.setup()
    try:
        yield cp
    finally:
        await driver.close()


class Neo4jCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by Neo4j.

    Schema
    ------
    (:Checkpoint {thread_id, checkpoint_ns, checkpoint_id,
                  parent_checkpoint_id, type, checkpoint_data,
                  metadata_data, created_at})
    (:CheckpointWrite {thread_id, checkpoint_ns, checkpoint_id,
                       task_id, idx, channel, type, value})
    (:Checkpoint)-[:HAS_WRITE]->(:CheckpointWrite)
    """

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        super().__init__()
        self._driver = driver
        self._db = database

    async def setup(self) -> None:
        """Create indexes on first run."""
        async with self._driver.session(database=self._db) as session:
            r1 = await session.run(
                "CREATE INDEX checkpoint_lookup IF NOT EXISTS "
                "FOR (c:Checkpoint) ON (c.thread_id, c.checkpoint_ns, c.checkpoint_id)"
            )
            await r1.consume()
            r2 = await session.run(
                "CREATE INDEX checkpoint_write_lookup IF NOT EXISTS "
                "FOR (w:CheckpointWrite) "
                "ON (w.thread_id, w.checkpoint_ns, w.checkpoint_id, w.task_id, w.idx)"
            )
            await r2.consume()
        logger.info("persistence | Neo4j checkpointer ready")

    # ------------------------------------------------------------------
    # Sync stubs (required by abstract base class; always use async path)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:  # type: ignore[override]
        raise NotImplementedError("Use async interface")

    def list(  # type: ignore[override]
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError("Use async interface")

    def put(  # type: ignore[override]
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        raise NotImplementedError("Use async interface")

    def put_writes(  # type: ignore[override]
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        raise NotImplementedError("Use async interface")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enc(self, data: bytes) -> str:
        return base64.b64encode(data).decode()

    def _dec(self, s: str) -> bytes:
        return base64.b64decode(s)

    def _cfg(self, thread_id: str, ns: str, cid: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": cid}}

    def _build_tuple(self, node: dict, writes: list[dict]) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((node["type"], self._dec(node["checkpoint_data"])))
        metadata = json.loads(node["metadata_data"])
        config = self._cfg(node["thread_id"], node["checkpoint_ns"], node["checkpoint_id"])
        parent_config: RunnableConfig | None = None
        if node.get("parent_checkpoint_id"):
            parent_config = self._cfg(
                node["thread_id"], node["checkpoint_ns"], node["parent_checkpoint_id"]
            )
        pending_writes = [
            (w["task_id"], w["channel"], self.serde.loads_typed((w["type"], self._dec(w["value"]))))
            for w in sorted(writes, key=lambda x: x["idx"])
        ]
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        cfg = config["configurable"]
        thread_id: str = cfg["thread_id"]
        ns: str = cfg.get("checkpoint_ns", "")
        cid: str | None = cfg.get("checkpoint_id")

        async with self._driver.session(database=self._db) as session:
            if cid:
                result = await session.run(
                    """
                    MATCH (c:Checkpoint {thread_id: $tid, checkpoint_ns: $ns, checkpoint_id: $cid})
                    OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:CheckpointWrite)
                    RETURN c{.*} AS cp, collect(w{.*}) AS writes
                    """,
                    tid=thread_id, ns=ns, cid=cid,
                )
            else:
                result = await session.run(
                    """
                    MATCH (c:Checkpoint {thread_id: $tid, checkpoint_ns: $ns})
                    OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:CheckpointWrite)
                    WITH c, collect(w{.*}) AS writes
                    ORDER BY c.created_at DESC LIMIT 1
                    RETURN c{.*} AS cp, writes
                    """,
                    tid=thread_id, ns=ns,
                )
            record = await result.single()
        if not record:
            return None
        return self._build_tuple(dict(record["cp"]), [dict(w) for w in record["writes"]])

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:  # type: ignore[override]
        if not config:
            return
        cfg = config["configurable"]
        thread_id: str = cfg["thread_id"]
        ns: str = cfg.get("checkpoint_ns", "")

        where = "c.thread_id = $tid AND c.checkpoint_ns = $ns"
        params: dict[str, Any] = {"tid": thread_id, "ns": ns}

        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                where += " AND c.checkpoint_id < $before_id"
                params["before_id"] = before_id

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            MATCH (c:Checkpoint)
            WHERE {where}
            OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:CheckpointWrite)
            WITH c, collect(w{{.*}}) AS writes
            ORDER BY c.created_at DESC {limit_clause}
            RETURN c{{.*}} AS cp, writes
        """

        async with self._driver.session(database=self._db) as session:
            result = await session.run(query, **params)
            async for record in result:
                yield self._build_tuple(dict(record["cp"]), [dict(w) for w in record["writes"]])

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        cfg = config["configurable"]
        thread_id: str = cfg["thread_id"]
        ns: str = cfg.get("checkpoint_ns", "")
        parent_id: str | None = cfg.get("checkpoint_id")
        cid: str = checkpoint["id"]

        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        meta_str = json.dumps(metadata)

        async with self._driver.session(database=self._db) as session:
            result = await session.run(
                """
                MERGE (c:Checkpoint {thread_id: $tid, checkpoint_ns: $ns, checkpoint_id: $cid})
                SET c.parent_checkpoint_id = $parent_id,
                    c.type = $type,
                    c.checkpoint_data = $cp_data,
                    c.metadata_data = $meta_data,
                    c.created_at = datetime()
                """,
                tid=thread_id, ns=ns, cid=cid,
                parent_id=parent_id,
                type=cp_type,
                cp_data=self._enc(cp_bytes),
                meta_data=meta_str,
            )
            await result.consume()
        return self._cfg(thread_id, ns, cid)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        cfg = config["configurable"]
        thread_id: str = cfg["thread_id"]
        ns: str = cfg.get("checkpoint_ns", "")
        cid: str = cfg["checkpoint_id"]

        async with self._driver.session(database=self._db) as session:
            for idx, (channel, value) in enumerate(writes):
                w_type, w_bytes = self.serde.dumps_typed(value)
                result = await session.run(
                    """
                    MATCH (c:Checkpoint {thread_id: $tid, checkpoint_ns: $ns, checkpoint_id: $cid})
                    MERGE (w:CheckpointWrite {
                        thread_id: $tid, checkpoint_ns: $ns, checkpoint_id: $cid,
                        task_id: $task_id, idx: $idx
                    })
                    SET w.channel = $channel, w.type = $type, w.value = $val
                    MERGE (c)-[:HAS_WRITE]->(w)
                    """,
                    tid=thread_id, ns=ns, cid=cid,
                    task_id=task_id, idx=idx,
                    channel=channel, type=w_type, val=self._enc(w_bytes),
                )
                await result.consume()
