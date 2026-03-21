"""Neo4j knowledge graph for Loom.

All search results, entities, and their relationships are stored here so the
agent can retrieve relevant prior knowledge before hitting external tools.

Schema
------
(:Thread {id})
(:SearchResult {id, query, result, thread_id, created_at})
(:Entity {name, type})
(:Thread)-[:HAS_RESULT]->(:SearchResult)
(:SearchResult)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)   (co-occurrence within same result)
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _driver() -> Any:
    """Return the module-level Neo4j async driver (created lazily)."""
    import neo4j  # noqa: PLC0415

    from loom.config import settings  # noqa: PLC0415

    return neo4j.AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


# ---------------------------------------------------------------------------
# One shared driver instance (created on first use, reused afterwards)
# ---------------------------------------------------------------------------

_shared_driver: Any = None
_shared_database: str = "neo4j"


async def get_kg_driver() -> Any:
    global _shared_driver
    if _shared_driver is None:
        _shared_driver = _driver()
        from loom.config import settings  # noqa: PLC0415

        global _shared_database
        _shared_database = settings.neo4j_database
        await _setup_schema(_shared_driver, _shared_database)
    return _shared_driver


async def _setup_schema(driver: Any, database: str) -> None:
    async with driver.session(database=database) as session:
        await session.run(
            "CREATE INDEX thread_id_idx IF NOT EXISTS FOR (t:Thread) ON (t.id)"
        )
        await session.run(
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)"
        )
        await session.run(
            "CREATE INDEX search_result_id_idx IF NOT EXISTS "
            "FOR (s:SearchResult) ON (s.id)"
        )
    logger.info("knowledge_graph | schema ready")


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def _extract_entities(text: str) -> list[str]:
    """Very lightweight entity extraction: capitalised words / quoted phrases."""
    # Quoted phrases
    quoted = re.findall(r"'([^']{3,40})'|\"([^\"]{3,40})\"", text)
    entities = [q[0] or q[1] for q in quoted]
    # Capitalised word sequences (e.g. "Quantum Entanglement")
    cap_seq = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    entities.extend(cap_seq)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen and len(e) > 2:
            seen.add(key)
            result.append(e)
    return result[:20]  # cap at 20 per result


async def store_search_result(query: str, result: str, thread_id: str) -> None:
    """Persist a search result and its entities into the knowledge graph."""
    driver = await get_kg_driver()
    result_id = hashlib.sha256(f"{thread_id}:{query}:{result}".encode()).hexdigest()[:16]
    entities = _extract_entities(result)

    async with driver.session(database=_shared_database) as session:
        # Upsert Thread
        await session.run(
            "MERGE (:Thread {id: $tid})",
            tid=thread_id,
        )
        # Create SearchResult linked to Thread
        await session.run(
            """
            MERGE (s:SearchResult {id: $rid})
            SET s.query = $query,
                s.result = $result,
                s.thread_id = $tid,
                s.created_at = datetime()
            WITH s
            MATCH (t:Thread {id: $tid})
            MERGE (t)-[:HAS_RESULT]->(s)
            """,
            rid=result_id, query=query, result=result, tid=thread_id,
        )
        # Upsert entities and link to result; connect co-occurring entities
        for entity in entities:
            await session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = 'concept'
                WITH e
                MATCH (s:SearchResult {id: $rid})
                MERGE (s)-[:MENTIONS]->(e)
                """,
                name=entity, rid=result_id,
            )
        # Co-occurrence edges between entities in the same result
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                await session.run(
                    """
                    MATCH (a:Entity {name: $a}), (b:Entity {name: $b})
                    MERGE (a)-[:RELATED_TO]->(b)
                    """,
                    a=a, b=b,
                )
    logger.debug(
        "knowledge_graph | stored | thread=%s | entities=%d", thread_id, len(entities)
    )


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


async def query_knowledge(query: str, thread_id: str, limit: int = 5) -> list[str]:
    """Return stored search results relevant to *query* for this thread.

    Matching strategy (best-effort, no vector index required):
    1. Full-text scan: results whose ``query`` field contains any word from the
       input query (case-insensitive).
    2. Entity match: results that mention entities found in the input query.
    Results are deduplicated and capped at *limit*.
    """
    driver = await get_kg_driver()
    words = [w.lower() for w in re.split(r"\W+", query) if len(w) > 3]
    if not words:
        return []

    # Build a WHERE clause that matches any keyword
    conditions = " OR ".join(
        [f"toLower(s.query) CONTAINS $w{i}" for i in range(len(words))]
        + [f"toLower(s.result) CONTAINS $w{i}" for i in range(len(words))]
    )
    params: dict[str, Any] = {f"w{i}": w for i, w in enumerate(words)}
    params["tid"] = thread_id
    params["limit"] = limit

    async with driver.session(database=_shared_database) as session:
        result = await session.run(
            f"""
            MATCH (s:SearchResult)
            WHERE s.thread_id = $tid AND ({conditions})
            RETURN s.result AS result
            ORDER BY s.created_at DESC
            LIMIT $limit
            """,
            **params,
        )
        records = await result.data()

    snippets = [r["result"] for r in records if r.get("result")]
    logger.debug(
        "knowledge_graph | query | thread=%s | hits=%d", thread_id, len(snippets)
    )
    return snippets


async def get_related_entities(entity: str, depth: int = 1) -> list[str]:
    """Return entity names related to *entity* within *depth* hops."""
    driver = await get_kg_driver()
    async with driver.session(database=_shared_database) as session:
        result = await session.run(
            f"""
            MATCH (e:Entity {{name: $name}})-[:RELATED_TO*1..{depth}]-(r:Entity)
            RETURN DISTINCT r.name AS name
            LIMIT 20
            """,
            name=entity,
        )
        records = await result.data()
    return [r["name"] for r in records if r.get("name")]
