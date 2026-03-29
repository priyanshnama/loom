"""Neo4j knowledge graph for Loom.

Primary use: querying the Constitution of India knowledge graph.

Constitution Schema
-------------------
(:Constitution)
(:Preamble {text})
(:Part {number, title})
(:Chapter {number, title})
(:Article {number, heading, omitted})
(:Clause {id, number, text, article_id})
(:Sub-clause {id, number, text, clause_id})
(:Schedule {number, title, ordinal})
(:Entry {id, number, text})

Relationships: CONTAINS, HAS_INTRO, HAS_SCHEDULE, HAS_APPENDIX

"""

from __future__ import annotations

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
    return _shared_driver


# ---------------------------------------------------------------------------
# Read operations (Neo4j is read-only — no writes to the constitution KG)
# ---------------------------------------------------------------------------


async def query_knowledge(query: str, thread_id: str, limit: int = 8) -> list[str]:  # noqa: ARG001
    """Search the Constitution of India KG for content relevant to *query*.

    Searches Articles (by heading), Clauses (by text), Schedules, and Entries.
    Returns formatted snippets like "Article 19 – <heading>: <clause text>".
    Falls back to session SearchResult cache if no constitutional matches found.
    """
    driver = await get_kg_driver()
    words = [w.lower() for w in re.split(r"\W+", query) if len(w) > 3]
    if not words:
        return []

    snippets: list[str] = []

    # --- Check for direct article number reference (e.g. "article 21", "Art. 356") ---
    article_num_match = re.search(r"\bart(?:icle)?\.?\s*(\d+[A-Za-z]?)\b", query, re.IGNORECASE)

    async with driver.session(database=_shared_database) as session:
        # 1. Direct article lookup by number
        if article_num_match:
            art_num = article_num_match.group(1)
            r = await session.run(
                """
                MATCH (a:Article {number: $num})
                OPTIONAL MATCH (a)-[:CONTAINS]->(c:Clause)
                RETURN a.number AS num, a.heading AS heading,
                       collect({n: c.number, t: c.text}) AS clauses
                """,
                num=art_num,
            )
            records = await r.data()
            for rec in records:
                heading = rec.get("heading") or ""
                intro = f"Article {rec['num']} – {heading}"
                clause_texts = [
                    f"  ({cl['n']}) {cl['t']}"
                    for cl in (rec.get("clauses") or [])
                    if cl.get("t")
                ]
                if clause_texts:
                    snippets.append(intro + ":\n" + "\n".join(clause_texts[:6]))
                else:
                    snippets.append(intro)

        # 2. Keyword search in Article headings
        heading_conds = " OR ".join(
            f"toLower(a.heading) CONTAINS $w{i}" for i in range(len(words))
        )
        params: dict[str, Any] = {f"w{i}": w for i, w in enumerate(words)}
        params["limit"] = limit

        r = await session.run(
            f"""
            MATCH (a:Article)
            WHERE NOT a.omitted AND ({heading_conds})
            OPTIONAL MATCH (a)-[:CONTAINS]->(c:Clause)
            WITH a, collect({{n: c.number, t: c.text}}) AS clauses
            RETURN a.number AS num, a.heading AS heading, clauses
            ORDER BY toInteger(a.number)
            LIMIT $limit
            """,
            params,
        )
        records = await r.data()
        for rec in records:
            heading = rec.get("heading") or ""
            intro = f"Article {rec['num']} – {heading}"
            clause_texts = [
                f"  ({cl['n']}) {cl['t']}"
                for cl in (rec.get("clauses") or [])
                if cl.get("t")
            ]
            entry = intro + (":\n" + "\n".join(clause_texts[:4]) if clause_texts else "")
            if entry not in snippets:
                snippets.append(entry)

        # 3. Keyword search in Clause text
        if len(snippets) < limit:
            clause_conds = " OR ".join(
                f"toLower(c.text) CONTAINS $w{i}" for i in range(len(words))
            )
            params2: dict[str, Any] = {f"w{i}": w for i, w in enumerate(words)}
            params2["limit"] = limit - len(snippets)

            r = await session.run(
                f"""
                MATCH (c:Clause)
                WHERE ({clause_conds})
                RETURN c.article_id AS art_id, c.number AS num, c.text AS text
                LIMIT $limit
                """,
                params2,
            )
            records = await r.data()
            seen_arts: set[str] = set()
            for rec in records:
                art_id = rec.get("art_id") or "?"
                if art_id in seen_arts:
                    continue
                seen_arts.add(art_id)
                snippet = f"Article {art_id}, Clause {rec['num']}: {rec['text']}"
                if snippet not in snippets:
                    snippets.append(snippet)

        # 4. Schedule / Entry search
        if len(snippets) < limit:
            sched_conds = " OR ".join(
                f"toLower(e.text) CONTAINS $w{i}" for i in range(len(words))
            )
            params3: dict[str, Any] = {f"w{i}": w for i, w in enumerate(words)}
            params3["limit"] = max(2, limit - len(snippets))

            r = await session.run(
                f"""
                MATCH (s:Schedule)-[:CONTAINS]->(e:Entry)
                WHERE ({sched_conds})
                RETURN s.ordinal AS sched, s.title AS stitle, e.number AS enum, e.text AS etext
                LIMIT $limit
                """,
                params3,
            )
            records = await r.data()
            for rec in records:
                snippet = (
                    f"{rec.get('sched','?')} Schedule ({rec.get('stitle','')}), "
                    f"Entry {rec.get('enum','?')}: {rec.get('etext','')}"
                )
                if snippet not in snippets:
                    snippets.append(snippet)

    logger.debug(
        "knowledge_graph | constitution query | hits=%d | query=%r", len(snippets), query
    )
    return snippets[:limit]


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
