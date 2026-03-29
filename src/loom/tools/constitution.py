"""Tool for querying the Constitution of India knowledge graph."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def query_constitution(query: str) -> str:
    """Search the Constitution of India for articles, clauses, or schedules relevant to the query.

    Use this tool whenever the user asks about Indian constitutional law, fundamental rights,
    directive principles, government structure, schedules, or any legal provision in the
    Constitution of India.

    Args:
        query: Natural language question or keywords (e.g. "right to equality",
               "Article 21", "president powers", "seventh schedule").

    Returns:
        Matching constitutional provisions as formatted text.
    """
    from loom.knowledge_graph import query_knowledge  # noqa: PLC0415

    try:
        snippets = await query_knowledge(query, thread_id="constitution", limit=8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("query_constitution | KG unavailable: %s", exc)
        return "The Constitution knowledge graph is currently unavailable. Please try again later."
    if not snippets:
        return "No matching provisions found in the Constitution of India for that query."
    return "\n\n---\n\n".join(snippets)
