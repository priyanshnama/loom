"""LangGraph StateGraph definition for Loom.

This module is intentionally thin: it wires together nodes and edges that are
implemented in dedicated modules, so the graph topology is easy to read at a
glance without being cluttered by business logic.

Graph topology
--------------

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  agent  │◄──────────┐
    └────┬────┘            │ (refine — low confidence)
         │                 │
         ▼ should_continue()
    ┌─────────┐
    │   END   │
    └─────────┘

Tools are invoked *inside* pydantic-ai's agent.run() call — there is no
separate tool node.  The LangGraph loop only fires when the final answer has
low confidence, triggering a second full agent turn.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from loom.edges import ROUTE_END, ROUTE_REFINE, should_continue
from loom.models import LoomResponse  # noqa: F401 — registers type for checkpoint serde
from loom.nodes import agent_node
from loom.state import LoomState

# Node name constant — referenced in tests.
NODE_AGENT = "agent"


def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Construct and compile the Loom ``StateGraph``.

    Args:
        checkpointer: A ready-to-use checkpointer (Postgres or in-memory).
                      Pass the value yielded by ``get_checkpointer()``.

    Returns:
        A compiled ``CompiledGraph`` ready for ``.ainvoke()`` / ``.astream()``.
    """
    builder = StateGraph(LoomState)

    # --- Nodes -----------------------------------------------------------
    builder.add_node(NODE_AGENT, agent_node)

    # --- Edges -----------------------------------------------------------
    builder.add_edge(START, NODE_AGENT)

    # After the agent runs, check confidence:
    #   "refine" → loop back to agent (pydantic-ai will re-invoke tools internally)
    #   "__end__" → done
    builder.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {
            ROUTE_REFINE: NODE_AGENT,
            ROUTE_END: END,
        },
    )

    # --- Compile ---------------------------------------------------------
    return builder.compile(checkpointer=checkpointer)
