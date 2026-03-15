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
    ┌─────────┐ ◄──────────────────────────────────────┐
    │  agent  │                                         │
    └────┬────┘                                         │
         │ should_continue()                            │
         ├─── "tool_error"  ──────────────────────────►─┤  (self-correction loop)
         ├─── "refine"      ──────────────────────────►─┘  (low-confidence loop)
         │
         ├─── "__end__"     ──────────────────────────► END  (max iterations hit)
         │
         ▼ "respond"
    ┌─────────┐   ← interrupt_before fires here (HITL)
    │ respond │
    └────┬────┘
         │
         ▼
        END

Tools are invoked *inside* pydantic-ai's agent.run() call — there is no
separate tool node.  The self-correction loop fires when a tool output looks
like an error (see ``edges._is_tool_error``); the refinement loop fires when
the agent's confidence score is below the threshold.

Human-in-the-Loop
-----------------
When ``hitl=True`` the compiled graph inserts a breakpoint *before* the
``respond`` node.  ``graph.ainvoke()`` returns at that point; the caller
inspects the state and resumes with ``graph.ainvoke(None, config=config)``.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from loom.edges import ROUTE_END, ROUTE_REFINE, ROUTE_RESPOND, ROUTE_TOOL_ERROR, should_continue
from loom.models import LoomResponse  # noqa: F401 — registers type for checkpoint serde
from loom.nodes import agent_node, respond_node
from loom.state import LoomState

# Node name constants — referenced in tests and __main__.
NODE_AGENT = "agent"
NODE_RESPOND = "respond"


def build_graph(checkpointer: BaseCheckpointSaver, *, hitl: bool = False) -> object:
    """Construct and compile the Loom ``StateGraph``.

    Args:
        checkpointer: A ready-to-use checkpointer (Postgres, SQLite, or
                      in-memory).  Pass the value yielded by
                      ``get_checkpointer()``.
        hitl: When ``True``, compile with ``interrupt_before=[NODE_RESPOND]``
              so the graph pauses before the final response node, enabling
              Human-in-the-Loop approval.  Default: ``False``.

    Returns:
        A compiled ``CompiledGraph`` ready for ``.ainvoke()`` / ``.astream()``.

    Every invocation **must** supply a ``thread_id`` in the LangGraph config so
    the checkpointer can persist and resume state::

        config = {"configurable": {"thread_id": "my-session"}}
        await graph.ainvoke(initial_state, config=config)
    """
    builder = StateGraph(LoomState)

    # --- Nodes -----------------------------------------------------------
    builder.add_node(NODE_AGENT, agent_node)
    builder.add_node(NODE_RESPOND, respond_node)

    # --- Edges -----------------------------------------------------------
    builder.add_edge(START, NODE_AGENT)

    # After agent_node, route based on confidence, errors, and iteration count.
    builder.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {
            ROUTE_TOOL_ERROR: NODE_AGENT,   # self-correction: loop with error ctx
            ROUTE_REFINE: NODE_AGENT,        # low confidence: loop for refinement
            ROUTE_RESPOND: NODE_RESPOND,     # good answer: proceed to HITL node
            ROUTE_END: END,                  # safety: max iterations exhausted
        },
    )

    builder.add_edge(NODE_RESPOND, END)

    # --- Compile ---------------------------------------------------------
    interrupt_before = [NODE_RESPOND] if hitl else []
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
