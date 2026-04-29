"""LangGraph StateGraph definition for Loom.

Graph topology
--------------

    ┌─────────────┐
    │    START    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  researcher │  — uses tools (Wikipedia, calculator), writes research_notes
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐ ◄──────────────────────────────────────┐
    │ synthesizer │                                         │
    └──────┬──────┘                                         │
           │ should_continue()                              │
           ├─── "refine"   ───────────────────────────────►─┘  (low-confidence loop)
           │
           ├─── "__end__"  ───────────────────────────────► END  (max iterations)
           │
           ▼ "respond"
    ┌─────────────┐   ← interrupt_before fires here (HITL)
    │   respond   │
    └──────┬──────┘
           │
           ▼
          END

The researcher runs once per query and stores its findings in research_notes.
The synthesizer loops until confidence is sufficient or max_iterations is hit.
Tools are invoked inside researcher's pydantic-ai agent.run() call — no separate
tool node.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from loom.edges import ROUTE_END, ROUTE_REFINE, ROUTE_RESPOND, ROUTE_TOOL_ERROR, should_continue
from loom.models import LoomResponse  # noqa: F401 — registers type for checkpoint serde
from loom.nodes import researcher_node, respond_node, synthesizer_node
from loom.state import LoomState

NODE_RESEARCHER = "researcher"
NODE_SYNTHESIZER = "synthesizer"
NODE_RESPOND = "respond"


def build_graph(checkpointer: BaseCheckpointSaver, *, hitl: bool = False) -> object:
    """Construct and compile the Loom StateGraph.

    Args:
        checkpointer: A ready-to-use checkpointer yielded by get_checkpointer().
        hitl: When True, interrupt before respond_node for Human-in-the-Loop approval.
    """
    builder = StateGraph(LoomState)

    builder.add_node(NODE_RESEARCHER, researcher_node)
    builder.add_node(NODE_SYNTHESIZER, synthesizer_node)
    builder.add_node(NODE_RESPOND, respond_node)

    builder.add_edge(START, NODE_RESEARCHER)
    builder.add_edge(NODE_RESEARCHER, NODE_SYNTHESIZER)

    builder.add_conditional_edges(
        NODE_SYNTHESIZER,
        should_continue,
        {
            ROUTE_TOOL_ERROR: NODE_SYNTHESIZER,  # synthesizer has no tools; dead path kept for symmetry
            ROUTE_REFINE: NODE_SYNTHESIZER,
            ROUTE_RESPOND: NODE_RESPOND,
            ROUTE_END: END,
        },
    )

    builder.add_edge(NODE_RESPOND, END)

    interrupt_before = [NODE_RESPOND] if hitl else []
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
