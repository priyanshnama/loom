"""Smoke test: build the graph with a MemorySaver and verify it compiles."""

from langgraph.checkpoint.memory import MemorySaver

from loom.graph import NODE_AGENT, build_graph


def test_graph_compiles() -> None:
    graph = build_graph(MemorySaver())
    assert NODE_AGENT in graph.get_graph().nodes


def test_graph_has_self_loop_on_agent() -> None:
    graph = build_graph(MemorySaver())
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    # Low-confidence path loops the agent back to itself.
    assert (NODE_AGENT, NODE_AGENT) in edges
