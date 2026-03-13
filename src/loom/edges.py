"""Conditional edge logic for the Loom graph.

Keeping routing decisions in their own module makes them easy to test
independently and easy to swap without touching the graph topology.
"""

from __future__ import annotations

import logging
from typing import Literal

from loom.state import LoomState

logger = logging.getLogger(__name__)

# Symbolic route constants — used in graph.py to name the branch targets.
ROUTE_REFINE = "refine"  # loop the agent node again with updated tool outputs
ROUTE_END = "__end__"


def should_continue(state: LoomState) -> Literal["refine", "__end__"]:
    """Decide whether the agent should refine its answer or finish.

    Routing logic (evaluated in order):
    1. If there is no response yet, refine (run the agent again).
    2. If ``max_iterations`` has been reached, end regardless of confidence.
    3. If the response confidence is below the threshold, refine.
    4. Otherwise, the answer is good enough — end.

    Note: tool execution now happens *inside* pydantic-ai's agent.run() call.
    This loop only fires when the final structured answer has low confidence,
    triggering another full agent turn (which may call tools again internally).

    Args:
        state: Current graph state snapshot.

    Returns:
        ``"refine"`` to run the agent node again, or ``"__end__"`` to finish.
    """
    if state.response is None:
        logger.debug("should_continue | no response yet → refine")
        return ROUTE_REFINE

    if state.iteration_count >= state.max_iterations:
        logger.info(
            "should_continue | max_iterations=%d reached → end",
            state.max_iterations,
        )
        return ROUTE_END

    if state.response.needs_refinement:
        logger.info(
            "should_continue | confidence=%.2f below threshold → refine",
            state.response.confidence_score,
        )
        return ROUTE_REFINE

    logger.info(
        "should_continue | confidence=%.2f sufficient → end",
        state.response.confidence_score,
    )
    return ROUTE_END
