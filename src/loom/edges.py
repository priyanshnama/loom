"""Conditional edge logic for the Loom graph.

Keeping routing decisions in their own module makes them easy to test
independently and easy to swap without touching the graph topology.

Routing table
-------------
After agent_node runs, should_continue() picks one of four routes:

  tool_error  — a tool returned an error and we still have retries left;
                loop back to agent_node with the error context in state so
                the model can reformulate its query (self-correction).

  refine      — no tool error, but the response confidence is below the
                threshold; loop back to agent_node for another attempt.

  respond     — answer is good enough; proceed to respond_node where the
                Human-in-the-Loop interrupt fires before finalisation.

  __end__     — safety valve when max_iterations is exhausted; skip HITL
                and terminate immediately.
"""

from __future__ import annotations

import logging
from typing import Literal

from loom.state import LoomState

logger = logging.getLogger(__name__)

# Symbolic route constants — used in graph.py to name the branch targets.
ROUTE_TOOL_ERROR = "tool_error"   # loop agent with self-correction context
ROUTE_REFINE = "refine"           # loop agent for low-confidence refinement
ROUTE_RESPOND = "respond"         # proceed to HITL respond node
ROUTE_END = "__end__"

# Maximum number of tool-error retries before giving up and ending the run.
MAX_ERROR_RETRIES = 2


def should_continue(
    state: LoomState,
) -> Literal["tool_error", "refine", "respond", "__end__"]:
    """Decide the next step after agent_node completes.

    Routing logic (evaluated in order):
    1. If there is no response yet, refine (run the agent again).
    2. If ``max_iterations`` has been reached, end regardless of confidence.
    3. If a tool error was detected and retries remain, self-correct.
    4. If the response confidence is below the threshold, refine.
    5. Otherwise, the answer is good enough — proceed to respond (HITL).

    Args:
        state: Current graph state snapshot.

    Returns:
        One of the four ROUTE_* constants above.
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

    if state.last_tool_error is not None and state.error_count <= MAX_ERROR_RETRIES:
        logger.warning(
            "should_continue | tool_error (attempt %d/%d) → self-correct",
            state.error_count,
            MAX_ERROR_RETRIES,
        )
        return ROUTE_TOOL_ERROR

    if state.response.needs_refinement:
        logger.info(
            "should_continue | confidence=%.2f below threshold → refine",
            state.response.confidence_score,
        )
        return ROUTE_REFINE

    logger.info(
        "should_continue | confidence=%.2f sufficient → respond (HITL)",
        state.response.confidence_score,
    )
    return ROUTE_RESPOND
