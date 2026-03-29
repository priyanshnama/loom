"""Tool registry for the Loom agent.

Adding a new tool
-----------------
1. Create ``src/loom/tools/<domain>.py`` with one or more async functions.
2. Import the function(s) here and add them to ``TOOLS``.
3. That's it — ``_get_agent()`` in ``nodes.py`` picks them up automatically.
"""

from __future__ import annotations

from loom.tools.constitution import query_constitution

# Ordered list of all tools the pydantic-ai agent may call.
TOOLS: list = [
    query_constitution,
]
