"""Tool registry for the Loom agent.

Adding a new tool
-----------------
1. Create ``src/loom/tools/<domain>.py`` with one or more async functions.
2. Import here and add to RESEARCHER_TOOLS (tools available to the researcher agent).
3. That's it — researcher_node picks them up automatically.
"""

from __future__ import annotations

from loom.tools.calculator import calculate
from loom.tools.wikipedia import wikipedia_search

# Tools available to the researcher agent.
RESEARCHER_TOOLS: list = [wikipedia_search, calculate]
