"""Tool registry for the Loom agent.

Adding a new tool
-----------------
1. Create ``src/loom/tools/<domain>.py`` with one or more async functions.
2. Import the function(s) here and add them to ``TOOLS``.
3. That's it — ``_get_agent()`` in ``nodes.py`` picks them up automatically.
"""

from __future__ import annotations

from loom.tools.booking import book_flight, cancel_flight
from loom.tools.documents import download_document
from loom.tools.onboarding import onboard_user
from loom.tools.search import web_search

# Ordered list of all tools the pydantic-ai agent may call.
# pydantic-ai introspects each function's type annotations and docstring to
# build the JSON schema it sends to the model as tool definitions.
TOOLS: list = [
    web_search,
    book_flight,
    cancel_flight,
    onboard_user,
    download_document,
]
