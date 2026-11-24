"""Agent 工具包。

外部代码统一从这里导入：

```python
from app.tools import LOCAL_TOOLS, RequestContext, load_mcp_tools
```
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.business import (
    get_spot_detail,
    recommend_spots,
    search_foods,
    search_scenic_spots,
    search_travel_guides,
)
from app.tools.context import (
    RequestContext,
    current_request_context,
    reset_request_context,
    set_request_context,
)
from app.tools.knowledge import retrieve_tourism_knowledge
from app.tools.mcp import load_mcp_tools

LOCAL_TOOLS: list[BaseTool] = [
    search_scenic_spots,
    get_spot_detail,
    search_travel_guides,
    search_foods,
    retrieve_tourism_knowledge,
    recommend_spots,
]

TOOLS = LOCAL_TOOLS

__all__ = [
    "LOCAL_TOOLS",
    "TOOLS",
    "RequestContext",
    "current_request_context",
    "get_spot_detail",
    "load_mcp_tools",
    "recommend_spots",
    "reset_request_context",
    "retrieve_tourism_knowledge",
    "search_foods",
    "search_scenic_spots",
    "search_travel_guides",
    "set_request_context",
]
