"""Tourism Agent construction through LangChain's top-level create_agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.prompts.system import TOURISM_AGENT_SYSTEM_PROMPT
from app.tools import TOOLS


def create_tourism_agent(
    *,
    chat_model: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any],
    tools: list[Any] | None = None,
) -> Any:
    """Build one reusable Agent graph with a static tool whitelist."""

    return create_agent(
        model=chat_model,
        tools=tools or TOOLS,
        system_prompt=TOURISM_AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name="luoyang_tourism_agent",
    )


def has_messages(state: Any) -> bool:
    if state is None:
        return False
    values = getattr(state, "values", None) or {}
    messages = values.get("messages", [])
    return bool(messages)


def extract_final_answer(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = _message_text(message.content)
        if content.strip():
            return content.strip()
    return ""


def extract_token_usage(result: dict[str, Any]) -> dict[str, int] | None:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return None
    prompt = 0
    completion = 0
    total = 0
    found = False
    for message in messages:
        if not isinstance(message, BaseMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        found = True
        prompt += int(usage.get("input_tokens", 0))
        completion += int(usage.get("output_tokens", 0))
        total += int(usage.get("total_tokens", 0))
    if not found:
        return None
    return {"prompt": prompt, "completion": completion, "total": total}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")
