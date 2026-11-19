"""验证锁定版本的 LangChain/LangGraph 顶层 API 仍可导入。"""

from __future__ import annotations

import inspect
from importlib.metadata import version

from langchain.agents import create_agent
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver


def main() -> None:
    checks = {
        "langchain": version("langchain"),
        "langchain-core": version("langchain-core"),
        "langchain-openai": version("langchain-openai"),
        "langgraph": version("langgraph"),
        "langgraph-checkpoint-redis": version("langgraph-checkpoint-redis"),
        "create_agent": "checkpointer" in inspect.signature(create_agent).parameters,
        "create_retrieval_chain": callable(create_retrieval_chain),
        "ChatOpenAI": callable(ChatOpenAI),
        "RedisSaver.setup": hasattr(RedisSaver, "setup"),
    }
    for key, value in checks.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()