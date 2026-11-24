"""千问模型的高层封装。

只放两个工厂函数：

- 聊天模型：ChatOpenAI，走 OpenAI 兼容端点；
- 向量模型：DashScopeEmbeddings，使用 LangChain 社区提供的官方高层类。

不在这里手写 HTTP，也不自己解析 DashScope 的响应。
"""

from __future__ import annotations

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def create_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """创建千问聊天模型。

    `ChatOpenAI` 只要求兼容 OpenAI 的接口，因此可以直接对接阿里云百炼。
    鉴权、超时和重试都由底层客户端处理。
    """

    config = settings or get_settings()
    return ChatOpenAI(
        model=config.qwen_chat_model,
        api_key=config.qwen_api_key,
        base_url=config.qwen_base_url,
        temperature=0.1,
        timeout=config.qwen_request_timeout_seconds,
        max_retries=config.qwen_max_retries,
    )


def create_embeddings(settings: Settings | None = None) -> DashScopeEmbeddings:
    """创建千问文本向量模型。

    这里保留你学习资料中的最简形式，只额外显式传入项目已有的
    `QWEN_API_KEY`，避免要求再维护一个 DASHSCOPE_API_KEY 环境变量。
    """

    config = settings or get_settings()
    return DashScopeEmbeddings(
        model=config.qwen_embedding_model,
        dashscope_api_key=config.qwen_api_key.get_secret_value(),
        max_retries=config.qwen_max_retries,
    )
