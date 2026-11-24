"""应用启动时的对象装配。

把所有对象的创建集中在这里，而不是让每个模块互相创建依赖。启动时的
顺序如下：

    Qwen Chat/Embedding
      -> Milvus VectorStore
      -> RAGService
      -> SpringBootClient
      -> IndexService
      -> 可选 MCP 工具
      -> create_agent
      -> ChatService
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agent import create_memory_checkpointer, create_tourism_agent
from app.chat import ChatService
from app.config import Settings, get_logger
from app.indexer import DocumentDownloader, IndexJobManager, IndexPipeline, IndexService
from app.llm import create_chat_model, create_embeddings
from app.rag import RAGService, create_vector_store
from app.springboot import SpringBootClient
from app.tools import LOCAL_TOOLS, load_mcp_tools

logger = get_logger(__name__)


@dataclass(slots=True)
class ApplicationRuntime:
    index_service: IndexService | None
    rag_service: RAGService | None
    springboot_client: SpringBootClient
    chat_service: ChatService
    agent: Any | None

    async def close(self) -> None:
        await self.chat_service.close()
        if self.index_service is not None:
            await self.index_service.close()
        await self.springboot_client.aclose()


async def build_runtime(settings: Settings) -> ApplicationRuntime:
    chat_model = create_chat_model(settings)
    embeddings = create_embeddings(settings)

    vector_store = None
    rag_service: RAGService | None = None
    index_service: IndexService | None = None
    try:
        vector_store = await asyncio.wait_for(
            asyncio.to_thread(create_vector_store, embeddings, settings=settings),
            timeout=15,
        )
    except Exception:
        # Milvus 暂时不可用时，健康检查会返回 not_ready；服务本身仍可启动，
        # 这样前端能看到明确的依赖错误，而不是整个进程直接退出。
        logger.exception("milvus_vector_store_initialization_failed")

    if vector_store is not None:
        rag_service = RAGService(
            vector_store=vector_store,
            chat_model=chat_model,
            settings=settings,
        )

    springboot_client = SpringBootClient(settings)
    if vector_store is not None:
        pipeline = IndexPipeline(
            vector_store=vector_store,
            springboot_client=springboot_client,
            downloader=DocumentDownloader(settings),
            settings=settings,
        )
        index_service = IndexService(IndexJobManager(pipeline, settings), pipeline)

    mcp_tools = []
    try:
        mcp_tools = await load_mcp_tools(settings)
    except Exception:
        # MCP 是可选增强。没有 Node、网络或 12306 服务时，不应该影响本地
        # 景点查询和 RAG 问答。
        logger.exception("mcp_tools_initialization_failed")

    checkpointer = create_memory_checkpointer()
    agent = create_tourism_agent(
        chat_model=chat_model,
        checkpointer=checkpointer,
        tools=[*LOCAL_TOOLS, *mcp_tools],
    )
    chat_service = ChatService(
        settings=settings,
        springboot_client=springboot_client,
        rag_service=rag_service,
        agent=agent,
        checkpointer=checkpointer,
    )
    return ApplicationRuntime(
        index_service=index_service,
        rag_service=rag_service,
        springboot_client=springboot_client,
        chat_service=chat_service,
        agent=agent,
    )
