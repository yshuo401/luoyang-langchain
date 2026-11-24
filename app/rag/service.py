"""RAG 检索服务与 `create_retrieval_chain` 示例。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_milvus import Milvus

from app.config import Settings, get_settings
from app.errors import MilvusUnavailableError


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    documents: list[Document]
    raw: dict[str, Any]


def create_retriever(vector_store: Milvus, k: int):
    """创建只检索 `enabled == true` 分片的 Retriever。"""

    return vector_store.as_retriever(
        search_kwargs={"k": k, "expr": "enabled == true"},
    )


def create_rag_chain(
    retriever: Any,
    chat_model: BaseChatModel,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """用官方高层 API 组合标准 RAG 链。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是洛阳旅游知识助手，只能依据本轮检索资料回答。"
                "每个关键事实使用 [S1]、[S2] 来源标记，资料不足时明确说资料不足。",
            ),
            (
                "human",
                "用户问题：{input}\n\n<knowledge>\n{context}\n</knowledge>",
            ),
        ]
    )
    combine_documents = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, combine_documents)


class RAGService:
    """Agent 知识库工具使用的检索服务。"""

    def __init__(
        self,
        *,
        vector_store: Milvus,
        chat_model: BaseChatModel,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.settings = settings or get_settings()
        self.chain = create_rag_chain(
            create_retriever(vector_store, self.settings.rag_top_k),
            chat_model,
        )

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """返回带分数和顺序号的文档，供工具登记为 [S1]、[S2]。"""

        k = min(max(top_k or self.settings.rag_top_k, 1), 8)
        try:
            pairs = await asyncio.wait_for(
                asyncio.to_thread(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k,
                    expr="enabled == true",
                ),
                timeout=self.settings.milvus_search_timeout_seconds,
            )
        except TimeoutError as exc:
            raise MilvusUnavailableError("知识库检索超时") from exc
        except Exception as exc:
            raise MilvusUnavailableError("知识库检索失败") from exc

        documents: list[Document] = []
        for rank_no, (document, raw_score) in enumerate(pairs, start=1):
            score = self._normalize_score(float(raw_score))
            document.metadata = {
                **document.metadata,
                "_score": float(raw_score),
                "_normalized_score": score,
                "rank_no": rank_no,
                "citation_marker": f"[S{rank_no}]",
            }
            documents.append(document)
        return documents

    async def answer(self, question: str) -> RAGAnswer:
        """标准 RAG 链的演示入口：检索 + 一次生成。"""

        raw = await self.chain.ainvoke({"input": question})
        return RAGAnswer(
            answer=str(raw.get("answer", "")),
            documents=list(raw.get("context", [])),
            raw=raw,
        )

    def _normalize_score(self, score: float) -> float:
        if self.settings.milvus_metric_type == "L2":
            return max(0.0, min(1.0, 1.0 / (1.0 + max(score, 0.0))))
        return max(0.0, min(1.0, score))

    async def aclose(self) -> None:
        return None
