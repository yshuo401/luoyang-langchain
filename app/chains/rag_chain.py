"""High-level RAG chain assembly and retrieval service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.core.config import Settings, get_settings
from app.integrations.milvus import create_vector_store
from app.integrations.qwen import create_chat_model, create_embedding_model
from app.prompts.rag import build_rag_prompt
from app.rag.embeddings import QwenMultiModalEmbeddings
from app.rag.retriever import ThresholdRetriever, create_retriever
from app.rag.vector_store import KnowledgeVectorStore

_RAG_DOCUMENT_PROMPT = ChatPromptTemplate.from_template(
    '<document source="{citation_marker}">\n{page_content}\n</document>'
)


def create_rag_chain(
    retriever: ThresholdRetriever,
    chat_model: BaseChatModel,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Use the official retrieval + document-combine high-level APIs."""

    combine_docs_chain = create_stuff_documents_chain(
        chat_model,
        build_rag_prompt(),
        document_prompt=_RAG_DOCUMENT_PROMPT,
        document_separator="\n\n",
    )
    return create_retrieval_chain(retriever, combine_docs_chain)


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    documents: list[Document]
    raw: dict[str, Any]


class RAGService:
    def __init__(
        self,
        *,
        embedding: QwenMultiModalEmbeddings,
        vector_store: KnowledgeVectorStore,
        retriever: ThresholdRetriever,
        chat_model: BaseChatModel,
        chain: Runnable[dict[str, Any], dict[str, Any]],
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.retriever = retriever
        self.chat_model = chat_model
        self.chain = chain

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RAGService:
        config = settings or get_settings()
        embedding = create_embedding_model(config)
        vector_store = KnowledgeVectorStore(
            create_vector_store(embedding, settings=config), config
        )
        retriever = create_retriever(vector_store, settings=config)
        chat_model = create_chat_model(config)
        return cls(
            embedding=embedding,
            vector_store=vector_store,
            retriever=retriever,
            chat_model=chat_model,
            chain=create_rag_chain(retriever, chat_model),
        )

    async def retrieve(
        self, query: str, *, top_k: int | None = None
    ) -> list[Document]:
        if top_k is not None and top_k != self.retriever.k:
            retriever = self.retriever.model_copy(update={"k": min(max(top_k, 1), 8)})
            return await retriever.ainvoke(query)
        return await self.retriever.ainvoke(query)

    async def answer(self, query: str) -> RAGAnswer:
        raw = await self.chain.ainvoke({"input": query})
        return RAGAnswer(
            answer=str(raw.get("answer", "")),
            documents=list(raw.get("context", [])),
            raw=raw,
        )

    async def aclose(self) -> None:
        # Milvus's LangChain wrapper owns no long-lived async HTTP session.
        return None
