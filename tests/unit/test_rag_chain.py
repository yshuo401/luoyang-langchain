from __future__ import annotations

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        return self.documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        return self.documents


def test_retrieval_chain_uses_official_high_level_api() -> None:
    from app.chains.rag_chain import create_rag_chain

    documents = [
        Document(
            page_content="建议游玩三到四小时。",
            metadata={"title": "龙门石窟", "citation_marker": "[S1]"},
        )
    ]
    chain = create_rag_chain(
        StaticRetriever(documents=documents),  # type: ignore[arg-type]
        FakeListChatModel(responses=["建议游玩三到四小时。[S1]"]),
    )
    result = chain.invoke({"input": "龙门石窟适合玩多久？"})
    assert result["answer"] == "建议游玩三到四小时。[S1]"
    assert result["context"][0].metadata["title"] == "龙门石窟"


def test_retrieval_only_chain_returns_documents_without_generation() -> None:
    from app.chains.rag_chain import create_retrieval_only_chain

    documents = [
        Document(
            page_content="龙门石窟建议游玩三到四小时。",
            metadata={"title": "龙门石窟", "citation_marker": "[S1]"},
        )
    ]
    chain = create_retrieval_only_chain(
        StaticRetriever(documents=documents)  # type: ignore[arg-type]
    )
    result = chain.invoke({"input": "龙门石窟适合玩多久？"})
    assert result["answer"] == ""
    assert result["context"][0].page_content == "龙门石窟建议游玩三到四小时。"
