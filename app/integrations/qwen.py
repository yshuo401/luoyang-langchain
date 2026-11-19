"""Qwen 高层客户端工厂。

- 聊天模型：ChatOpenAI -> 阿里云百炼 OpenAI-compatible endpoint。
- 向量模型：qwen3-vl-embedding 不支持 OpenAI-compatible Embeddings，
  使用官方 dashscope SDK 的 MultiModalEmbedding 高层接口，并适配为
  LangChain Embeddings。禁止在业务代码中手写 requests/httpx 模型请求。
"""

from __future__ import annotations

from collections.abc import Sequence

from dashscope import MultiModalEmbedding
from dashscope.legacy_1x.embeddings.multimodal_embedding import (
    MultiModalEmbeddingItemBase,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
)
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


def create_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """创建 Qwen 聊天模型；鉴权、重试、超时由官方客户端处理。"""

    config = settings or get_settings()
    return ChatOpenAI(
        model=config.qwen_chat_model,
        api_key=config.qwen_api_key,
        base_url=config.qwen_base_url,
        temperature=0.1,
        timeout=config.qwen_request_timeout_seconds,
        max_retries=config.qwen_max_retries,
    )


class QwenMultiModalEmbeddings(Embeddings):
    """把 qwen3-vl-embedding 的官方 SDK 调用适配为 LangChain Embeddings。

    该模型是原生多模态 Embedding，不支持 OpenAI-compatible /embeddings
    契约。本类只封装官方 SDK，不直接发 HTTP，也不打印文本或向量。
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        batch_size: int = 16,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    def _call(self, items: Sequence[MultiModalEmbeddingItemBase]) -> list[list[float]]:
        response = MultiModalEmbedding.call(
            model=self.model,
            input=list(items),
            api_key=self._api_key,
            dimension=self.dimension,
            output_type="dense",
            enable_fusion=False,
        )
        if response.status_code != 200 or not isinstance(response.output, dict):
            raise RuntimeError(
                "Qwen Embedding 调用失败: "
                f"status={response.status_code}, code={response.code}, "
                f"request_id={response.request_id}"
            )

        raw_embeddings = response.output.get("embeddings", [])
        ordered = sorted(raw_embeddings, key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
        if len(vectors) != len(items):
            raise RuntimeError("Qwen Embedding 返回数量与输入数量不一致")
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("Qwen Embedding 返回维度与 QWEN_EMBEDDING_DIM 不一致")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(
                self._call([MultiModalEmbeddingItemText(text=text, factor=1.0) for text in batch])
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call([MultiModalEmbeddingItemText(text=text, factor=1.0)])[0]

    def embed_image_urls(self, image_urls: Sequence[str]) -> list[list[float]]:
        """保留多模态图片向量入口，供后续景区图片检索使用。"""

        if not image_urls:
            return []
        return self._call(
            [MultiModalEmbeddingItemImage(image=url, factor=1.0) for url in image_urls]
        )


def create_embedding_model(settings: Settings | None = None) -> QwenMultiModalEmbeddings:
    """创建项目锁定的 Qwen 多模态向量模型。"""

    config = settings or get_settings()
    return QwenMultiModalEmbeddings(
        api_key=config.qwen_api_key.get_secret_value(),
        model=config.qwen_embedding_model,
        dimension=config.qwen_embedding_dim,
        batch_size=config.embedding_batch_size,
    )