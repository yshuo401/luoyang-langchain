"""LangChain Embeddings 导出点。

实际实现位于 app.integrations.qwen，避免 RAG 层重复持有模型凭据。
"""

from app.integrations.qwen import QwenMultiModalEmbeddings, create_embedding_model

__all__ = ["QwenMultiModalEmbeddings", "create_embedding_model"]