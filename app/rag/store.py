"""Milvus 连接、健康检查和向量存储工厂。"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import MilvusClient

from app.config import Settings, get_settings


def ensure_no_proxy_for_url(url: str) -> None:
    """让内网 Milvus gRPC 绕过误配置的 HTTP 代理。"""

    host = urlparse(url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if host not in entries:
            entries.append(host)
        os.environ[key] = ",".join(entries)


def create_vector_store(
    embedding_function: Embeddings,
    *,
    settings: Settings | None = None,
    collection_name: str | None = None,
) -> Milvus:
    """创建 LangChain 的 Milvus VectorStore。"""

    config = settings or get_settings()
    ensure_no_proxy_for_url(config.milvus_uri)
    connection_args = {"uri": config.milvus_uri}
    token = config.milvus_token.get_secret_value()
    if token:
        connection_args["token"] = token

    return Milvus(
        embedding_function=embedding_function,
        collection_name=collection_name or config.milvus_collection,
        connection_args=connection_args,
        index_params={
            "metric_type": config.milvus_metric_type,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
        search_params={
            "metric_type": config.milvus_metric_type,
            "params": {"ef": 64},
        },
        auto_id=False,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        enable_dynamic_field=True,
    )


def check_milvus(settings: Settings) -> bool:
    """健康检查使用的最小连接测试。"""

    ensure_no_proxy_for_url(settings.milvus_uri)
    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
    )
    try:
        client.list_collections()
        return True
    finally:
        client.close()
