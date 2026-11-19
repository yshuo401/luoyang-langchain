"""Milvus / LangChain VectorStore 工厂。"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

from app.core.config import Settings, get_settings


def ensure_no_proxy_for_url(url: str) -> None:
    """让内网 Milvus gRPC 绕过系统 HTTP 代理。

    gRPC 在部分 Windows 环境中会复用 HTTP_PROXY。若 192.168.100.128 未列入
    NO_PROXY，pymilvus 会错误地尝试连接本地代理端口并报服务不可用。
    """

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
) -> Milvus:
    """创建 Milvus VectorStore；连接参数全部来自统一配置。"""

    config = settings or get_settings()
    ensure_no_proxy_for_url(config.milvus_uri)
    connection_args = {"uri": config.milvus_uri}
    token = config.milvus_token.get_secret_value()
    if token:
        connection_args["token"] = token
    return Milvus(
        embedding_function=embedding_function,
        collection_name=config.milvus_collection,
        connection_args=connection_args,
        index_params={
            "metric_type": config.milvus_metric_type,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )