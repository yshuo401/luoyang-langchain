"""Pydantic Settings：所有运行配置的唯一入口。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从 ai-service/.env 与进程环境读取配置，并执行启动期校验。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=8000, ge=1, le=65535)

    # 聊天走 OpenAI-compatible；qwen3-vl-embedding 走 DashScope 原生 SDK。
    qwen_api_key: SecretStr
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen3.8-flash"
    qwen_embedding_model: str = "qwen3-vl-embedding"
    qwen_embedding_dim: int = Field(default=1024, gt=0)
    qwen_request_timeout_seconds: float = Field(default=25.0, gt=0)
    qwen_max_retries: int = Field(default=2, ge=0)
    model_max_concurrency: int = Field(default=4, ge=1)

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "luoyang_tourism_knowledge_v1"
    milvus_metric_type: str = "COSINE"

    redis_url: str = "redis://127.0.0.1:6379/1"
    checkpoint_key_prefix: str = "langgraph:checkpoint:"
    checkpoint_ttl_seconds: int = Field(default=604800, gt=0)

    springboot_internal_base_url: str = "http://127.0.0.1:8080"
    internal_shared_secret: SecretStr
    internal_timestamp_tolerance_seconds: int = Field(default=300, gt=0)

    rag_top_k: int = Field(default=5, ge=1)
    rag_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rag_max_context_chars: int = Field(default=8000, gt=0)
    agent_max_tool_calls: int = Field(default=6, ge=1)
    agent_recursion_limit: int = Field(default=12, ge=1)
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    chat_total_timeout_seconds: float = Field(default=28.0, gt=0)

    index_max_chunks: int = Field(default=2000, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)

    @field_validator("qwen_base_url", "milvus_uri", "redis_url", "springboot_internal_base_url")
    @classmethod
    def validate_urls(cls, value: str) -> str:
        if not value.startswith(("http://", "https://", "redis://", "rediss://")):
            raise ValueError("必须是合法的 http(s) 或 redis(s) URL")
        return value.rstrip("/") if value.startswith(("http://", "https://")) else value

    @field_validator("fastapi_host")
    @classmethod
    def validate_loopback_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("FastAPI 只允许绑定本机回环地址")
        return value

    @field_validator("milvus_metric_type")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"COSINE", "IP", "L2"}:
            raise ValueError("MILVUS_METRIC_TYPE 仅支持 COSINE、IP 或 L2")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级配置单例；测试中可调用 get_settings.cache_clear()。"""

    return Settings()