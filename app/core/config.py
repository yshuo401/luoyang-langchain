"""Pydantic Settings: the single entry point for runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Read ai-service/.env and process environment with startup validation."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_max_field_chars: int = Field(default=2000, ge=100)
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=8000, ge=1, le=65535)

    # Chat uses OpenAI-compatible ChatOpenAI; embedding uses DashScope SDK.
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
    milvus_search_timeout_seconds: float = Field(default=5.0, gt=0)

    redis_url: str = "redis://127.0.0.1:6379/1"
    redis_socket_timeout_seconds: float = Field(default=3.0, gt=0)
    checkpoint_key_prefix: str = "langgraph:checkpoint:"
    checkpoint_ttl_seconds: int = Field(default=604800, gt=0)
    checkpoint_setup_timeout_seconds: float = Field(default=10.0, gt=0)
    chat_result_cache_ttl_seconds: int = Field(default=600, gt=0)
    chat_inflight_ttl_seconds: int = Field(default=60, gt=0)
    index_job_ttl_seconds: int = Field(default=604800, gt=0)

    springboot_internal_base_url: str = "http://127.0.0.1:8080"
    internal_shared_secret: SecretStr
    internal_timestamp_tolerance_seconds: int = Field(default=300, gt=0)

    rag_top_k: int = Field(default=5, ge=1, le=8)
    rag_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rag_max_context_chars: int = Field(default=8000, gt=0)
    agent_max_tool_calls: int = Field(default=6, ge=1)
    agent_max_history_messages: int = Field(default=12, ge=1)
    agent_recursion_limit: int = Field(default=12, ge=1)
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    chat_total_timeout_seconds: float = Field(default=28.0, gt=0)

    index_max_chunks: int = Field(default=2000, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    index_callback_batch_size: int = Field(default=50, ge=1)
    download_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)

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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL 必须是标准 logging 级别")
        return normalized

    @model_validator(mode="after")
    def validate_time_budget(self) -> Settings:
        if self.chat_total_timeout_seconds >= 30:
            raise ValueError("CHAT_TOTAL_TIMEOUT_SECONDS 必须小于 Java 侧 30 秒预算")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-level settings singleton."""

    return Settings()
