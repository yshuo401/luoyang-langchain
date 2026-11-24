"""进程配置、协议常量和少量基础设施函数。

把配置集中在一个文件的原因：读代码时只需要打开这里，就能知道项目依赖哪些
环境变量；不需要在 7 个 `core/*.py` 文件之间跳转。
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Header
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import InternalAuthError

# ai-service/app/config.py -> parents[1] 就是 ai-service 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVICE_NAME = "luoyang-ai-service"
SERVICE_VERSION = "0.4.0"

INTERNAL_API_PREFIX = "/internal/v1/ai"
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
REQUEST_ID_HEADER = "X-Internal-Request-Id"
ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"


class Settings(BaseSettings):
    """从 `.env` 和系统环境变量读取配置。

    字段名小写，环境变量使用同名大写。例如 `qwen_api_key` 对应
    `QWEN_API_KEY`。Pydantic 会在启动时做类型校验，避免运行到一半才发现
    端口写成字符串、超时写成负数。
    """

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

    # 千问聊天模型：OpenAI 兼容接口
    qwen_api_key: SecretStr
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen3.8-flash"
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_request_timeout_seconds: float = Field(default=25.0, gt=0)
    qwen_max_retries: int = Field(default=2, ge=0)

    # MinerU 文档解析。文本文件不会调用 MinerU，因此该 Key 可为空；
    # PDF/DOCX/图片解析时如果没有配置，会返回明确的配置错误。
    mineru_api_key: SecretStr = SecretStr("")
    mineru_base_url: str | None = None
    mineru_model: str | None = None
    mineru_timeout_seconds: int = Field(default=300, ge=30)
    mineru_bypass_proxy: bool = True

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "luoyang_tourism_knowledge_v2"
    milvus_metric_type: str = "COSINE"
    milvus_search_timeout_seconds: float = Field(default=8.0, gt=0)

    springboot_internal_base_url: str = "http://127.0.0.1:8080"
    internal_token: SecretStr

    rag_top_k: int = Field(default=5, ge=1, le=8)
    agent_max_tool_calls: int = Field(default=6, ge=1)
    agent_recursion_limit: int = Field(default=12, ge=1)
    tool_timeout_seconds: float = Field(default=20.0, gt=0)
    chat_total_timeout_seconds: float = Field(default=28.0, gt=0)

    index_max_chunks: int = Field(default=2000, ge=1)
    index_callback_batch_size: int = Field(default=50, ge=1)
    download_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)

    # MCP 是可选的。关闭时只加载本地六个工具，启动不依赖 Node/npx/网络。
    mcp_12306_enabled: bool = False
    mcp_12306_command: str = Field(default_factory=lambda: "npx.cmd" if os.name == "nt" else "npx")
    mcp_tool_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("qwen_base_url", "milvus_uri", "springboot_internal_base_url")
    @classmethod
    def validate_urls(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("必须是合法的 http(s) URL")
        return value.rstrip("/")

    @field_validator("mineru_base_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("MINERU_BASE_URL 必须是 http(s) URL")
        return value.rstrip("/") if value else None

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
    """进程内只创建一次 Settings，避免每个请求重复读取 .env。"""

    return Settings()


def configure_logging(level: str = "INFO") -> None:
    """使用简单文本日志。

    毕业设计不需要先把日志做成 JSON 可观测平台。保留时间、级别、模块和
    消息已经足够定位问题。
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_ulid(timestamp_ms: int | None = None) -> str:
    """生成 26 位 ULID，用于 jobId、threadId 和请求 ID。

    ULID 的字符集不包含 I/L/O/U，便于人工抄写。这里实现的是项目内部
    ID 生成，不涉及安全加密。
    """

    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    timestamp = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms

    def encode(value: int, length: int) -> str:
        chars = ["0"] * length
        for index in range(length - 1, -1, -1):
            chars[index] = alphabet[value & 31]
            value >>= 5
        return "".join(chars)

    return encode(timestamp, 10) + encode(secrets.randbits(80), 16)


async def verify_internal_token(
    x_internal_token: str = Header(alias=INTERNAL_TOKEN_HEADER),
) -> None:
    """校验 Java 调用 FastAPI 时携带的固定内部令牌。

    这是本地内网调用。它不替代公网 JWT；公网 JWT 永远由 Spring Boot 校验。
    """

    expected = get_settings().internal_token.get_secret_value()
    if not expected or not hmac.compare_digest(expected, x_internal_token):
        raise InternalAuthError()
