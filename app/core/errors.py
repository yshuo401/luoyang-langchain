"""Classified service errors and safe HTTP mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.constants import ErrorCode


@dataclass(slots=True)
class ServiceError(Exception):
    """Base class for errors that may cross the internal HTTP boundary."""

    code: int
    message: str
    http_status: int = 500
    retryable: bool = False
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class InvalidRequestError(ServiceError):
    def __init__(self, message: str = "请求参数不合法") -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, message, 400, False)


class InternalAuthError(ServiceError):
    def __init__(self, message: str = "内部鉴权失败") -> None:
        super().__init__(ErrorCode.INVALID_INTERNAL_AUTH, message, 401, False)


class ResourceNotFoundError(ServiceError):
    def __init__(self, message: str = "资源不存在") -> None:
        super().__init__(ErrorCode.JOB_NOT_FOUND, message, 404, False)


class ConflictError(ServiceError):
    def __init__(
        self,
        message: str,
        *,
        code: int = ErrorCode.INDEX_JOB_CONFLICT,
        retryable: bool = False,
    ) -> None:
        super().__init__(code, message, 409, retryable)


class DependencyUnavailableError(ServiceError):
    def __init__(
        self,
        message: str = "依赖服务暂时不可用",
        *,
        code: int = ErrorCode.DEPENDENCY_UNAVAILABLE,
        retryable: bool = True,
    ) -> None:
        super().__init__(code, message, 503, retryable)


class MilvusUnavailableError(DependencyUnavailableError):
    def __init__(self, message: str = "知识库暂时不可用") -> None:
        super().__init__(message, code=ErrorCode.MILVUS_UNAVAILABLE)


class ModelUnavailableError(DependencyUnavailableError):
    def __init__(self, message: str = "智能问答模型暂时不可用") -> None:
        super().__init__(message, code=ErrorCode.MODEL_UNAVAILABLE)


class CheckpointerUnavailableError(DependencyUnavailableError):
    def __init__(self, message: str = "多轮会话存储暂时不可用") -> None:
        super().__init__(message, code=ErrorCode.DEPENDENCY_UNAVAILABLE)


class InternalError(ServiceError):
    def __init__(self, message: str = "AI 服务内部错误") -> None:
        super().__init__(ErrorCode.INTERNAL_ERROR, message, 500, False)


class SpringBootError(ServiceError):
    """Sanitized Spring Boot client failure suitable for a tool result."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool = False,
        http_status: int = 502,
    ) -> None:
        super().__init__(ErrorCode.INTERNAL_ERROR, message, http_status, retryable)
        self.error_code = error_code
