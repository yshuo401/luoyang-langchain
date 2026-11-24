"""统一业务异常。

FastAPI 的接口层只需要捕获 ServiceError，就能把内部错误转换成
`{code, message, data, traceId, timestamp}` 格式的统一响应。
"""

from __future__ import annotations


class ErrorCode:
    """项目内部错误码。

    这些数字与 Spring Boot 侧的错误处理保持同一套风格，但没有必要为每个
    异常都设计一个类。只有需要携带特殊状态或可重试信息时才定义子类。
    """

    INVALID_REQUEST = 40000
    UNAUTHORIZED = 40100
    RESOURCE_NOT_FOUND = 40400
    CONFLICT = 40900
    INTERNAL_ERROR = 50000
    DEPENDENCY_UNAVAILABLE = 50300
    MILVUS_UNAVAILABLE = 50310
    MODEL_UNAVAILABLE = 50320


class ServiceError(Exception):
    """所有可预期业务异常的基类。

    code: 业务错误码，返回给 Java 侧判断。
    message: 可以直接展示给用户的中文说明，不包含堆栈和密钥。
    http_status: HTTP 状态码。
    retryable: Java 或前端是否可以在短时间内重试。
    """

    def __init__(
        self,
        code: int,
        message: str,
        http_status: int = 400,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable


class InternalAuthError(ServiceError):
    """Spring Boot -> FastAPI 的内部令牌不正确。"""

    def __init__(self, message: str = "内部调用令牌不正确") -> None:
        super().__init__(ErrorCode.UNAUTHORIZED, message, 401, False)


class InvalidRequestError(ServiceError):
    def __init__(self, message: str = "请求参数不合法") -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, message, 400, False)


class ResourceNotFoundError(ServiceError):
    def __init__(self, message: str = "资源不存在") -> None:
        super().__init__(ErrorCode.RESOURCE_NOT_FOUND, message, 404, False)


class ConflictError(ServiceError):
    def __init__(self, message: str = "资源状态冲突") -> None:
        super().__init__(ErrorCode.CONFLICT, message, 409, False)


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


class SpringBootError(ServiceError):
    """调用 Java 内部接口失败时使用的异常。

    工具层会把该异常转换成结构化错误返回给模型，而不是把 Java 的完整堆栈
    暴露给用户。
    """

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
