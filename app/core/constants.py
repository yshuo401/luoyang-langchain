"""Service-level constants and stable protocol values."""

from __future__ import annotations

from enum import IntEnum

SERVICE_NAME = "luoyang-ai-service"
SERVICE_VERSION = "0.2.0"
INTERNAL_API_PREFIX = "/internal/v1/ai"

HMAC_HEADER = "X-Internal-Signature"
TIMESTAMP_HEADER = "X-Internal-Timestamp"
REQUEST_ID_HEADER = "X-Internal-Request-Id"
IDEMPOTENCY_HEADER = "Idempotency-Key"

ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"

TOOL_MAX_RESULTS = 10
DEFAULT_SUGGESTED_QUESTIONS = [
    "洛阳有哪些适合半天游的景点？",
    "换个问法继续问。",
]


class ErrorCode(IntEnum):
    """Stable error codes shared with the Spring Boot caller."""

    OK = 0
    INVALID_REQUEST = 40000
    INVALID_INTERNAL_AUTH = 40110
    INDEX_JOB_CONFLICT = 40901
    CHAT_IN_PROGRESS = 40903
    JOB_NOT_FOUND = 40410
    THREAD_NOT_FOUND = 40410
    INTERNAL_ERROR = 50000
    DEPENDENCY_UNAVAILABLE = 50300
    MILVUS_UNAVAILABLE = 50301
    MODEL_UNAVAILABLE = 50302


class InsufficientReason:
    """Deterministic fallback reasons returned to the front end."""

    RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD"
    NO_PUBLISHED_BUSINESS_DATA = "NO_PUBLISHED_BUSINESS_DATA"
    TOOL_DEPENDENCY_UNAVAILABLE = "TOOL_DEPENDENCY_UNAVAILABLE"
    CITATION_MISSING = "CITATION_MISSING"
    CITATION_INVALID = "CITATION_INVALID"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    QUESTION_TOO_VAGUE = "QUESTION_TOO_VAGUE"
    SERVICE_DEGRADED = "SERVICE_DEGRADED"


class SourceType:
    KNOWLEDGE = "KNOWLEDGE"
    SPOT = "SPOT"
    GUIDE = "GUIDE"
    FOOD = "FOOD"
    RECOMMENDATION = "RECOMMENDATION"


class RecommendationScene:
    HOME = "HOME"
    HALF_DAY = "HALF_DAY"
    ONE_DAY = "ONE_DAY"
    NEARBY = "NEARBY"
