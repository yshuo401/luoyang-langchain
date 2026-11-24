"""工具共享上下文与辅助函数。

这个文件只处理“一次工具调用期间需要共享的东西”：

- 当前用户/会话；
- Spring Boot 和 RAG 客户端；
- 工具调用预算；
- 来源登记；
- 统一的成功/失败返回结构。

具体的景点、攻略、美食工具放在 `business.py`；
知识库工具放在 `knowledge.py`；MCP 放在 `mcp.py`。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain_core.documents import Document

from app.errors import SpringBootError
from app.schemas import SourceItem

SourceType = Literal["KNOWLEDGE", "SPOT", "GUIDE", "FOOD", "RECOMMENDATION"]


def compact_text(value: Any, *, max_chars: int = 1200) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def safe_deep_link(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.startswith("/pages/") and "://" not in value else None


@dataclass(slots=True)
class RequestContext:
    """一次聊天请求的临时状态。

    模型在一次回答里可能调用多个工具。工具不能把来源直接塞进全局变量，
    否则两个用户同时聊天时会串数据。因此这里使用 Python 的 ContextVar：
    每个异步请求拥有自己的上下文。
    """

    user_id: str
    thread_id: str
    springboot_client: Any
    rag_service: Any | None
    max_tool_calls: int
    sources: list[SourceItem] = field(default_factory=list)
    _source_keys: dict[tuple[str, str, str], int] = field(default_factory=dict)
    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)

    def reserve_tool_call(self, name: str) -> bool:
        """检查并消耗一次工具预算，防止模型无限调用工具。"""

        if self.tool_call_count >= self.max_tool_calls:
            self.record_error("TOOL_BUDGET_EXCEEDED")
            return False
        self.tool_call_count += 1
        self.tool_names.append(name)
        return True

    def record_error(self, code: str) -> None:
        self.tool_errors.append(code)

    def add_source(self, source: SourceItem) -> str:
        """登记来源并返回引用标记，例如 `S1`。

        同一个业务对象被多个工具重复查询时，只返回原来的编号，避免回答里
        出现重复的 [S1]、[S4] 指向同一景点。
        """

        key = (source.source_type, source.source_id or "", source.chunk_id or "")
        if key in self._source_keys:
            return f"S{self._source_keys[key]}"

        rank_no = len(self.sources) + 1
        stored = source.model_copy(update={"rank_no": rank_no})
        self.sources.append(stored)
        self._source_keys[key] = rank_no
        return f"S{rank_no}"

    def add_document(self, document: Document) -> str:
        """把 Milvus 检索结果登记为 KNOWLEDGE 来源。"""

        metadata = document.metadata
        normalized_score = metadata.get("_normalized_score")
        score = (
            max(0.0, min(1.0, float(normalized_score)))
            if isinstance(normalized_score, int | float)
            else None
        )
        vector_id = str(metadata.get("vector_id") or metadata.get("pk") or "")
        return self.add_source(
            SourceItem(
                source_type="KNOWLEDGE",
                source_id=str(metadata.get("document_id") or "") or None,
                chunk_id=vector_id or None,
                title=str(metadata.get("title") or "知识库资料"),
                snippet=compact_text(document.page_content, max_chars=500),
                score=score,
            )
        )

    def add_business_item(self, source_type: SourceType, item: dict[str, Any]) -> str:
        """把 Java 返回的景点/攻略/美食登记为来源。"""

        item_id = item.get("id") or item.get("spotId") or item.get("guideId") or item.get("foodId")
        title = item.get("name") or item.get("title") or "旅游业务数据"
        summary = item.get("summary") or item.get("descriptionText") or item.get("reason")
        return self.add_source(
            SourceItem(
                source_type=source_type,
                source_id=str(item_id) if item_id is not None else None,
                title=str(title),
                snippet=compact_text(summary) if summary else None,
                deep_link=item.get("deepLink") if isinstance(item.get("deepLink"), str) else None,
            )
        )

    def add_mcp_result(self, tool_name: str, content: str) -> str:
        """登记 MCP 结果。

        当前 Java 的来源类型没有 MCP，因此使用 KNOWLEDGE。标题和摘要明确
        写明 12306 实时查询，避免把它伪装成普通知识库文档。
        """

        queried_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        snippet = (
            f"{compact_text(content, max_chars=420)}（查询时间：{queried_at}，实时数据仅供参考）"
        )
        return self.add_source(
            SourceItem(
                source_type="KNOWLEDGE",
                source_id=f"MCP:{tool_name}",
                title=f"12306 实时查询（{tool_name}）",
                snippet=snippet,
            )
        )


_current_context: ContextVar[RequestContext | None] = ContextVar(
    "luoyang_request_context", default=None
)


def current_request_context() -> RequestContext:
    context = _current_context.get()
    if context is None:
        raise RuntimeError("工具在请求上下文之外被调用")
    return context


def set_request_context(context: RequestContext) -> Token[RequestContext | None]:
    return _current_context.set(context)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _current_context.reset(token)


def begin_tool(name: str) -> tuple[RequestContext | None, dict[str, Any] | None]:
    context = current_request_context()
    if not context.reserve_tool_call(name):
        return None, {
            "ok": False,
            "errorCode": "TOOL_BUDGET_EXCEEDED",
            "message": "本轮工具调用次数已达上限",
            "retryable": False,
        }
    return context, None


def tool_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, SpringBootError):
        return {
            "ok": False,
            "errorCode": exc.error_code,
            "message": exc.message,
            "retryable": exc.retryable,
        }
    return {
        "ok": False,
        "errorCode": "TOOL_EXECUTION_ERROR",
        "message": "数据查询暂时不可用",
        "retryable": True,
    }


def records_from_response(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        records = response.get("records", [])
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def register_records(
    context: RequestContext,
    source_type: SourceType,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        marker = context.add_business_item(source_type, record)
        enriched.append({**record, "marker": f"[{marker}]"})
    return enriched


def success_response(items: list[dict[str, Any]], *, truncated: bool = False) -> dict[str, Any]:
    return {
        "ok": True,
        "data": items,
        "meta": {"count": len(items), "truncated": truncated},
    }
