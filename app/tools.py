"""Agent 可调用的工具集合。

一个工具就是一段“模型可以请求执行的函数”。本文件包含：

1. 六个业务工具：景点、景点详情、攻略、美食、知识库、推荐；
2. 工具调用预算和请求级上下文；
3. 来源登记（让每个工具结果变成 [S1]、[S2] 这样的引用）；
4. 可选的 MCP 工具包装。

工具函数不直接操作数据库或 Milvus SDK，而是通过：
`SpringBootClient`（业务事实）和 `RAGService`（知识库）完成工作。
"""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.errors import SpringBootError
from app.schemas import SourceItem

SourceType = Literal["KNOWLEDGE", "SPOT", "GUIDE", "FOOD", "RECOMMENDATION"]


# ---------------------------------------------------------------------------
# 1. 请求上下文：保存一次请求中的工具调用次数、错误和来源
# ---------------------------------------------------------------------------


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
                snippet=_clean_text(document.page_content, max_chars=500),
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
                snippet=_clean_text(str(summary)) if summary else None,
                deep_link=item.get("deepLink") if isinstance(item.get("deepLink"), str) else None,
            )
        )

    def add_mcp_result(self, tool_name: str, content: str) -> str:
        """登记 MCP 结果。

        当前 Java 的来源类型没有 MCP，因此使用 KNOWLEDGE。标题和摘要明确
        写明 12306 实时查询，避免把它伪装成普通知识库文档。
        """

        queried_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        compact = _clean_text(content, max_chars=420)
        snippet = f"{compact}（查询时间：{queried_at}，实时数据仅供参考）"
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


def _clean_text(value: Any, *, max_chars: int = 1200) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _safe_deep_link(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.startswith("/pages/") and "://" not in value else None


def _records(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        records = response.get("records", [])
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def _success(items: list[dict[str, Any]], *, truncated: bool = False) -> dict[str, Any]:
    return {
        "ok": True,
        "data": items,
        "meta": {"count": len(items), "truncated": truncated},
    }


def _begin_tool(name: str) -> tuple[RequestContext | None, dict[str, Any] | None]:
    context = current_request_context()
    if not context.reserve_tool_call(name):
        return None, {
            "ok": False,
            "errorCode": "TOOL_BUDGET_EXCEEDED",
            "message": "本轮工具调用次数已达上限",
            "retryable": False,
        }
    return context, None


def _tool_error(exc: Exception) -> dict[str, Any]:
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


def _register_records(
    context: RequestContext,
    source_type: SourceType,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        marker = context.add_business_item(source_type, record)
        enriched.append({**record, "marker": f"[{marker}]"})
    return enriched


# ---------------------------------------------------------------------------
# 2. 六个本地业务工具
# ---------------------------------------------------------------------------


class SpotSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="景点名称或关键词")
    category_id: str | None = Field(default=None, max_length=64, description="可选分类 ID")
    limit: int = Field(default=5, ge=1, le=10, description="最多返回条数")


class SpotDetailInput(BaseModel):
    spot_id: str = Field(min_length=1, max_length=64, description="景点 ID")


@tool("search_scenic_spots", args_schema=SpotSearchInput)
async def search_scenic_spots(
    keyword: str,
    category_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """搜索当前已发布景点，返回名称、摘要、地址、开放时间摘要和深链。"""

    context, error = _begin_tool("search_scenic_spots")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_spots(
            keyword=keyword,
            category_id=category_id,
            limit=limit,
        )
        records = _records(response)
        if not records:
            result = _success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = _register_records(context, "SPOT", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": _clean_text(item.get("name"), max_chars=100),
                "summary": _clean_text(item.get("summary"), max_chars=500),
                "categoryName": _clean_text(item.get("categoryName"), max_chars=100),
                "address": _clean_text(item.get("address"), max_chars=200),
                "openTimeDesc": _clean_text(item.get("openTimeDesc"), max_chars=200),
                "tags": item.get("tags", [])[:10] if isinstance(item.get("tags"), list) else [],
                "deepLink": _safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return _success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = _tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


@tool("get_spot_detail", args_schema=SpotDetailInput)
async def get_spot_detail(spot_id: str) -> dict[str, Any]:
    """获取单个已发布景点的摘要、开放时间、票务提示和深链。"""

    context, error = _begin_tool("get_spot_detail")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.get_spot(spot_id)
        if not isinstance(response, dict):
            return {
                "ok": False,
                "errorCode": "SPOT_NOT_FOUND",
                "message": "未找到该景点",
                "retryable": False,
            }
        marker = context.add_business_item("SPOT", response)
        item = {
            "id": str(response.get("id", "")),
            "name": _clean_text(response.get("name"), max_chars=100),
            "summary": _clean_text(response.get("summary"), max_chars=500),
            "descriptionText": _clean_text(response.get("descriptionText"), max_chars=1200),
            "openTimeDesc": _clean_text(response.get("openTimeDesc"), max_chars=200),
            "ticketInfo": _clean_text(response.get("ticketInfo"), max_chars=300),
            "address": _clean_text(response.get("address"), max_chars=200),
            "playDurationMinutes": response.get("playDurationMinutes"),
            "tags": response.get("tags", [])[:10] if isinstance(response.get("tags"), list) else [],
            "deepLink": _safe_deep_link(response.get("deepLink")),
            "marker": f"[{marker}]",
        }
        return _success([item])
    except Exception as exc:
        result = _tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


class GuideSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="路线、攻略或景点关键词")
    limit: int = Field(default=5, ge=1, le=10)


@tool("search_travel_guides", args_schema=GuideSearchInput)
async def search_travel_guides(keyword: str, limit: int = 5) -> dict[str, Any]:
    """搜索已发布洛阳攻略和路线，返回行程摘要与深链。"""

    context, error = _begin_tool("search_travel_guides")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_guides(keyword=keyword, limit=limit)
        records = _records(response)
        if not records:
            result = _success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = _register_records(context, "GUIDE", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "title": _clean_text(item.get("title"), max_chars=120),
                "summary": _clean_text(item.get("summary"), max_chars=500),
                "durationDays": item.get("durationDays"),
                "routeSummary": _clean_text(item.get("routeSummary"), max_chars=800),
                "deepLink": _safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return _success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = _tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


class FoodSearchInput(BaseModel):
    keyword: str = Field(min_length=1, max_length=100, description="美食名称或关键词")
    category_id: str | None = Field(default=None, max_length=64, description="可选分类 ID")
    limit: int = Field(default=5, ge=1, le=10)


@tool("search_foods", args_schema=FoodSearchInput)
async def search_foods(
    keyword: str,
    category_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """搜索洛阳美食，返回名称、摘要、推荐景点与深链。"""

    context, error = _begin_tool("search_foods")
    if error is not None:
        return error
    assert context is not None
    try:
        response = await context.springboot_client.search_foods(
            keyword=keyword,
            category_id=category_id,
            limit=limit,
        )
        records = _records(response)
        if not records:
            result = _success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = _register_records(context, "FOOD", records[:limit])
        items = [
            {
                "id": str(item.get("id", "")),
                "name": _clean_text(item.get("name"), max_chars=100),
                "summary": _clean_text(item.get("summary"), max_chars=500),
                "categoryName": _clean_text(item.get("categoryName"), max_chars=100),
                "recommendedSpots": item.get("recommendedSpots", [])[:10]
                if isinstance(item.get("recommendedSpots"), list)
                else [],
                "deepLink": _safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for item in enriched
        ]
        return _success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = _tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="历史文化或旅游知识问题")
    top_k: int = Field(default=5, ge=1, le=8, description="最多检索分片数")


@tool("retrieve_tourism_knowledge", args_schema=KnowledgeSearchInput)
async def retrieve_tourism_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """从 Milvus 知识库检索景点讲解、历史文化、洞窟、年代和游玩建议。"""

    context, error = _begin_tool("retrieve_tourism_knowledge")
    if error is not None:
        return error
    assert context is not None
    if context.rag_service is None:
        return {
            "ok": False,
            "errorCode": "MILVUS_UNAVAILABLE",
            "message": "知识库暂时不可用",
            "retryable": True,
        }
    try:
        documents = await context.rag_service.retrieve(query, top_k=top_k)
        if not documents:
            result = _success([])
            result["meta"]["reason"] = "RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD"
            return result
        items: list[dict[str, Any]] = []
        for document in documents:
            marker = context.add_document(document)
            items.append(
                {
                    "marker": f"[{marker}]",
                    "title": _clean_text(document.metadata.get("title"), max_chars=120),
                    "content": _clean_text(document.page_content, max_chars=1200),
                    "page": document.metadata.get("page"),
                    "sectionPath": document.metadata.get("section_path"),
                    "score": document.metadata.get("_normalized_score"),
                }
            )
        return _success(items)
    except Exception as exc:
        result = _tool_error(exc)
        context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


class RecommendationInput(BaseModel):
    scene: Literal["HOME", "HALF_DAY", "ONE_DAY", "NEARBY"] = Field(description="推荐场景")
    limit: int = Field(default=5, ge=1, le=10)
    context: str | None = Field(default=None, max_length=500, description="用户补充偏好")


@tool("recommend_spots", args_schema=RecommendationInput)
async def recommend_spots(
    scene: str,
    limit: int = 5,
    context: str | None = None,
) -> dict[str, Any]:
    """获取 Java 确定性推荐候选，AI 只能解释候选顺序和理由。"""

    request_context, error = _begin_tool("recommend_spots")
    if error is not None:
        return error
    assert request_context is not None
    try:
        response = await request_context.springboot_client.recommend(
            user_id=request_context.user_id,
            scene=scene,
            limit=limit,
            context=context,
        )
        records = _records(response)
        if not records:
            result = _success([])
            result["meta"]["reason"] = "NO_PUBLISHED_BUSINESS_DATA"
            return result
        enriched = _register_records(request_context, "RECOMMENDATION", records[:limit])
        items = [
            {
                "id": str(item.get("id") or item.get("spotId") or ""),
                "name": _clean_text(item.get("name") or item.get("title"), max_chars=100),
                "reason": _clean_text(item.get("reason") or item.get("summary"), max_chars=500),
                "rank": index + 1,
                "deepLink": _safe_deep_link(item.get("deepLink")),
                "marker": item["marker"],
            }
            for index, item in enumerate(enriched)
        ]
        return _success(items, truncated=len(records) > len(items))
    except Exception as exc:
        result = _tool_error(exc)
        request_context.record_error(str(result.get("errorCode", "TOOL_EXECUTION_ERROR")))
        return result


LOCAL_TOOLS: list[BaseTool] = [
    search_scenic_spots,
    get_spot_detail,
    search_travel_guides,
    search_foods,
    retrieve_tourism_knowledge,
    recommend_spots,
]

# 兼容旧阅读习惯：说明这六个就是 Agent 的静态工具白名单。
TOOLS = LOCAL_TOOLS


# ---------------------------------------------------------------------------
# 3. MCP：把远程/子进程工具变成同样的 LangChain Tool
# ---------------------------------------------------------------------------


def _mcp_text(result: Any) -> str:
    """把 MCP 返回的 content blocks 合并成普通字符串。"""

    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, list):
        parts = [_mcp_text(item) for item in result]
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        if "content" in result:
            return _mcp_text(result["content"])
        return json.dumps(result, ensure_ascii=False)
    return str(result or "")


def _wrap_mcp_tool(mcp_tool: BaseTool, settings: Settings) -> BaseTool:
    """给 MCP 工具增加工具预算、超时和来源登记。

    原工具的参数模型不改变，因此模型看到的仍然是 `get-tickets` 这类原始
    工具名和原始参数；变化只发生在工具执行后的结果包装。
    """

    async def invoke(**kwargs: Any) -> dict[str, Any]:
        context, error = _begin_tool(mcp_tool.name)
        if error is not None:
            return error
        assert context is not None
        try:
            result = await asyncio.wait_for(
                mcp_tool.ainvoke(kwargs),
                timeout=settings.mcp_tool_timeout_seconds,
            )
            content = _mcp_text(result).strip() or "12306 MCP 没有返回内容"
            marker = context.add_mcp_result(mcp_tool.name, content)
            return {
                "ok": True,
                "source": "12306-MCP",
                "marker": f"[{marker}]",
                "data": content[:4000],
                "notice": "实时数据仅供参考，请以 12306 官方页面为准。",
            }
        except TimeoutError:
            context.record_error("MCP_TIMEOUT")
            return {
                "ok": False,
                "errorCode": "MCP_TIMEOUT",
                "message": "12306 查询超时",
                "retryable": True,
            }
        except Exception as exc:
            context.record_error("MCP_TOOL_ERROR")
            return {
                "ok": False,
                "errorCode": "MCP_TOOL_ERROR",
                "message": f"12306 查询失败：{type(exc).__name__}",
                "retryable": True,
            }

    return StructuredTool.from_function(
        coroutine=invoke,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
    )


async def load_mcp_tools(settings: Settings | None = None) -> list[BaseTool]:
    """按配置加载 12306 MCP。

    默认关闭，原因是它会通过 npx 启动 Node 子进程并访问公网；如果失败，
    整个 AI 服务不应该启动失败，因此调用方会捕获异常并只记录日志。
    """

    config = settings or get_settings()
    if not config.mcp_12306_enabled:
        return []

    client = MultiServerMCPClient(
        {
            "rail12306": {
                "transport": "stdio",
                "command": config.mcp_12306_command,
                "args": ["-y", "12306-mcp"],
            }
        }
    )
    tools = await client.get_tools()
    return [_wrap_mcp_tool(item, config) for item in tools]


__all__ = [
    "LOCAL_TOOLS",
    "TOOLS",
    "RequestContext",
    "current_request_context",
    "reset_request_context",
    "set_request_context",
    "load_mcp_tools",
]
