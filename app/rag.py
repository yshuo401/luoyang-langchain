"""RAG 的唯一入口：文档解析、切分、Milvus、检索链。

阅读顺序建议：

1. `extract_markdown()`：把上传文件变成一段 Markdown；
2. `split_markdown()`：把 Markdown 切成带元数据的 `Document`；
3. `create_vector_store()`：把 `Document` 和 Embedding 交给 Milvus；
4. `RAGService.retrieve()`：Agent 工具调用它取资料；
5. `create_rag_chain()`：保留一个标准 `create_retrieval_chain` 示例。

以前这里有 `rag/embeddings.py`、`rag/loaders.py`、`rag/splitter.py`、
`rag/retriever.py`、`rag/vector_store.py` 五个文件。现在合并成一个文件，
但每一步仍然用小函数分开，方便单独测试。
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from mineru import MinerU
from pymilvus import MilvusClient

from app.config import Settings, get_settings
from app.errors import MilvusUnavailableError
from app.schemas import IndexJobSubmitRequest

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_MINERU_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
MAX_CONTEXT_CHARS = 12000


class DocumentParseError(Exception):
    """文档解析失败。`code` 用于索引任务状态，不把异常堆栈返回给 Java。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    documents: list[Document]
    raw: dict[str, Any]


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
    """创建 LangChain 的 Milvus VectorStore。

    这里只保留知识库需要的字段和 HNSW 索引。`auto_id=False` 表示主键由
    项目生成，便于重复索引同一文档时覆盖旧向量。
    """

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


def _decode_text(content: bytes) -> str:
    """中文知识文档常见的三种编码。"""

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("PARSE_ERROR", "文本编码不受支持")


def sanitize_text(text: str) -> str:
    """去掉控制字符和空行，保留 Markdown 标题结构。"""

    cleaned = _CONTROL.sub("", text)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


@contextmanager
def _mineru_proxy_context(enabled: bool):
    """仅对 MinerU 同步调用临时移除代理环境变量。

    这是为了让 Windows 上常见的全局 HTTP_PROXY 不干扰 MinerU 的下载地址。
    调用结束后立即恢复原值。
    """

    if not enabled:
        yield
        return

    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def extract_markdown(
    content: bytes,
    file_name: str,
    settings: Settings | None = None,
) -> str:
    """把上传文件转成 Markdown。

    TXT/Markdown 直接读取；PDF/DOCX/图片交给 MinerU。MinerU 是同步 SDK，
    因此 Indexer 会把这个函数放到线程中执行，避免阻塞 FastAPI 事件循环。
    """

    if not content:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档内容为空")

    config = settings or get_settings()
    suffix = Path(file_name).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        text = sanitize_text(_decode_text(content))
        if not text:
            raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可解析正文")
        return text

    if suffix not in _MINERU_SUFFIXES:
        raise DocumentParseError(
            "UNSUPPORTED_FILE_TYPE",
            "仅支持 TXT、Markdown、PDF、DOC/DOCX 和常见图片格式",
        )

    token = config.mineru_api_key.get_secret_value()
    if not token:
        raise DocumentParseError(
            "MINERU_NOT_CONFIGURED",
            "解析 PDF/DOCX/图片需要配置 MINERU_API_KEY",
        )

    # MinerU 需要文件路径，不能直接接收 bytes。临时目录退出时会自动清理。
    with tempfile.TemporaryDirectory(prefix="luoyang-ai-") as temp_dir:
        safe_name = Path(file_name).name or f"document{suffix}"
        local_path = Path(temp_dir) / safe_name
        local_path.write_bytes(content)

        kwargs: dict[str, Any] = {"token": token}
        if config.mineru_base_url:
            kwargs["base_url"] = config.mineru_base_url
        client = MinerU(**kwargs)
        try:
            # 本机开发环境可能配置了 HTTP_PROXY，而 MinerU 返回的下载地址走
            # 另一个域名，代理握手会失败。只在这个同步调用期间临时绕过代理，
            # 不改变 Spring Boot / Qwen 客户端的网络设置。
            with _mineru_proxy_context(config.mineru_bypass_proxy):
                result = client.extract(
                    str(local_path),
                    model=config.mineru_model,
                    ocr=True,
                    timeout=config.mineru_timeout_seconds,
                )
        except Exception as exc:
            raise DocumentParseError("MINERU_FAILED", "MinerU 文档解析失败") from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    if result.state != "done" or not result.markdown:
        raise DocumentParseError("MINERU_FAILED", result.error or "MinerU 没有返回 Markdown")

    markdown = sanitize_text(result.markdown)
    if not markdown:
        raise DocumentParseError("EMPTY_DOCUMENT", "MinerU 返回的正文为空")
    return markdown


def split_markdown(
    markdown: str,
    request: IndexJobSubmitRequest,
    *,
    settings: Settings | None = None,
) -> list[Document]:
    """先按 Markdown 标题切分，再按长度切分。

    `MarkdownHeaderTextSplitter` 会保留 h1/h2/h3 元数据，后续来源卡片可以
    显示“章节路径”。`RecursiveCharacterTextSplitter` 负责把太长的小节切成
    800 字左右的 chunk。
    """

    config = settings or get_settings()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "，", " "],
        keep_separator=True,
    )

    sections = header_splitter.split_text(markdown)
    if not sections:
        sections = [Document(page_content=markdown, metadata={})]

    documents: list[Document] = []
    for section in sections:
        chunks = text_splitter.split_documents([section])
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue
            if len(documents) >= config.index_max_chunks:
                raise DocumentParseError("TOO_MANY_CHUNKS", "文档分片数超过上限")

            metadata = dict(chunk.metadata)
            heading_parts = [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
            metadata.update(
                {
                    "document_id": request.document_id,
                    "document_version": request.document_version,
                    "chunk_index": len(documents),
                    "title": request.title,
                    "category": request.category or "",
                    "source_url": "",
                    "section_path": " > ".join(heading_parts),
                    "enabled": True,
                }
            )
            documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise DocumentParseError("EMPTY_DOCUMENT", "文档没有可索引的分片")
    return documents


def create_retriever(vector_store: Milvus, k: int):
    """LangChain 标准 Retriever。

    `expr` 是 Milvus 过滤表达式，只检索 `enabled == true` 的知识分片。
    """

    return vector_store.as_retriever(
        search_kwargs={"k": k, "expr": "enabled == true"},
    )


def create_rag_chain(
    retriever: Any,
    chat_model: BaseChatModel,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """用官方高层 API 组合出标准 RAG 链。

    `create_retrieval_chain` 负责“先检索，再把 context 交给回答链”。
    Agent 问答不会重复调用这条链，而是把检索封装成工具；这里保留它作为
    RAG 技术点本身的可运行示例。
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是洛阳旅游知识助手，只能依据本轮检索资料回答。"
                "每个关键事实使用 [S1]、[S2] 来源标记，资料不足时明确说资料不足。",
            ),
            (
                "human",
                "用户问题：{input}\n\n<knowledge>\n{context}\n</knowledge>",
            ),
        ]
    )
    combine_documents = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, combine_documents)


class RAGService:
    """Agent 工具使用的检索服务。

    这个类只做三件事：把问题变成 Milvus 检索、把结果整理成 Document、
    保留一个标准 RAG 链供论文演示。
    """

    def __init__(
        self,
        *,
        vector_store: Milvus,
        chat_model: BaseChatModel,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.settings = settings or get_settings()
        self.chain = create_rag_chain(
            create_retriever(vector_store, self.settings.rag_top_k),
            chat_model,
        )

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """返回带分数和顺序号的文档，供工具登记为 [S1]、[S2]。"""

        k = min(max(top_k or self.settings.rag_top_k, 1), 8)
        try:
            pairs = await asyncio.wait_for(
                asyncio.to_thread(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k,
                    expr="enabled == true",
                ),
                timeout=self.settings.milvus_search_timeout_seconds,
            )
        except TimeoutError as exc:
            raise MilvusUnavailableError("知识库检索超时") from exc
        except Exception as exc:
            raise MilvusUnavailableError("知识库检索失败") from exc

        documents: list[Document] = []
        for rank_no, (document, raw_score) in enumerate(pairs, start=1):
            score = self._normalize_score(float(raw_score))
            document.metadata = {
                **document.metadata,
                "_score": float(raw_score),
                "_normalized_score": score,
                "rank_no": rank_no,
                "citation_marker": f"[S{rank_no}]",
            }
            documents.append(document)
        return documents

    async def answer(self, question: str) -> RAGAnswer:
        """标准 RAG 链的演示入口：检索 + 一次生成。"""

        raw = await self.chain.ainvoke({"input": question})
        return RAGAnswer(
            answer=str(raw.get("answer", "")),
            documents=list(raw.get("context", [])),
            raw=raw,
        )

    def _normalize_score(self, score: float) -> float:
        if self.settings.milvus_metric_type == "L2":
            return max(0.0, min(1.0, 1.0 / (1.0 + max(score, 0.0))))
        return max(0.0, min(1.0, score))

    async def aclose(self) -> None:
        # 当前 Milvus 包装不持有需要显式关闭的异步会话。
        return None
