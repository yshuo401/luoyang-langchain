# `ai-service` 全方位代码讲解（学习友好版）

> 阅读对象：已经会基础 Python，能看懂函数、类、异步 `async/await` 的初学者。
> 阅读目标：读完后你能从一次 HTTP 请求开始，讲清楚 Agent、工具、RAG、Milvus、MinerU、checkpointer 和索引任务分别做了什么。
> 当前结构：`app/` 下只保留 `tools/`、`rag/`、`indexer/` 三个有明确边界的包，其余模块继续平铺。旧的 `agents/`、`chains/`、`core/`、`services/`、`integrations/` 等目录已经移除。

---

## 0. 先记住一张图

```text
                     Spring Boot（唯一业务入口）
                              │
                 X-Internal-Token + JSON
                              ▼
                    FastAPI :8000（仅本机）
                              │
                    ChatService.chat()
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Guard 固定回答        LangGraph Agent        IndexService
                               │                     │
                    create_agent + InMemorySaver   下载图片/PDF
                               │                     │
             ┌─────────────────┼──────────────┐      ▼
             ▼                 ▼              ▼   MinerU -> Markdown
       本地六个工具         RAGService        MCP 12306
             │                 │              │
             ▼                 ▼              ▼
      Spring Boot          Milvus         npx 12306-mcp
      内部业务接口         向量检索         实时车票查询
```

你只需要先理解三句话：

1. **Spring Boot 管业务**：用户、收藏、景点、攻略、美食、推荐、聊天记录和 MySQL 都在 Java 侧；
2. **FastAPI 管 AI**：它不做登录，不直连业务 MySQL，只接收 Java 的内部请求；
3. **Agent 管选择工具**：模型负责决定“查景点还是查知识库”，Java 和 Python 负责保证结果真实、可追溯。

---

## 1. 三个包 + 平铺文件

```text
app/
├─ config.py          配置和常量：程序从哪里读 Key、端口、超时
├─ errors.py          可预期错误：参数错、依赖不可用、内部 token 错
├─ schemas.py         HTTP 协议：请求和响应到底长什么样
├─ llm.py             模型：千问聊天 + 千问向量
├─ springboot.py      反向调用 Java 内部接口
├─ agent.py           create_agent、提示词、引用校验
├─ chat.py            一次问答的主流程
├─ runtime.py         启动时把所有对象组装起来
├─ main.py            FastAPI 的 7 个接口
├─ tools/             工具包
│  ├─ context.py      RequestContext、预算、来源登记
│  ├─ business.py     景点/攻略/美食/推荐工具
│  ├─ knowledge.py    RAG 工具
│  └─ mcp.py          12306 MCP
├─ rag/               RAG 包
│  ├─ documents.py    MinerU、文本读取、切分
│  ├─ store.py        Milvus 连接和 collection
│  └─ service.py      retrieve + create_retrieval_chain
└─ indexer/           索引包
   ├─ downloader.py   OSS 临时 URL 下载
   ├─ pipeline.py     一条索引流水线
   ├─ jobs.py         任务状态、幂等、后台调度
   └─ service.py      给 FastAPI 的索引服务门面
```

阅读顺序建议：

```text
config -> schemas -> llm -> rag/documents.py -> rag/store.py -> rag/service.py
  -> springboot.py -> tools/context.py -> tools/business.py -> tools/knowledge.py
  -> tools/mcp.py -> agent.py -> chat.py -> indexer/ -> runtime.py -> main.py
```

不要从 `main.py` 开始逐行啃。`main.py` 只是 HTTP 入口，真正的业务在 `chat.py`、`tools/` 和 `indexer/`。第一次阅读可以只追一条链路：**`chat.py -> agent.py -> tools/knowledge.py -> rag/service.py`**。

---

## 2. `config.py`：配置到底从哪里来

### 2.1 `Settings` 类

`Settings` 继承 `BaseSettings`，它会自动读取：

1. 系统环境变量；
2. 项目根目录的 `ai-service/.env`。

字段名和环境变量名大小写不敏感，例如：

```python
qwen_api_key
```

对应：

```text
QWEN_API_KEY
```

### 2.2 关键配置分组

#### Qwen

```text
QWEN_API_KEY
QWEN_BASE_URL
QWEN_CHAT_MODEL
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

- `QWEN_CHAT_MODEL` 给 `ChatOpenAI` 使用；
- `QWEN_EMBEDDING_MODEL` 给 `DashScopeEmbeddings` 使用；
- `QWEN_BASE_URL` 是 OpenAI 兼容端点，不是自己拼的 URL。

#### MinerU

```text
MINERU_API_KEY
MINERU_BASE_URL
MINERU_MODEL
MINERU_TIMEOUT_SECONDS
MINERU_BYPASS_PROXY
```

TXT/Markdown 不会调用 MinerU；PDF、DOCX、图片才会。

`MINERU_BYPASS_PROXY=true` 是为了解决 Windows 上常见的全局代理导致 MinerU 下载结果失败的问题。代码只在一次 MinerU 调用期间临时移除代理变量，调用结束后恢复。

#### Milvus

```text
MILVUS_URI
MILVUS_TOKEN
MILVUS_COLLECTION=luoyang_tourism_knowledge_v2
MILVUS_METRIC_TYPE=COSINE
```

`v2` 不是随手改的名字。Embedding 模型从旧的 `qwen3-vl-embedding` 换成 `text-embedding-v4` 后，旧向量空间不能和新查询向量混用，必须重新建 collection。

#### Spring Boot 内部接口

```text
SPRINGBOOT_INTERNAL_BASE_URL
INTERNAL_TOKEN
```

这两个值只用于 FastAPI 调 Java，不用于公网 JWT。

### 2.3 为什么用 `SecretStr`

```python
qwen_api_key: SecretStr
```

`SecretStr` 在日志中默认显示为 `**********`，避免你不小心打印配置时泄露 Key。读取真实值时必须显式调用：

```python
settings.qwen_api_key.get_secret_value()
```

### 2.4 `get_settings()`

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

`lru_cache` 表示第一次读取后缓存，后续调用复用同一个配置对象，避免每个请求都去解析 `.env`。

### 2.5 `new_ulid()`

生成 26 位 ID，例如 `01ARZ3NDEKTSV4RRFFQ69G5FAV`。用于：

- `threadId`；
- 索引任务 ID；
- 内部请求 ID。

### 2.6 `verify_internal_token()`

它比较请求头 `X-Internal-Token` 和服务端配置是否一致。这里使用 `hmac.compare_digest` 做常量时间比较，避免简单的字符串比较时序差异。

注意：它只保护 Spring Boot 到 FastAPI 的内部调用，不替代用户 JWT。

---

## 3. `errors.py`：错误为什么要分类

FastAPI 代码可能遇到：

- 请求参数不合法；
- 内部 token 错误；
- 资源不存在；
- Milvus 不可用；
- 千问不可用；
- Spring Boot 返回错误。

如果每个地方都直接 `raise Exception("出错")`，`main.py` 无法区分该返回 400、404、409、503 还是 500。

所以项目定义：

```python
class ServiceError(Exception):
    code: int
    message: str
    http_status: int
    retryable: bool
```

所有可预期异常都继承它，例如：

```python
class MilvusUnavailableError(DependencyUnavailableError): ...
class ModelUnavailableError(DependencyUnavailableError): ...
```

`main.py` 只需要一个处理器：

```python
@app.exception_handler(ServiceError)
async def service_error_handler(...):
    ...
```

就能把异常统一转换为：

```json
{
  "code": 50310,
  "message": "知识库暂时不可用",
  "data": {"retryable": true},
  "traceId": "...",
  "timestamp": "..."
}
```

### 为什么不让错误堆栈直接返回给用户

因为错误堆栈可能包含：

- 内网地址；
- 环境变量；
- 文件路径；
- 第三方调用细节。

用户只需要知道“暂时不可用，可以重试”，详细堆栈写到服务端日志。

---

## 4. `schemas.py`：请求和响应的合同

### 4.1 `CamelModel`

Java 习惯用：

```json
{"threadId": "..."}
```

Python 习惯用：

```python
request.thread_id
```

`CamelModel` 做了两件事：

1. 序列化时自动把 `thread_id` 转成 `threadId`；
2. 读取时允许 `threadId` 映射到 `thread_id`。

```python
class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )
```

`extra="forbid"` 很重要：Java 若拼错字段，FastAPI 直接返回参数错误，而不是悄悄忽略。

### 4.2 `ApiEnvelope`

所有内部接口统一返回：

```json
{
  "code": 0,
  "message": "success",
  "data": {},
  "traceId": "request-001",
  "timestamp": "..."
}
```

`ApiEnvelope.ok()` 和 `ApiEnvelope.error()` 负责创建这个结构。

### 4.3 `ChatRequest`

```python
class ChatRequest(CamelModel):
    thread_id: str
    user_id: str
    message: str
    client_message_id: str
    request_id: str
    recovery_history: list[RecoveryMessage]
```

其中：

- `thread_id`：一次会话的 ID，LangGraph 用它找 checkpoint；
- `user_id`：Java 已完成登录校验后传来的用户 ID，用于推荐工具；
- `message`：用户本轮问题；
- `client_message_id`：Java 侧消息幂等 ID；
- `request_id`：链路追踪 ID；
- `recovery_history`：进程重启后恢复最近历史使用。

### 4.4 `SourceItem`

来源卡片的数据结构：

```python
class SourceItem(CamelModel):
    source_type: Literal["KNOWLEDGE", "SPOT", "GUIDE", "FOOD", "RECOMMENDATION"]
    source_id: str | None
    chunk_id: str | None
    title: str
    snippet: str | None
    score: float | None
    rank_no: int
    deep_link: str | None
```

`deep_link` 必须是 `/pages/...`，不能是 `http://evil.example`，这是基本的前端跳转安全限制。

### 4.5 `ChatResult`

```python
class ChatResult(CamelModel):
    answer: str
    grounded: bool
    insufficient: bool
    insufficient_reason: str | None
    sources: list[SourceItem]
    suggested_questions: list[str]
    model_name: str | None
    latency_ms: int
    token_usage: dict[str, Any] | None
```

`insufficient=true` 时 HTTP 仍然是 200。原因见第 10 节：资料不足是业务结果，不是网络错误。

### 4.6 索引模型

- `IndexJobSubmitRequest`：Spring Boot 提交索引任务；
- `IndexJobSubmitResponse`：返回 jobId；
- `IndexJobView`：查询任务进度；
- `KnowledgeChunkBatch`：FastAPI 回调 Java，写入 chunk 元数据。

---

## 5. `llm.py`：只保留两个模型工厂

### 5.1 `create_chat_model()`

```python
return ChatOpenAI(
    model=config.qwen_chat_model,
    api_key=config.qwen_api_key,
    base_url=config.qwen_base_url,
    temperature=0.1,
    timeout=config.qwen_request_timeout_seconds,
    max_retries=config.qwen_max_retries,
)
```

为什么用 `ChatOpenAI`：

- 千问兼容 OpenAI 协议；
- 鉴权、超时、重试、消息格式都由 LangChain/OpenAI 客户端处理；
- 不需要手写 `requests.post`。

### 5.2 `create_embeddings()`

```python
return DashScopeEmbeddings(
    model=config.qwen_embedding_model,
    dashscope_api_key=config.qwen_api_key.get_secret_value(),
    max_retries=config.qwen_max_retries,
)
```

这是你学习资料中的写法。项目额外显式传入 `QWEN_API_KEY`，因为项目已经统一使用这个名字，不想再维护一个 `DASHSCOPE_API_KEY`。

### 5.3 为什么 Embedding 不用 `ChatOpenAI`

聊天和向量是两种不同接口：

- Chat：`/chat/completions`；
- Embedding：`/embeddings` 或 DashScope 原生接口。

`DashScopeEmbeddings` 已经帮你把文本转成向量，不需要自己处理返回 JSON。

---

## 6. `rag/`：RAG 的全部关键步骤

这个包负责四件事，内部按职责拆成三个文件：`documents.py` 负责解析和切分，`store.py` 负责 Milvus，`service.py` 负责检索链。

```text
文档解析 -> 文本切分 -> 向量存取 -> 检索/回答链
```

### 6.1 `DocumentParseError`

```python
class DocumentParseError(Exception):
    code: str
    message: str
```

它带有 `code`，索引任务会把它写入任务状态：

```text
MINERU_NOT_CONFIGURED
UNSUPPORTED_FILE_TYPE
MINERU_FAILED
EMPTY_DOCUMENT
TOO_MANY_CHUNKS
```

### 6.2 `extract_markdown()`：为什么用 MinerU

处理规则：

```text
.txt / .md / .markdown -> 直接读取
.pdf / .doc / .docx / 图片 -> MinerU
其他 -> 报错
```

MinerU 的实际调用：

```python
client = MinerU(token=settings.mineru_api_key.get_secret_value())
result = client.extract(
    str(local_path),
    model=settings.mineru_model,
    ocr=True,
    timeout=settings.mineru_timeout_seconds,
)
markdown = result.markdown
```

流程细节：

1. 先把上传内容写入临时目录；
2. 调用 MinerU；
3. 取 `result.markdown`；
4. 删除临时目录；
5. 如果结果为空，抛出 `DocumentParseError`。

### 6.3 为什么有 `_mineru_proxy_context()`

开发机上可能设置了：

```text
HTTP_PROXY
HTTPS_PROXY
```

MinerU 上传接口能通，但结果 ZIP 下载接口可能被代理 TLS 握手拦截。这个上下文管理器只对 MinerU 调用临时移除代理变量，结束后恢复，不影响：

- Spring Boot 内部调用；
- Qwen 模型调用；
- 其他 HTTP 请求。

### 6.4 `split_markdown()`：先按标题，再按长度

```python
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
```

举例：

```markdown
# 龙门石窟

## 游览建议

建议游玩三到四小时。
```

会得到带元数据的 `Document`：

```python
{
    "h1": "龙门石窟",
    "h2": "游览建议",
    "document_id": "18",
    "document_version": 1,
    "chunk_index": 0,
    "title": "龙门石窟讲解",
    "section_path": "龙门石窟 > 游览建议",
    "enabled": True,
}
```

这些元数据有两个用途：

1. Milvus 过滤：只查 `enabled == true`；
2. 前端来源卡片：显示来源标题、章节和片段。

### 6.5 `create_vector_store()`：Milvus 是怎么接进来的

```python
return Milvus(
    embedding_function=embedding_function,
    collection_name=...,
    connection_args=...,
    index_params={...HNSW...},
    search_params={...},
    auto_id=False,
    primary_field="pk",
    text_field="text",
    vector_field="vector",
    enable_dynamic_field=True,
)
```

解释：

- `embedding_function`：告诉 Milvus 如何把文本转成向量；
- `auto_id=False`：主键由项目自己生成，便于覆盖旧版本；
- `primary_field="pk"`：Milvus 主键字段；
- `text_field="text"`：保存原文；
- `vector_field="vector"`：保存向量；
- `enable_dynamic_field=True`：允许 `document_id`、`title` 等业务元数据进入 Milvus。

### 6.6 `RAGService.retrieve()`

Agent 的知识库工具调用它：

```python
pairs = await asyncio.wait_for(
    asyncio.to_thread(
        self.vector_store.similarity_search_with_score,
        query,
        k,
        expr="enabled == true",
    ),
    timeout=self.settings.milvus_search_timeout_seconds,
)
```

为什么要 `asyncio.to_thread`：

- Milvus 的 LangChain 封装里有同步方法；
- FastAPI 运行在事件循环里；
- 同步网络调用会卡住整个服务；
- 放到线程里执行，才不会阻塞其他请求。

返回结果会加：

```python
document.metadata["_normalized_score"] = score
document.metadata["rank_no"] = rank_no
document.metadata["citation_marker"] = f"[S{rank_no}]"
```

### 6.7 `create_rag_chain()`：保留标准 RAG 示例

```python
combine_documents = create_stuff_documents_chain(chat_model, prompt)
return create_retrieval_chain(retriever, combine_documents)
```

`create_retrieval_chain` 的工作是：

1. 用 `retriever` 根据用户问题检索文档；
2. 把文档塞进 `{context}`；
3. 调用聊天模型；
4. 返回 `answer + context`。

Agent 实际使用的是 `RAGService.retrieve()`，因为 Agent 需要自己决定什么时候检索，而不是每次问答都强行先检索。

---

## 7. `springboot.py`：Python 如何读取 Java 的业务数据

### 7.1 为什么不直接查 MySQL

如果 Python 直接连 MySQL：

- Python 和 Java 都会修改业务数据；
- 两套 ORM/表结构会产生不一致；
- 用户权限和发布状态无法统一；
- 事务边界混乱。

所以所有业务数据通过 Java 内部接口读取：

```text
GET /internal/v1/ai/spots/search
GET /internal/v1/ai/spots/{id}
GET /internal/v1/ai/guides/search
GET /internal/v1/ai/foods/search
POST /internal/v1/ai/recommendations
POST /internal/v1/ai/knowledge/chunks/batch
```

### 7.2 `_request()` 做了什么

1. 过滤 `None` 参数；
2. 加上 `X-Internal-Token`；
3. 加上 `X-Internal-Request-Id`（每请求新建 ULID）；
4. GET 遇到 5xx 最多重试一次；
5. 解析 Java 的统一响应；
6. 如果 `code != 0`，抛 `SpringBootError`。

### 7.3 为什么工具不直接抛异常给模型

工具层会把 `SpringBootError` 转成：

```json
{
  "ok": false,
  "errorCode": "SPRINGBOOT_TIMEOUT",
  "message": "业务数据服务暂时不可用",
  "retryable": true
}
```

模型看到这个结构化结果后，可以选择换工具或告诉用户服务暂时不可用，而不是得到一段无法理解的 Python traceback。

---

## 8. `tools/`：Agent 的工具箱

本包按职责拆成四个文件：`context.py` 管请求上下文和预算，`business.py` 放五个 Java 业务工具，`knowledge.py` 放 RAG 工具，`mcp.py` 放 12306 MCP。`__init__.py` 只汇总 `LOCAL_TOOLS`。

### 8.1 `RequestContext`

一次聊天请求可能调用多个工具，每个工具都需要：

- 当前 `user_id`；
- 当前 `thread_id`；
- Spring Boot 客户端；
- RAG 服务；
- 本轮已登记来源；
- 已调用工具次数。

这些数据不能放到全局变量，否则两个用户同时聊天会串数据。项目使用 `ContextVar`：

```python
_current_context: ContextVar[RequestContext | None]
```

请求开始时：

```python
token = set_request_context(context)
```

请求结束时：

```python
reset_request_context(token)
```

这样每个异步请求拥有自己的上下文。

### 8.2 工具预算

```python
def reserve_tool_call(self, name: str) -> bool:
    if self.tool_call_count >= self.max_tool_calls:
        return False
    self.tool_call_count += 1
    self.tool_names.append(name)
    return True
```

如果没有这个限制，模型可能在两个工具之间来回调用，导致超时或产生大量费用。

### 8.3 六个工具

| 工具名 | 数据来源 | 作用 |
|---|---|---|
| `search_scenic_spots` | Java | 搜索景点 |
| `get_spot_detail` | Java | 景点详情 |
| `search_travel_guides` | Java | 攻略和路线 |
| `search_foods` | Java | 美食 |
| `retrieve_tourism_knowledge` | Milvus | 历史文化知识 |
| `recommend_spots` | Java | 确定性推荐候选 |

每个工具的固定结构：

```python
context, error = begin_tool("工具名")
if error is not None:
    return error
try:
    调用 Java 或 RAG
    把结果登记为 SourceItem
    返回 {ok, data, meta}
except Exception:
    返回结构化错误
```

### 8.4 来源登记

以景点为例：

```python
marker = context.add_business_item("SPOT", response)
item["marker"] = f"[{marker}]"
```

假设这是本轮第一个来源，工具返回：

```json
{
  "name": "龙门石窟",
  "marker": "[S1]"
}
```

模型看到 `marker` 后，回答时写：

```text
龙门石窟是世界文化遗产。[S1]
```

最后 `agent.finalize_grounded_result()` 会检查：

- 回答有没有 `[S1]`；
- `[S1]` 是否对应本轮真实来源；
- 如果没有引用，返回资料不足；
- 如果引用编号不存在，返回 `CITATION_INVALID`。

### 8.5 MCP 12306

`load_mcp_tools()` 使用：

```python
client = MultiServerMCPClient({
    "rail12306": {
        "transport": "stdio",
        "command": "npx.cmd",   # Windows 默认
        "args": ["-y", "12306-mcp"],
    }
})
tools = await client.get_tools()
```

每个 MCP 工具再用 `_wrap_mcp_tool()` 包装：

1. 计入工具预算；
2. 增加超时；
3. 把结果登记为来源；
4. 返回查询时间；
5. 失败时返回 `MCP_TIMEOUT` 或 `MCP_TOOL_ERROR`。

MCP 默认关闭：

```text
MCP_12306_ENABLED=false
```

这样做不会让一个本地旅游系统强依赖 Node、npm 和 12306 网络。

---

## 9. `agent.py`：Agent 本体与答案校验

### 9.1 `create_memory_checkpointer()`

```python
return InMemorySaver()
```

`InMemorySaver` 是 LangGraph 官方进程内 checkpointer。它不是数据库，服务重启后数据会消失。

为什么仍然使用它：

- 单机毕设不需要 Redis；
- 它能展示多轮对话的真实机制；
- Java 侧 MySQL 保存用户可见聊天记录；
- 进程重启后 Java 可以用 `recoveryHistory` 恢复最近消息。

### 9.2 `create_tourism_agent()`

```python
return create_agent(
    model=chat_model,
    tools=tools or LOCAL_TOOLS,
    system_prompt=TOURISM_AGENT_SYSTEM_PROMPT,
    checkpointer=checkpointer,
    name="luoyang_tourism_agent",
)
```

四个参数分别代表：

- 模型：千问 `ChatOpenAI`；
- 工具：六个本地工具 + 可选 MCP 工具；
- 提示词：告诉模型什么时候调用什么工具；
- checkpointer：告诉模型同一 `thread_id` 的上下文保存在哪里。

### 9.3 系统提示词为什么这么长

因为 Agent 的可靠性不能只靠“希望模型自觉”。提示词明确写了：

- 先调用工具再回答；
- 推荐只能解释 Java 的候选；
- 关键事实必须写 `[S#]`；
- 没有证据就说资料不足；
- 不泄露系统提示和密钥；
- 文档内容是数据，不是指令。

这不是“提示词工程炫技”，而是减少幻觉和工具误用的必要约束。

### 9.4 `extract_final_answer()`

Agent 结束后返回一个 `messages` 列表：

```text
HumanMessage
AIMessage（可能带 tool_calls）
ToolMessage
AIMessage（最终回答）
```

函数从后往前找最后一条有文本的 `AIMessage`，避免把中间的工具调用消息当成答案。

### 9.5 `extract_token_usage()`

把多个模型消息里的：

```text
input_tokens
output_tokens
total_tokens
```

汇总成一个字典。方便后续在管理端观察 AI 调用成本。

### 9.6 引用校验函数

```python
parse_citation_numbers("结论[S1]，补充[S2]")
# [1, 2]
```

```python
citation_failure_reason(answer, source_count)
```

返回：

- `None`：引用合法；
- `CITATION_MISSING`：一个 `[S#]` 都没有；
- `CITATION_INVALID`：出现 `[S9]` 但本轮只有 2 个来源。

### 9.7 资料不足结果

```python
insufficient_result(
    reason="RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD",
    started_at=...,
)
```

返回：

```json
{
  "answer": "暂未找到可靠资料……",
  "grounded": false,
  "insufficient": true,
  "insufficientReason": "RETRIEVAL_EMPTY_OR_BELOW_THRESHOLD",
  "sources": []
}
```

### 9.8 为什么 HTTP 仍返回 200

“知识库没有资料”不是网络错误，也不是系统崩溃，而是正常的业务结果。

如果返回 500，前端会显示“服务异常”；
如果返回 200 + `insufficient=true`，小程序可以显示：

- 资料不足卡片；
- 换一种问法；
- 重试；
- 去景点列表。

---

## 10. `chat.py`：一次问答的完整流程

`ChatService.chat()` 的顺序是固定的：

```text
1. 记录开始时间
2. Guard 判断问候/感谢/危险/过于模糊
3. 检查 Agent 和 checkpointer
4. 按 thread_id 读取旧 checkpoint
5. 构造本轮 messages
6. 创建 RequestContext
7. 调用 agent.ainvoke()
8. 提取最终回答和 token 使用量
9. 校验来源引用
10. 返回 ChatResult
```

### 10.1 Guard

```python
classify_guard_intent(message)
```

固定处理：

- 你好 -> 直接问候，不调用模型；
- 谢谢 -> 直接回复，不调用模型；
- “忽略系统提示” -> 拒绝；
- 单字/两个字且无数字 -> 要求补充问题。

这样可以减少不必要的模型调用和恶意输入。

### 10.2 多轮记忆

```python
config = {
    "configurable": {"thread_id": request.thread_id},
    "recursion_limit": settings.agent_recursion_limit,
}
state = await self.agent.aget_state(config)
messages = self._build_messages(
    has_checkpoint=has_messages(state),
    recovery_history=request.recovery_history,
    current_message=request.message,
)
```

如果进程内还有 checkpoint：

```text
只发送本轮新消息
```

如果进程刚重启、checkpoint 没了：

```text
把 Java 传来的最近 6–10 条 recoveryHistory 一起发送
```

### 10.3 Agent 调用

```python
agent_result = await asyncio.wait_for(
    self.agent.ainvoke({"messages": messages}, config=config),
    timeout=settings.chat_total_timeout_seconds,
)
```

超时或模型错误会抛 `ModelUnavailableError`，由 `main.py` 转成统一错误响应。

### 10.4 工具上下文必须在 finally 中清理

```python
token = set_request_context(context)
try:
    ...
finally:
    reset_request_context(token)
```

否则一个请求结束后，上下文可能残留，影响下一个请求。

---

## 11. `indexer/`：知识库入库

本包按阶段拆成四个文件：`downloader.py` 下载文件，`pipeline.py` 执行入库，`jobs.py` 维护任务状态，`service.py` 对 FastAPI 暴露稳定入口。

### 11.1 为什么用后台任务

上传 PDF/DOCX 后，MinerU 解析和 Embedding 可能需要几十秒到几分钟。
如果 HTTP 请求一直等，Spring Boot 会超时。

所以流程是：

```text
POST /index/jobs -> 立即返回 jobId（202 Accepted）
后台 asyncio.Task 执行
GET /index/jobs/{jobId} -> 查询进度
```

### 11.2 `DocumentDownloader`

从 Spring Boot 传来的 OSS 临时签名 URL 下载文件：

```python
async with self._client.stream("GET", url) as response:
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > settings.download_max_bytes:
            raise DownloadError(...)
```

使用流式读取，避免一次性把超大文件全部塞进内存。`download_max_bytes` 默认 50 MB。

### 11.3 `IndexPipeline.run()`

```text
10%  开始
25%  下载完成
      MinerU / 文本读取
      Markdown 切分
40%  分片完成
      Milvus 写入
75%  Milvus 写入完成
      分批回调 Java knowledge_chunk
95%  回调完成、清理旧版本
100% 任务成功
```

MinerU 是同步函数，所以这样调用：

```python
markdown = await asyncio.to_thread(
    extract_markdown,
    downloaded.content,
    file_name,
    settings,
)
```

### 11.4 稳定 `vector_id`

```python
hashlib.sha256(f"{document_id}:{version}:{chunk_index}".encode()).hexdigest()
```

同一个文档、同一个版本、同一个分片编号永远得到同一个 ID。
这样重复提交索引任务时，可以覆盖旧向量，不会无限增加重复数据。

### 11.5 `_upsert()`

```python
try:
    delete(ids=ids)
except collection_not_found:
    忽略
add_documents(documents, ids=ids)
```

第一次创建 collection 时，collection 还不存在，所以删除可能报 “collection not found”。
第一次直接交给 `add_documents` 创建；以后再删除旧主键覆盖。

### 11.6 `_delete_old_versions()`

```python
expr = f'document_id == "{safe_id}" and document_version != {version}'
```

新版本文档索引成功后，删除同文档的旧版本向量，避免检索到过期内容。

### 11.7 `_notify_chunks()`

FastAPI 不直接写 MySQL。它把 chunk 信息分批回传给 Java：

```python
await self.springboot_client.upsert_knowledge_chunks(
    batch,
    idempotency_key=f"{request.idempotency_key}:chunks:{start}",
)
```

Java 负责把这些记录写入 `knowledge_chunk` 表。

### 11.8 `IndexJobManager`

保存三类内存数据：

```python
self._jobs: dict[str, IndexJobView]
self._idempotency: dict[str, tuple[fingerprint, job_id]]
self._latest_versions: dict[document_id, version]
```

作用：

- 相同 `idempotencyKey` 返回同一个任务；
- 相同 key 但参数不同 -> 409 冲突；
- 旧版本不能覆盖新版本；
- 任务状态从 `QUEUED -> RUNNING -> SUCCEEDED/FAILED`。

服务重启后这些状态会丢，这是单机毕设的明确取舍。

---

## 12. `runtime.py`：启动时组装对象

应用启动时依次创建：

```text
ChatOpenAI
DashScopeEmbeddings
Milvus VectorStore
RAGService
SpringBootClient
IndexService
可选 MCP tools
InMemorySaver
create_agent
ChatService
```

### 12.1 为什么允许 Milvus 初始化失败

```python
try:
    vector_store = await asyncio.wait_for(
        asyncio.to_thread(create_vector_store, embeddings, settings=settings),
        timeout=15,
    )
except Exception:
    logger.exception("milvus_vector_store_initialization_failed")
```

如果 Milvus 没启动：

- FastAPI 仍然启动；
- `/health/ready` 返回 `not_ready`；
- 本地景点工具仍然可用；
- 知识库检索工具会返回 `MILVUS_UNAVAILABLE`。

这比整个进程直接退出更容易排查。

### 12.2 MCP 失败不能拖垮主流程

```python
try:
    mcp_tools = await load_mcp_tools(settings)
except Exception:
    logger.exception("mcp_tools_initialization_failed")
```

MCP 是可选增强，不是主链路。

### 12.3 `ApplicationRuntime.close()`

应用关闭时：

- 关闭 ChatService；
- 取消索引后台任务；
- 关闭 SpringBootClient 的 httpx 连接池。

---

## 13. `main.py`：FastAPI 接口层

### 13.1 `lifespan`

FastAPI 启动时执行 `build_runtime()`，把对象放进：

```python
app.state.runtime
app.state.index_service
app.state.chat_service
```

关闭时执行 `runtime.close()`。

### 13.2 七个路由

#### 聊天

```python
@internal.post("/chat", response_model=ApiEnvelope[ChatResult])
async def chat(payload, request, service):
    result = await service.chat(payload, _request_id(request))
    return ApiEnvelope.ok(result, request_id=payload.request_id)
```

#### 删除 checkpoint

```text
POST /internal/v1/ai/checkpoints/{thread_id}/reset
```

先用正则检查是不是 26 位 ULID，再调用 `adelete_thread()`。

#### 提交索引

```text
POST /internal/v1/ai/index/jobs
```

返回 202 Accepted，而不是 200。因为任务只是“已接收”，还没有完成。

#### 查询索引

```text
GET /internal/v1/ai/index/jobs/{job_id}
```

#### 删除向量

```text
POST /internal/v1/ai/index/documents/{document_id}/delete
```

#### 健康检查

```text
GET /health/live   进程是否活着
GET /health/ready  Milvus + Agent 是否可用
```

### 13.3 `X-Internal-Token` 在哪里校验

```python
internal = APIRouter(
    prefix=INTERNAL_API_PREFIX,
    dependencies=[Depends(verify_internal_token)],
)
```

放在 router 上，表示这个 router 下所有内部接口自动校验，不需要每个函数重复写。

### 13.4 异常处理器

三种：

1. `ServiceError`：业务异常，按 `http_status` 返回；
2. `RequestValidationError`：参数错误，返回 400；
3. `Exception`：未预料异常，返回 500，同时记录堆栈。

---

## 14. 一次 AI 问答的完整数据流

用户在小程序输入：

```text
龙门石窟适合玩多久？
```

### 第一步：小程序 -> Spring Boot

- 小程序带用户 JWT；
- Spring Boot 校验 JWT、会话归属、限流；
- 从 MySQL 读取 `threadId` 和 `recoveryHistory`。

### 第二步：Spring Boot -> FastAPI

```http
POST /internal/v1/ai/chat
X-Internal-Token: ...
Content-Type: application/json
```

```json
{
  "threadId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "userId": "10001",
  "message": "龙门石窟适合玩多久？",
  "clientMessageId": "client-message-001",
  "requestId": "request-001",
  "recoveryHistory": []
}
```

### 第三步：FastAPI

1. 校验 token；
2. `ChatService` 判断不是问候/危险/过短；
3. 用 `threadId` 读取 checkpoint；
4. 构造 `RequestContext`；
5. Agent 决定调用 `retrieve_tourism_knowledge`；
6. RAG 检索 Milvus；
7. 文档登记为 `[S1]`；
8. 模型回答。

### 第四步：返回

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": "龙门石窟建议游玩三到四小时。[S1]",
    "grounded": true,
    "insufficient": false,
    "sources": [
      {
        "sourceType": "KNOWLEDGE",
        "sourceId": "18",
        "chunkId": "vec-18-0",
        "title": "龙门石窟历史文化与主要洞窟讲解",
        "snippet": "建议游玩三到四小时……",
        "score": 0.88,
        "rankNo": 1
      }
    ],
    "suggestedQuestions": ["这个景点附近还有什么值得去？"],
    "modelName": "qwen3.8-flash",
    "latencyMs": 1200
  },
  "traceId": "request-001",
  "timestamp": "..."
}
```

### 第五步：Spring Boot

- 把用户消息和助手消息写入 MySQL；
- 把来源写入 `ai_message_source`；
- 返回给小程序。

### 第六步：小程序

- 显示回答；
- 显示来源卡片；
- 如果 `insufficient=true`，显示资料不足卡片。

---

## 15. 一次文档索引的完整数据流

```text
管理端上传 PDF
  -> Spring Boot 保存文件到 OSS
  -> 生成短时签名 URL
  -> POST /internal/v1/ai/index/jobs
  -> FastAPI 返回 jobId
  -> 后台下载
  -> MinerU 解析 Markdown
  -> MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
  -> DashScopeEmbeddings 生成向量
  -> Milvus add_documents
  -> FastAPI 分批回调 Java
  -> Java 写 knowledge_chunk
  -> 任务状态 SUCCEEDED
```

### 为什么要有 `KnowledgeChunkPayload`

Milvus 只保存向量和部分元数据，Java 还需要知道：

- 哪个文档；
- 哪个版本；
- 第几个分片；
- 分片内容；
- 内容 hash；
- 字符数。

`KnowledgeChunkPayload` 就是 FastAPI 回调 Java 时的标准格式。

---

## 16. MCP 12306 的完整数据流

```text
用户：明天从洛阳到西安还有高铁票吗？
  -> Agent 看到 MCP 工具
  -> 调用 get-current-date
  -> 调用 get-station-code-by-names
  -> 调用 get-tickets
  -> 包装层登记 [S1]
  -> 模型回答时写 [S1]
  -> 前端显示来源卡片
```

当前来源类型仍然使用 `KNOWLEDGE`，标题写成：

```text
12306 实时查询（get-tickets）
```

摘要会带：

```text
查询时间：2026-11-24T15:00:00+08:00，实时数据仅供参考
```

### 为什么默认关闭

因为 MCP 需要：

- Node.js；
- npx；
- 网络；
- 第三方 12306 接口。

这些都不是本地 RAG 问答的必要条件。需要演示时改：

```text
MCP_12306_ENABLED=true
```

---

## 17. 调试时应该看哪里

### 17.1 启动失败

先看：

```text
uv run python -c "from app.config import get_settings; get_settings()"
```

常见原因：

- `.env` 缺 `QWEN_API_KEY`；
- `INTERNAL_TOKEN` 为空；
- 端口/URL 格式不对；
- `CHAT_TOTAL_TIMEOUT_SECONDS >= 30`。

### 17.2 `/health/ready` 返回 503

看 `checks`：

```json
{"status": "not_ready", "checks": {"milvus": false, "agent": true}}
```

说明 Milvus 没连上，检查：

```powershell
Test-NetConnection 192.168.100.128 -Port 19530
```

### 17.3 Agent 回答“资料不足”

依次检查：

1. 工具有没有被调用；
2. 工具结果有没有 marker；
3. 回答有没有 `[S1]`；
4. RAG 是否真的返回了文档；
5. Milvus collection 是否是 `v2`；
6. 文档是否已经重新索引。

### 17.4 RAG 检索不到

检查：

- 是否使用了新的 `text-embedding-v4`；
- 是否重建了 `luoyang_tourism_knowledge_v2`；
- 文档 metadata 是否有 `enabled=true`；
- Milvus 是否可达。

### 17.5 MinerU 失败

检查：

- `.env` 是否有 `MINERU_API_KEY`；
- 文件后缀是否在支持列表；
- `MINERU_BYPASS_PROXY=true`；
- 网络是否能访问 MinerU。

---

## 18. 测试说明

```powershell
cd D:\luoyang-smart-tourism\ai-service
uv run ruff check app tests
uv run mypy app
uv run pytest -q
```

默认测试结果：

```text
25 passed, 5 skipped
```

被跳过的是真实外部依赖测试。需要时逐个打开：

```powershell
$env:RUN_INFRA_TESTS='1'; uv run pytest tests/integration/test_infrastructure_health.py tests/integration/test_milvus_rag.py -q
$env:RUN_QWEN_SMOKE='1'; uv run pytest tests/integration/test_qwen_smoke.py -q
$env:RUN_MINERU_SMOKE='1'; uv run pytest tests/integration/test_mineru_smoke.py -q
$env:RUN_MCP_TESTS='1'; uv run pytest tests/integration/test_mcp_12306.py -q
```

### 测试文件怎么看

| 文件 | 验证内容 |
|---|---|
| `tests/test_config_schemas.py` | 配置、camelCase、未知字段 |
| `tests/test_rag.py` | Markdown 读取、切分、RAG 链、检索分数 |
| `tests/test_tools.py` | 工具调用、来源登记、预算 |
| `tests/test_agent.py` | create_agent、checkpointer、引用校验 |
| `tests/test_chat.py` | 一次问答、恢复历史、问候 |
| `tests/test_indexer.py` | 幂等、冲突、索引流水线 |
| `tests/test_contract.py` | OpenAPI 和内部接口契约 |
| `tests/test_security.py` | 内部 token |

---

## 19. 如何新增一个工具

假设你要新增“附近停车场查询”。

### 第一步：在 Spring Boot 增加内部只读接口

例如：

```text
GET /internal/v1/ai/parking/nearby?spotId=1001
```

### 第二步：在 `SpringBootClient` 增加方法

```python
async def nearby_parking(self, spot_id: str) -> Any:
    return await self._request(
        "GET",
        f"/internal/v1/ai/parking/nearby",
        params={"spotId": spot_id},
        retry_get=True,
    )
```

### 第三步：在 `tools/business.py` 增加输入模型和工具

```python
class ParkingInput(BaseModel):
    spot_id: str = Field(min_length=1, max_length=64)

@tool("nearby_parking", args_schema=ParkingInput)
async def nearby_parking(spot_id: str) -> dict[str, Any]:
    context, error = begin_tool("nearby_parking")
    if error:
        return error
    response = await context.springboot_client.nearby_parking(spot_id)
    marker = context.add_business_item("SPOT", response)
    return {"ok": True, "data": [response], "marker": f"[{marker}]"}
```

### 第四步：加入白名单

```python
LOCAL_TOOLS = [..., nearby_parking]
```

### 第五步：增加测试

至少测试：

- 工具成功；
- Java 返回空；
- Java 超时；
- 工具预算耗尽；
- 来源 marker 登记。

---

## 20. 如何更换 Embedding 模型

不要直接改 `.env` 后继续用旧 collection。

正确步骤：

1. 修改 `QWEN_EMBEDDING_MODEL`；
2. 新建 collection，例如 `luoyang_tourism_knowledge_v3`；
3. 重新上传/重新索引全部知识文档；
4. 验证 `/health/ready` 和 RAG 检索；
5. 删除旧 collection。

原因：不同模型生成的向量不在同一个语义空间，哪怕维度相同也不能混用。

---

## 21. 最后用一句话总结

> **Spring Boot 管业务，FastAPI 管 AI；Agent 选择工具，工具负责取证，引用校验防止胡说，MinerU 负责复杂文档，Milvus 负责语义检索，InMemorySaver 负责多轮上下文。**

如果你能沿着下面这条链讲清楚，就掌握了本项目 AI 服务：

```text
main.py
  -> chat.py
    -> agent.py
      -> tools/business.py / tools/knowledge.py
        -> springboot.py / rag/service.py
          -> Milvus / Spring Boot / 12306 MCP
    -> agent.py 引用校验
  -> ChatResult
```

---

## 22. 推荐阅读顺序与断点

### 第一次阅读

1. 本文第 0 节；
2. `config.py`；
3. `schemas.py`；
4. `llm.py`；
5. `rag/documents.py` 中 `split_markdown`，以及 `rag/service.py` 中 `RAGService.retrieve`；
6. `tools/context.py` 中 `RequestContext`，以及 `tools/knowledge.py` 中 `retrieve_tourism_knowledge`；
7. `agent.py` 中 `create_tourism_agent`；
8. `chat.py` 中 `ChatService.chat`。

### 调试断点

建议在以下位置打断点：

- `main.py::chat`：确认 Java 请求是否到达；
- `chat.py::ChatService.chat`：确认 Guard 和 threadId；
- `agent.py::create_tourism_agent`：确认工具列表；
- `tools/context.py::begin_tool`：确认工具预算；
- `tools/knowledge.py::retrieve_tourism_knowledge`：确认检索结果；
- `agent.py::finalize_grounded_result`：确认引用是否通过；
- `indexer/pipeline.py::IndexPipeline.run`：确认每个索引进度。

### 答辩时建议展示

1. 启动 FastAPI，打开 `/docs`；
2. 打开 `/health/live`；
3. 让小程序问一个景点问题；
4. 展示回答中的 `[S1]` 和来源卡片；
5. 问一个知识库没有的问题，展示资料不足卡片；
6. 上传一份 PDF，展示索引进度；
7. 如果环境允许，再展示 12306 MCP 查票。
