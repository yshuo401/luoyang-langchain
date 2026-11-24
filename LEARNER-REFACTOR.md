# AI 服务学习友好化重构方案

> 项目：基于 Spring Boot 的洛阳景点智能推荐管理系统
> 范围：`D:\luoyang-smart-tourism\ai-service`
> 目标：保留毕业设计需要的 AI 问答、RAG、Milvus、Agent、工具调用、索引任务和 MCP 能力，同时把代码压缩到本科生能够逐文件阅读和答辩讲解的程度。
> 原则：**保留功能，减少包装；保留工作量，减少企业化基础设施。**

---

## 1. 先说结论

当前 `app` 目录下有 13 个代码子包：

```text
agents
api
chains
core
ingestion
integrations
memory
prompts
rag
schemas
scripts
services
tools
```

代码功能是完整的，但阅读路径被切得太碎。一次普通问答，读者需要同时理解：

```text
api -> services -> agents -> tools -> integrations -> chains -> rag -> schemas
```

而且每一层还可能有多个文件、多个抽象类、多个错误转换。对于企业团队，这样做方便多人并行维护；对于本科毕设，它会让“我只是想知道一次问答怎么走完”变得非常困难。

本次重构后的目标结构是：

```text
ai-service/
├─ app/
│  ├─ __init__.py
│  ├─ config.py          # 配置、常量、日志、内部 token
│  ├─ errors.py          # 统一业务异常
│  ├─ schemas.py         # 所有 Pydantic 请求/响应模型
│  ├─ llm.py             # ChatOpenAI + DashScopeEmbeddings
│  ├─ rag.py             # MinerU、切分、Milvus、检索、create_retrieval_chain
│  ├─ springboot.py      # 调用 Spring Boot 内部只读接口
│  ├─ tools.py           # 六个本地工具 + 工具上下文 + MCP 工具合并
│  ├─ agent.py           # create_agent、提示词、来源登记、引用校验、资料不足
│  ├─ chat.py            # 一次聊天从输入到响应的主流程
│  ├─ indexer.py         # 下载 -> MinerU -> 切分 -> Milvus -> Java chunk 回写
│  ├─ runtime.py         # 启动时组装所有对象
│  └─ main.py            # FastAPI 入口与 7 个内部接口
├─ tests/
├─ .env
├─ .env.example
├─ pyproject.toml
├─ README.md
├─ LEARNER-REFACTOR.md   # 本文：为什么改、怎么改
└─ CODE-GUIDE.md         # 重构后的逐文件代码讲解
```

注意：这里仍然是“一个 `app` Python 包”，只是**不再在 `app` 下继续创建 13 个子包**。这些文件都是平级的，打开任何一个都不需要先穿过三层 `__init__.py`。

---

## 2. 为什么 13 个包对毕设过重

### 2.1 当前规模

重构前大致规模：

| 指标 | 数量 |
|---|---:|
| `app` 下子包 | 13 |
| Python 文件 | 约 59 |
| Python 代码行 | 约 2900 |
| 测试文件 | 14 |

代码量本身并不是特别大，真正的问题是“概念层数太多”。例如：

- `tools/` 为了复用几个辅助函数拆成 7 个文件；
- `rag/` 为了解析、切分、检索、向量库拆成 5 个文件；
- `integrations/` 又拆出 Qwen、Milvus、Spring Boot、下载器；
- `schemas/` 拆成 5 个文件；
- `core/` 拆成配置、常量、错误、ID、日志、安全 6 个文件。

这像是一个多人维护的中型项目结构，不像一个学生需要独立讲清楚的毕业设计。

### 2.2 不是所有“标准做法”都适合毕设

企业项目强调：

- 可替换实现；
- 多人并行开发；
- 依赖注入和接口隔离；
- 监控、重试、熔断、任务持久化；
- 为未来扩展预留抽象。

毕设更强调：

- 功能闭环；
- 能演示；
- 能解释；
- 能定位错误；
- 代码量适中；
- 技术点真实存在，不是为了“显得高级”而加。

因此本次不删除功能，而是删除“为不存在的团队和流量准备的抽象”。

---

## 3. 与你提供的 LangChain 学习资料逐项对齐

| 学习资料中的知识点 | 本项目保留方式 | 重构前的问题 | 重构后 |
|---|---|---|---|
| `DashScopeEmbeddings(model="text-embedding-v4")` | 直接使用官方高层类 | 自定义 `QwenMultiModalEmbeddings`，手写适配层 | `llm.py` 中一行模型初始化 |
| `ChatOpenAI` 兼容千问 | 保留 | 包装在 `integrations/qwen.py` | `llm.py` 中集中创建 |
| MinerU 文档解析 | 使用 `mineru-open-sdk` | 手写 PDF、DOCX 解析 | `rag.py` 中调用 MinerU `extract()` |
| `RecursiveCharacterTextSplitter` | 保留 | 被多层 `TextSection` 包装 | 直接切 `Document` |
| `Milvus` 向量库 | 保留 | `Milvus -> KnowledgeVectorStore -> ThresholdRetriever` 三层 | `Milvus` + 少量直接操作函数 |
| `create_retrieval_chain` | 保留 | 包装在 `chains/rag_chain.py` | `rag.py` 中保留一个清晰示例 |
| `create_agent` | 保留 | `agents/` 包内再拆 state 和 agent | `agent.py` 一个文件 |
| `InMemorySaver` | 保留 | `memory/checkpointer.py` | `agent.py` 或直接 `InMemorySaver()` |
| 工具调用 | 保留六个业务工具 | 每个工具一个文件 + `_common.py` | `tools.py` 一个文件 |
| MCP | 新增可选能力 | 原来没有 | `MultiServerMCPClient` + 12306 MCP 可选启用 |
| 多轮对话 | 保留 `thread_id` | 包装在多个服务层 | `chat.py + agent.py` 直接可见 |

---

## 4. 必须保留的功能清单

重构不是删功能。以下功能继续存在：

### 4.1 对外（由 Spring Boot 调用）

1. `POST /internal/v1/ai/chat`：AI 问答；
2. `POST /internal/v1/ai/checkpoints/{threadId}/reset`：清理会话；
3. `POST /internal/v1/ai/index/jobs`：提交知识文档索引；
4. `GET /internal/v1/ai/index/jobs/{jobId}`：查询索引进度；
5. `POST /internal/v1/ai/index/documents/{documentId}/delete`：删除文档向量；
6. `GET /health/live`：进程存活；
7. `GET /health/ready`：Milvus 和 Agent 就绪。

### 4.2 Agent 本地工具

- `search_scenic_spots`：搜索景点；
- `get_spot_detail`：景点详情；
- `search_travel_guides`：攻略查询；
- `search_foods`：美食查询；
- `retrieve_tourism_knowledge`：Milvus 知识检索；
- `recommend_spots`：读取 Java 的确定性推荐结果。

### 4.3 RAG 能力

- Qianwen Embedding；
- Milvus 写入、相似度检索、删除；
- MinerU 解析 PDF、DOCX 等复杂文档；
- Markdown / TXT 直接读取；
- 文档切分；
- 来源追溯；
- 引用标记校验；
- 资料不足时降级。

### 4.4 会话能力

- `threadId` 对应 LangGraph `thread_id`；
- `InMemorySaver` 维护进程内多轮状态；
- Java 传入 `recoveryHistory` 时恢复上下文；
- 删除 checkpoint。

### 4.5 MCP 能力

- 使用 `MultiServerMCPClient`；
- 支持 12306 购票查询 MCP；
- MCP 失败不影响本地工具和普通 RAG；
- MCP 结果必须有查询时间与“实时数据仅供参考”的提示。

---

## 5. 核心技术决策

### 5.1 Embedding 改成 `DashScopeEmbeddings`

学习资料中的写法是：

```python
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(model="text-embedding-v4")
```

项目内部改用同样风格的封装：

```python
def create_embeddings(settings: Settings) -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model=settings.qwen_embedding_model,
        dashscope_api_key=settings.qwen_api_key.get_secret_value(),
    )
```

这样做的原因：

- 不再手写 `MultiModalEmbedding.call`；
- 不再自己排序返回结果；
- 不再手写维度校验；
- `text-embedding-v4` 是标准文本向量模型，适合知识库问答；
- LangChain 的 `Embeddings` 接口已经由官方社区包实现。

### 5.2 为什么必须换 Milvus collection

Embedding 模型变了，向量空间就变了。即使维度同为 1024，也不能保证旧向量和新查询向量可比较。因此：

```text
旧：luoyang_tourism_knowledge_v1
新：luoyang_tourism_knowledge_v2
```

更换后需要重新索引已有知识文档。这是正确的数据迁移行为，不是 bug。

### 5.3 MinerU 的统一文档入口

处理规则：

| 文件类型 | 解析方式 |
|---|---|
| `.txt` / `.md` / `.markdown` | 直接读取并清洗，避免无意义调用云服务 |
| `.pdf` / `.docx` / 图片等 | 保存临时文件，调用 MinerU `extract()`，取回 Markdown |
| 其他类型 | 明确报错 `UNSUPPORTED_FILE_TYPE` |

MinerU SDK 的核心用法：

```python
from mineru import MinerU

client = MinerU(token=settings.mineru_api_key.get_secret_value())
result = client.extract(str(temp_path), ocr=True)
markdown = result.markdown
```

MinerU 返回 Markdown 后，再交给 `MarkdownHeaderTextSplitter` 和 `RecursiveCharacterTextSplitter`。这样 PDF 解析不再由项目维护，项目只负责“拿到 Markdown 后如何切分和入库”。

### 5.4 Chat 仍然使用 `ChatOpenAI`

你之前明确要求千问通过 OpenAI 兼容接口接入，因此保留：

```python
ChatOpenAI(
    model=settings.qwen_chat_model,
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)
```

Embedding 用 `DashScopeEmbeddings`，Chat 用 `ChatOpenAI`，两者职责不同，不矛盾。

### 5.5 RAG 链保留高层 API

`rag.py` 中保留：

```python
combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
```

Agent 的 `retrieve_tourism_knowledge` 工具直接调用 `RAGService.retrieve()`，避免为了“取文档”再让模型生成一次答案。`create_retrieval_chain` 作为独立 RAG 演示入口保留，确保论文中讲的链路在代码里真实存在。

### 5.6 Agent 使用 `create_agent`

```python
agent = create_agent(
    model=chat_model,
    tools=local_tools + mcp_tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
```

多轮对话：

```python
config = {"configurable": {"thread_id": thread_id}}
await agent.ainvoke({"messages": [...]}, config=config)
```

### 5.7 MCP 12306 采用可选启用

已验证可用的开源项目：

- 仓库：`Joooook/12306-mcp`
- npm 包：`12306-mcp`
- 传输：stdio
- 可用工具包括：
  - `get-current-date`
  - `get-station-code-by-names`
  - `get-tickets`
  - `get-interline-tickets`
  - `get-train-route-stations`

调用方式：

```python
client = MultiServerMCPClient({
    "rail12306": {
        "transport": "stdio",
        "command": "npx.cmd",   # Windows
        "args": ["-y", "12306-mcp"],
    }
})
tools = await client.get_tools()
```

原因与限制：

- 12306 是第三方实时数据，不能保证任何时候都可用；
- 启动 MCP 会下载/启动 Node 进程，因此默认关闭；
- 设置 `MCP_12306_ENABLED=true` 后才合并 MCP 工具；
- MCP 失败时只记录日志，继续使用本地六个工具；
- 回答中必须提示“实时数据仅供参考”，并展示查询时间。

### 5.8 MCP 结果的来源处理

现有 Java 和前端只认识五种来源：

```text
KNOWLEDGE / SPOT / GUIDE / FOOD / RECOMMENDATION
```

为了不修改 Java 和小程序，MCP 结果暂时登记为 `KNOWLEDGE`，但标题写成：

```text
12306 实时查询（get-tickets）
```

并在摘要中附带查询时间。这样：

- 不破坏现有 API；
- MCP 结果仍可进入 `[S#]` 引用校验；
- 前端可以正常显示来源卡片；
- 后续若要扩展 `MCP` 类型，再统一改 Java 和小程序即可。

---

## 6. 重构后的实际调用链

### 6.1 AI 问答

```text
Spring Boot
  -> POST /internal/v1/ai/chat
  -> main.py 校验 X-Internal-Token
  -> chat.py::ChatService.chat()
  -> 固定问候/危险问题拦截
  -> agent.py::create_agent 已创建好的 Agent
  -> 模型决定调用工具
  -> tools.py 中的六个业务工具
  -> SpringBootClient 调用 Java 内部接口
  -> 或 RAGService 调用 Milvus
  -> agent.py 校验 [S#]
  -> ChatResult 返回 Spring Boot
```

### 6.2 文档索引

```text
Spring Boot
  -> POST /internal/v1/ai/index/jobs
  -> indexer.py 创建后台任务
  -> 下载 OSS 临时签名文件
  -> MinerU / 文本读取
  -> MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
  -> DashScopeEmbeddings
  -> Milvus add_documents
  -> 分批回写 Java knowledge_chunk 元数据
  -> 查询任务状态
```

### 6.3 MCP 12306

```text
Agent
  -> 12306 MCP Tool
  -> npx -y 12306-mcp 子进程
  -> 12306 官方查询接口
  -> 工具包装层登记来源和查询时间
  -> Agent 生成回答
```

---

## 7. 文件级职责表

| 文件 | 职责 | 预计代码量 |
|---|---|---:|
| `app/config.py` | 环境变量、常量、日志、ULID、内部 token | 150 行左右 |
| `app/errors.py` | 业务异常与错误码 | 100 行左右 |
| `app/schemas.py` | 所有请求/响应模型 | 180 行左右 |
| `app/llm.py` | ChatOpenAI 和 DashScopeEmbeddings | 50 行左右 |
| `app/rag.py` | MinerU、切分、Milvus、检索、RAG 链 | 430 行左右（含注释） |
| `app/springboot.py` | Java 内部接口客户端 | 180 行左右 |
| `app/tools.py` | 六个工具、工具预算、MCP wrapper | 640 行左右（工具部分可逐个读） |
| `app/agent.py` | prompt、create_agent、来源和引用校验 | 270 行左右 |
| `app/chat.py` | 聊天主流程、固定回答、checkpoint 重置 | 150 行左右 |
| `app/indexer.py` | 下载、任务状态、索引流水线 | 390 行左右 |
| `app/runtime.py` | 启动装配 | 100 行左右 |
| `app/main.py` | FastAPI 入口和 7 个接口 | 220 行左右 |

全部文件合计约 3100 行，其中包含大量面向初学者的中文注释。行数不是唯一目标：更重要的是不再有 13 个子包和层层包装，核心调用链可以直接顺着文件阅读。

---

## 8. 13 个包的去向

| 原目录 | 处理方式 | 去向 |
|---|---|---|
| `agents/` | 合并 | `agent.py` |
| `api/` | 合并 | `main.py` |
| `chains/` | 合并 | `rag.py` + `agent.py` |
| `core/` | 合并 | `config.py` + `errors.py` |
| `ingestion/` | 合并 | `indexer.py` |
| `integrations/` | 拆分精简 | `llm.py` + `springboot.py` + `rag.py` |
| `memory/` | 删除包装 | 直接 `InMemorySaver()` |
| `prompts/` | 合并 | `agent.py` |
| `rag/` | 合并 | `rag.py` |
| `schemas/` | 合并 | `schemas.py` |
| `scripts/` | 删除 | 不再保留兼容旧 API 的脚本 |
| `services/` | 合并 | `chat.py` + `indexer.py` |
| `tools/` | 合并 | `tools.py` |

---

## 9. 明确不做的企业化功能

本次继续不引入：

- Redis 缓存或 Redis checkpointer；
- HMAC 签名、时间窗、防重放；
- 多 Agent 协作；
- Middleware 链；
- Runtime Context 框架；
- 复杂任务队列；
- 数据库保存索引任务；
- 多租户；
- 生产级限流；
- 模型结果缓存。

这些能力可以学习和了解，但没有必要塞进一个单机演示系统。

---

## 10. 迁移步骤

1. 写本文档；
2. 新增平铺模块，不先删除旧目录；
3. 修改 `runtime.py`，让新模块完成原有装配；
4. 修改测试 import；
5. 运行 `ruff`、`mypy`、`pytest`；
6. 验证 OpenAPI 仍包含全部内部接口；
7. 验证资料不足、引用校验、多轮对话和索引任务；
8. 验证 MinerU 文本路径和 DashScopeEmbeddings 初始化；
9. 删除旧子包；
10. 更新 README 和 CODE-GUIDE。

---

## 11. 验收标准

- [ ] `app/` 下不再有 13 个子包；
- [ ] 所有核心文件可直接打开阅读；
- [ ] `create_agent` 仍然用于 Agent；
- [ ] `InMemorySaver` 仍然用于多轮记忆；
- [ ] `create_retrieval_chain` 仍然存在；
- [ ] Embedding 使用 `DashScopeEmbeddings`；
- [ ] Chat 使用 `ChatOpenAI`；
- [ ] PDF/DOCX 解析使用 MinerU；
- [ ] 六个本地工具全部存在；
- [ ] MCP 12306 可选启用，失败可降级；
- [ ] 引用校验和资料不足逻辑保留；
- [ ] 内部 `X-Internal-Token` 鉴权保留；
- [ ] 原有七个内部接口路径不变；
- [ ] 测试和静态检查通过。

---

## 12. 给答辩时的解释

如果老师问“你为什么这样拆分”，可以这样回答：

> 我最初参考企业项目做了较细的分层，但在实际开发中发现，毕业设计的核心是展示 RAG 和 Agent 的完整闭环，而不是模拟大型团队协作。因此我把同一业务链路上的代码合并到平级模块：`rag.py` 负责知识入库和检索，`agent.py` 负责 Agent 编排和来源校验，`tools.py` 负责业务工具，`chat.py` 负责一次问答。这样既保留了 LangChain 的官方高层 API，也能让我独立解释每一行的作用。

这个回答比“为了简单所以删了很多”更准确，因为功能没有被删除，删除的是与单机毕设不匹配的抽象层。
