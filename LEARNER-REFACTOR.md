# AI 服务学习友好化重构方案

> 项目：基于 Spring Boot 的洛阳景点智能推荐管理系统
> 范围：`D:\luoyang-smart-tourism\ai-service`
> 目标：保留毕业设计需要的 AI 问答、RAG、Milvus、Agent、工具调用、索引任务和 MCP 能力，同时把代码压缩到本科生能够逐文件阅读和答辩讲解的程度。
> 原则：**保留功能，减少包装；保留工作量，减少企业化基础设施。**

---

## 1. 先说结论

第一版 `ai-service` 按企业项目的方式拆成了 13 个子包（`api`、`services`、`agents`、`tools`、`integrations`、`chains`、`rag`、`schemas` 等）。一次普通问答要在八九个目录之间跳转，读者很难回答“这个请求到底经过了哪些代码”。

后来先做过一次“全部平铺”，把代码压成 12 个同级 `.py` 文件。功能虽然清楚了，但 `tools.py`、`rag.py`、`indexer.py` 各自包含多个技术动作，单个文件又变成几百行，仍然不好阅读。

因此最终采用折中方案：**只保留 3 个有意义、边界稳定的包，其余模块继续平铺。**

```text
最终结构 = 3 个包 + 9 个平铺模块

tools/      工具上下文、业务工具、RAG 工具、MCP 工具
rag/        文档解析、Milvus、检索链
indexer/    下载、索引流水线、任务表、服务门面

config.py / errors.py / schemas.py / llm.py
springboot.py / agent.py / chat.py / runtime.py / main.py
```

这不是恢复旧版 13 个包，而是只对“确实有多个文件、确实需要分组”的三块知识做分包。**包的存在是为了减少单个文件长度，不是为了增加架构层级。**

---

## 2. 最终目录

```text
ai-service/
├─ app/
│  ├─ __init__.py
│  ├─ config.py              配置、常量、日志、ULID、内部 token
│  ├─ errors.py              统一业务异常
│  ├─ schemas.py             所有 Pydantic 请求/响应模型
│  ├─ llm.py                 ChatOpenAI + DashScopeEmbeddings
│  ├─ springboot.py          Java 内部接口客户端
│  ├─ agent.py               create_agent、提示词、引用校验
│  ├─ chat.py                一次问答的主流程
│  ├─ runtime.py             启动时组装所有对象
│  ├─ main.py                FastAPI 路由和异常处理
│  ├─ tools/
│  │  ├─ __init__.py         统一导出 LOCAL_TOOLS 和工具上下文
│  │  ├─ context.py          RequestContext、工具预算、来源登记
│  │  ├─ business.py         景点/攻略/美食/推荐五个业务工具
│  │  ├─ knowledge.py        RAG 知识库检索工具
│  │  └─ mcp.py              12306 MCP 加载、超时和来源包装
│  ├─ rag/
│  │  ├─ __init__.py         统一导出 RAG API
│  │  ├─ documents.py        TXT/Markdown、MinerU、标题和长度切分
│  │  ├─ store.py            Milvus 连接、collection、健康检查
│  │  └─ service.py          retrieve + create_retrieval_chain
│  └─ indexer/
│     ├─ __init__.py         统一导出 IndexService 及其依赖
│     ├─ downloader.py       OSS 临时 URL 下载
│     ├─ pipeline.py         下载 -> MinerU -> 切分 -> Milvus -> Java
│     ├─ jobs.py             进程内任务表、幂等键、状态机
│     └─ service.py          提供给 FastAPI 的索引入口
├─ tests/
├─ .env.example
├─ pyproject.toml
├─ README.md
├─ LEARNER-REFACTOR.md       本文：为什么这样拆
└─ CODE-GUIDE.md             逐文件代码讲解
```

打开 `app/` 后，你只需要先认出三个文件夹和九个平铺文件。没有 `api/ -> services/ -> agents/ -> chains/` 这样的逐层转发。

---

## 3. 为什么是这 3 个包

### 3.1 `tools/`：工具天然适合按业务分文件

Agent 会调用六类本地工具：

- 查询景点；
- 查询景点详情；
- 查询攻略；
- 查询美食；
- 推荐景点；
- 检索知识库。

另外还要把 12306 MCP 工具包装成统一格式。如果把所有这些内容放在一个 `tools.py` 中，文件会超过 600 行。

拆成 4 个文件后，阅读路径非常直接：

```text
想理解工具如何限制调用次数
  -> tools/context.py

想看景点、美食、推荐怎么查 Java
  -> tools/business.py

想看 RAG 工具怎么检索 Milvus
  -> tools/knowledge.py

想看 12306 MCP 怎么加载
  -> tools/mcp.py
```

`tools/__init__.py` 只负责把 `LOCAL_TOOLS` 汇总出来，`agent.py` 不需要知道工具内部怎么实现。

### 3.2 `rag/`：解析、存储、检索链是三个技术动作

`rag/` 内部分成：

```text
documents.py   文件内容 -> Markdown -> 文本分片
store.py       Embedding + Milvus collection
service.py     查询向量 -> 组装来源 -> create_retrieval_chain
```

这三个动作的输入输出不同，出错后的排查位置也不同：

- MinerU 失败，看 `documents.py`；
- Milvus 连接失败，看 `store.py`；
- 检索结果没有来源，看 `service.py` 和 `tools/knowledge.py`。

如果把这三部分塞回一个 `rag.py`，文件会超过 400 行，而且修改其中的 embedding 时容易误碰文档切分。

### 3.3 `indexer/`：一条后台流水线本来就有多个阶段

索引不是一次普通函数调用，而是一个后台任务：

```text
下载 OSS 文件
  -> MinerU 解析
  -> 切分 Markdown
  -> 写入 Milvus
  -> 把 chunk 回写 Spring Boot
  -> 清理旧版本向量
```

所以自然拆成：

- `downloader.py`：只负责 HTTP 下载和大小限制；
- `pipeline.py`：只负责执行一次索引；
- `jobs.py`：只负责任务状态、幂等和串行调度；
- `service.py`：只负责把能力暴露给 FastAPI。

这样每个文件都能用一句话说明职责，不需要 `manager/factory/helper` 这类空壳层。

### 3.4 其余文件为什么不继续建包

`config.py`、`errors.py`、`schemas.py`、`llm.py`、`springboot.py`、`agent.py`、`chat.py`、`runtime.py`、`main.py` 各自只有一个主要概念。给每个文件再建一个包，只会增加 `__init__.py` 和 import 跳转，不会提高理解速度。

判断标准很简单：

> **一个概念一个文件；多个紧密相关且各自有独立职责的文件，才建一个包。**

---

## 4. 与旧版 13 个包的对应关系

| 旧版目录 | 最终去向 |
|---|---|
| `api/` | 合并到 `main.py` |
| `services/` | 问答流程放进 `chat.py`，索引门面放进 `indexer/service.py` |
| `agents/` | 合并到 `agent.py` |
| `chains/` | RAG 链放进 `rag/service.py`，Agent 编排放进 `agent.py` |
| `rag/` | 保留为有边界的 `rag/` 包，但只保留 3 个实现文件 |
| `tools/` | 保留为有边界的 `tools/` 包，按业务拆成 4 个文件 |
| `integrations/` | Qwen 放进 `llm.py`，Java 客户端放进 `springboot.py`，Milvus 放进 `rag/store.py`，下载放进 `indexer/downloader.py` |
| `ingestion/` | 合并到 `indexer/` |
| `core/` | 配置/错误/ID/日志放进 `config.py`、`errors.py` |
| `schemas/` | 合并到 `schemas.py` |
| `memory/` | checkpointer 创建放进 `agent.py` |
| `prompts/` | 系统提示词放进 `agent.py` |
| `scripts/` | 不再保留一次性脚手架脚本 |

这不是“全删”，而是把每个旧目录中真正服务主流程的代码迁到最近的位置。功能没有减少。

---

## 5. 与 LangChain 学习资料对齐

| 学习资料中的知识点 | 最终实现 |
|---|---|
| `DashScopeEmbeddings(model="text-embedding-v4")` | `llm.py::create_embeddings()` 直接创建官方类 |
| `ChatOpenAI` 兼容千问 | `llm.py::create_chat_model()` |
| `create_agent` | `agent.py::create_tourism_agent()` |
| 短期记忆 `checkpointer` | `agent.py::create_memory_checkpointer()`，使用 `InMemorySaver` |
| `create_retrieval_chain` | `rag/service.py::create_rag_chain()` |
| Milvus 向量库 | `rag/store.py::create_vector_store()` |
| 工具定义 | `tools/business.py`、`tools/knowledge.py` |
| MCP 工具 | `tools/mcp.py::load_mcp_tools()` |
| MinerU 文档解析 | `rag/documents.py::extract_markdown()` |
| FastAPI 接口 | `main.py` |

这里故意没有使用 `AgentExecutor`、手写 Prompt 拼接、手写 HTTP 调用 Qwen，也没有自己实现 Embedding。能交给官方高层 API 的地方都交给高层 API。

---

## 6. 必须保留的功能清单

### 6.1 对外能力

- AI 问答；
- 多轮对话；
- 来源卡片；
- 资料不足时返回结构化降级结果；
- 知识库索引任务；
- 查询索引进度；
- 删除文档向量；
- 健康检查。

### 6.2 Agent 本地工具

- `search_scenic_spots`；
- `get_spot_detail`；
- `search_travel_guides`；
- `search_foods`；
- `retrieve_tourism_knowledge`；
- `recommend_spots`。

### 6.3 RAG 能力

- TXT/Markdown 直接解析；
- PDF/DOCX/图片使用 MinerU；
- Markdown 标题切分 + 长度切分；
- `DashScopeEmbeddings` 写入 Milvus；
- `create_retrieval_chain` 检索；
- 检索来源回填到 `SourceItem`；
- 更换 Embedding 时通过 collection 版本隔离。

### 6.4 会话能力

- Java 生成 `threadId`；
- Python 用 `threadId` 作为 LangGraph `configurable.thread_id`；
- `InMemorySaver` 保存图状态；
- checkpoint 丢失时用 Java 传来的最近 6–10 条历史恢复。

### 6.5 MCP 能力

- 12306 查询默认关闭；
- 开启时通过 `MultiServerMCPClient` 加载；
- 每个 MCP 工具都有超时和调用预算；
- MCP 结果和本地工具一样登记来源；
- MCP 失败不能拖垮本地 RAG 和业务工具。

---

## 7. 为什么不再拆更多文件

当前主要文件规模如下（包含中文注释，行数只是粗略参考）：

| 文件 | 主要职责 | 大致规模 |
|---|---|---:|
| `agent.py` | create_agent、提示词、引用校验 | 约 270 行 |
| `main.py` | FastAPI 路由和异常处理 | 约 280 行 |
| `tools/business.py` | 五个业务工具 | 约 260 行 |
| `tools/context.py` | 上下文和来源登记 | 约 220 行 |
| `schemas.py` | 全部接口模型 | 约 230 行 |
| `indexer/pipeline.py` | 一条完整索引流水线 | 约 170 行 |
| `indexer/jobs.py` | 任务状态和幂等 | 约 140 行 |

这些文件确实有几百行，但它们都是**围绕一个概念连续阅读**的代码。继续把它们拆成 `agent_factory.py`、`prompt_builder.py`、`source_validator.py`、`job_store.py`、`job_runner.py`，会让读者多跳转 5–6 次，却不会让逻辑更简单。

更合适的优化不是继续拆文件，而是：

1. 先读 `CODE-GUIDE.md` 中有编号的章节；
2. 每个文件只抓一个入口函数（例如 `create_tourism_agent()`、`RAGService.retrieve()`、`IndexPipeline.run()`）；
3. 用断点跟着一条真实请求走，不要从第一行读到最后一行；
4. 只有当某个文件出现第二个独立业务概念时，才考虑再拆一个包。

对本科毕设来说，**当前 3 个包是阅读成本和文件长度之间的平衡点**。再拆是过度设计，全部平铺又会重新产生几百行的单文件。

---

## 8. 明确不做的企业化功能

本轮不引入：

- Redis 任务队列；
- 消息队列；
- 多实例分布式锁；
- 数据库任务表；
- HMAC、时间窗和防重放；
- 多级缓存；
- 熔断、限流和服务注册中心；
- 为每个类写接口再写实现；
- 为了“看起来分层”而创建空目录。

单机答辩需要的是：能启动、能演示、能解释、能定位错误。任务状态放进程内、checkpoint 放 `InMemorySaver`，都是明确写进文档的取舍，不是隐藏缺陷。

---

## 9. 验收标准

```powershell
cd D:\luoyang-smart-tourism\ai-service
uv run ruff format app tests
uv run ruff check app tests
uv run mypy app
uv run pytest -q
```

同时人工确认：

- `app/` 下只有 `tools/`、`rag/`、`indexer/` 三个业务包；
- `tools/__init__.py` 能导出 `LOCAL_TOOLS`；
- `rag/__init__.py` 能导出 `RAGService` 和 `create_vector_store`；
- `indexer/__init__.py` 能导出 `IndexService`；
- `/health/ready` 能显示 Milvus 和 Agent 状态；
- `/docs` 仍能看到 7 个内部接口；
- MCP 默认关闭，开启后 `get-tickets` 可以加载；
- 索引任务能完成 `QUEUED -> RUNNING -> SUCCEEDED`。

---

## 10. 答辩时的一句话解释

> 我最初参考企业项目做了 13 个包，但毕业设计的核心是展示 RAG 和 Agent 的完整闭环，而不是模拟大型团队协作。因此最终只保留 `tools`、`rag`、`indexer` 三个确实需要多文件协作的包，其余模块平铺。这样既保留了工程边界，也能让我沿着一条请求独立讲清楚每个文件的作用。