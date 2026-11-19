# AI 服务（`ai-service/`）

> 状态：M6–M7 核心实现已完成（2026-11-19）：知识解析/切分/Milvus 索引、阈值 RAG、`create_retrieval_chain`、`create_agent` 工具编排、请求级证据登记、引用校验、Redis/InMemory checkpointer、HMAC 内部接口和资料不足降级均已落地并通过自动测试。
> **Redis 状态（2026-11-19）**：VM 已升级为 `redis:8.2`（`8.2.11`），entrypoint 自动加载 ReJSON/search/bf/timeseries；checkpointer 固定 DB 0 + `langgraph:checkpoint:` 前缀，`checkpointer: READY` 和真实多轮/reset 均已通过。单进程本地调试仍可显式使用 `CHECKPOINTER_BACKEND=memory`，该模式禁止用于生产。
> 技术栈：Python 3.14.2 + uv 0.13.7、FastAPI、Pydantic v2、LangChain、LangGraph、Milvus、Redis checkpointer、阿里云千问（Qwen）。
> 本地监听：`127.0.0.1:8000`，**只允许 Spring Boot 内网调用**。
> 系统角色：旅游知识 RAG、工具型 Agent 编排、短期记忆、来源追溯和“资料不足”降级；不是第二业务后端。

## 1. 开发前门禁（必须阅读）

修改本目录任何代码之前，按顺序阅读：

1. [项目总入口](../README.md)：四工程边界、端口和全局硬约束。
2. [系统架构](../docs/architecture/SYSTEM-ARCHITECTURE.md)：FastAPI 与 Java 的调用方向、Nginx 禁止代理范围。
3. [AI 服务设计](../docs/ai/AI-SERVICE-DESIGN.md)：Qwen、Milvus、RAG、Agent、checkpointer 和降级细节。
4. [API 契约](../docs/api/API-CONTRACT.md)：内部 HMAC、聊天、索引和来源响应字段。
5. [数据库设计](../docs/database/DATABASE-DESIGN.md)：AI 会话、消息、知识库表的所有权边界。
6. [开发规范](../docs/development/DEVELOPMENT-STANDARD.md)：命名、版本锁定、测试、日志和 DoD。
7. [实施路线图](../docs/roadmap/IMPLEMENTATION-ROADMAP.md)：M6–M7 的交付顺序。

开工前自检：

- [ ] 我已确认本改动属于 AI 能力层，不把用户 JWT、管理员权限或业务 CRUD 放进 FastAPI。
- [ ] 我已确认 FastAPI 不直连业务 MySQL，不直接读写 `scenic_spot`、`travel_guide`、`food` 等表。
- [ ] 我已确认聊天通过 `ChatOpenAI` 调用 `qwen3.8-flash`；`qwen3-vl-embedding` 通过 DashScope 官方多模态 Embedding SDK 调用，不手写 HTTP JSON。
- [ ] 我已确认 LangChain 使用锁定版本验证过的 `create_agent`、checkpointer、`create_retrieval_chain`。
- [ ] 我已确认 `threadId` 只由 Java 创建，FastAPI 不自行生成、不信任客户端透传。
- [ ] 我已确认检索不足时不调用生成模型，返回结构化 `insufficient=true`。
- [ ] 我已确认不把真实 API Key、用户聊天全文、OSS 签名 URL 或内部 Prompt 写入日志和仓库。

## 2. 模块职责与边界

### 2.1 本模块负责

| 能力 | 说明 |
|---|---|
| 知识库入库 | 解析 PDF/DOCX/Markdown/TXT，中文友好切分，Qwen Embedding，写入 Milvus |
| RAG 问答 | `create_retrieval_chain` 组合 Retriever、Prompt 和 ChatOpenAI，返回带来源的回答 |
| Agent 编排 | `create_agent` 注册白名单工具，按问题自动调度知识库和业务数据查询 |
| 短期记忆 | 通过 Redis checkpointer 按 `thread_id` 维持同一会话的多轮图状态 |
| 来源追溯 | 从检索结果和工具结果构造 `sources`，校验 `rankNo`、`snippet`、`deepLink` |
| 降级防止幻觉 | 低于阈值、无证据、工具失败或引用无效时返回“资料不足”分支 |
| 内部协议 | 提供 FastAPI OpenAPI、HMAC 校验、requestId 和统一内部响应信封 |
| 健康检查 | `/health/live`、`/health/ready`，区分进程存活和依赖就绪 |

### 2.2 本模块不负责

- 不签发、刷新或验证用户/管理员 JWT；不判断会话归属和 RBAC。
- 不保存用户可见聊天事实；事实源是 Java 的 `ai_conversation`、`ai_message`、`ai_message_source`。
- 不直接操作业务 MySQL、OSS AccessKey、微信 AppSecret。
- 不向前端暴露 Swagger、Milvus、Redis、模型 Key 或内部 HMAC secret。
- 不把 Milvus 当业务数据库，不用向量检索替代 Java 的发布状态和数据权限过滤。
- 不把模型生成的排序当作推荐结果；推荐候选和顺序以 Java 的确定性算法为准。
- 不在第一版实现模型微调、任意 SQL/Shell/HTTP 工具或 token 级流式输出。

## 3. 调用拓扑

```text
admin-web / user-wx
        │  只请求 /api/v1/**
        ▼
Spring Boot :8080  ── JWT、RBAC、限流、会话归属、MySQL 事实持久化
        │
        │  HMAC + requestId + 内部 JSON
        ▼
FastAPI :8000（仅 127.0.0.1）
        ├─ ChatOpenAI ──> qwen3.8-flash / OpenAI-compatible endpoint
        ├─ QwenMultiModalEmbeddings ──> qwen3-vl-embedding / DashScope SDK
        ├─ create_agent + Redis checkpointer
        ├─ create_retrieval_chain + Milvus
        └─ 只读工具回调 ──> Spring Boot /internal/v1/ai/**
                               （已发布景点、攻略、美食、推荐）
```

硬规则：

1. Nginx 只反代 Spring Boot，不为 FastAPI 配置 `location`。
2. FastAPI 不接收小程序或管理端 JWT；只接收 Java 已校验的会话上下文。
3. AI 服务需要业务摘要时通过 `integrations/springboot_client.py` 回调 Java，不能直连业务库。
4. Java 的 `/internal/v1/ai/**` 只读接口只返回已发布、未删除、当前可见的数据。
5. `threadId` 与 MySQL `ai_conversation.thread_id` 同值；checkpoint 与用户可见消息是两套职责。
6. 任何模型调用都受总超时、并发信号量和工具预算约束。

## 4. 目标目录结构

```text
ai-service/
├─ README.md
├─ pyproject.toml                 # 直接依赖、Python 3.14 约束、pytest/ruff/mypy 配置
├─ uv.lock                        # 锁定实际安装版本，提交到仓库
├─ .env.example                   # 只含占位值，不含真实 Key
├─ app/
│  ├─ main.py                     # FastAPI 实例、lifespan、异常处理、路由注册
│  ├─ api/
│  │  ├─ health.py                # /health/live、/health/ready
│  │  ├─ internal_chat.py         # /internal/v1/ai/chat、checkpoint reset
│  │  └─ internal_index.py        # /internal/v1/ai/index/**
│  ├─ core/
│  │  ├─ config.py                # pydantic-settings 环境变量解析
│  │  ├─ logging.py               # 结构化 JSON 日志、requestId/threadId、脱敏和耗时
│  │  ├─ security.py              # HMAC、时间窗、常量时间比较、重放防护
│  │  ├─ errors.py                # 内部错误码、异常映射、脱敏信封
│  │  └─ constants.py             # 工具预算、角色、来源类型等常量
│  ├─ integrations/
│  │  ├─ qwen.py                  # ChatOpenAI + QwenMultiModalEmbeddings 工厂
│  │  ├─ milvus.py                # Milvus VectorStore/collection 工厂
│  │  └─ springboot_client.py     # httpx 客户端 + HMAC 只读业务查询
│  ├─ agents/
│  │  ├─ tourism_agent.py         # create_agent、工具白名单、Agent 构建
│  │  └─ state.py                 # Agent 状态和恢复上下文定义
│  ├─ chains/
│  │  ├─ rag_chain.py             # create_retrieval_chain、docs chain
│  │  └─ grounding.py             # 阈值、引用校验、来源构造、结构化结果
│  ├─ tools/
│  │  ├─ spot_tools.py            # search_scenic_spots、get_spot_detail
│  │  ├─ guide_tools.py           # search_travel_guides
│  │  ├─ food_tools.py            # search_foods
│  │  ├─ knowledge_tools.py       # retrieve_tourism_knowledge
│  │  └─ recommendation_tools.py  # recommend_spots（只解释 Java 结果）
│  ├─ rag/
│  │  ├─ loaders.py               # PDF/DOCX/Markdown/TXT 解析
│  │  ├─ splitter.py              # RecursiveCharacterTextSplitter 中文切分
│  │  ├─ embeddings.py            # QwenMultiModalEmbeddings 导出
│  │  ├─ vector_store.py          # Milvus 写入、删除、集合生命周期
│  │  └─ retriever.py             # topK、score threshold、过滤已禁用文档
│  ├─ ingestion/
│  │  ├─ pipeline.py              # 解析→切分→向量化→Milvus→回写元数据
│  │  └─ jobs.py                  # 任务幂等、进度、重试、取消和清理
│  ├─ memory/
│  │  └─ checkpointer.py          # Redis checkpointer 创建、setup、健康检查、关闭
│  ├─ schemas/
│  │  ├─ chat.py                  # 聊天请求/响应/恢复上下文的 Pydantic 模型
│  │  ├─ index.py                 # 索引任务请求、进度和错误模型
│  │  └─ source.py                # 来源对象、引用和展示字段
│  ├─ prompts/
│  │  ├─ system.py                # Agent System Prompt
│  │  ├─ rag.py                   # RAG Prompt
│  │  └─ guard.py                 # 资料不足、注入攻击和输出约束
│  ├─ services/
│  │  ├─ chat_service.py          # 一次聊天用例的完整编排
│  │  └─ index_service.py         # 一次索引用例的完整编排
│  └─ scripts/
│     ├─ check_langchain_api.py   # 顶层 LangChain API 导入冒烟
│     └─ check_redis_checkpointer.py # RedisJSON/RediSearch + RedisSaver.setup 诊断
└─ tests/
   ├─ unit/                       # 阈值、引用、HMAC、Pydantic、工具参数
   ├─ contract/                   # OpenAPI 快照、字段名、错误码、跨语言签名向量
   └─ integration/                # Redis、Milvus、Java 工具和 Qwen 冒烟
```

职责原则：

- `api/` 只做协议转换、HMAC 入口和响应映射，不写 Agent/RAG 编排。
- `services/` 组织用例；`chains/`、`agents/`、`tools/` 是可组合能力。
- `integrations/` 是唯一发起外部网络调用的位置；其他模块不各自创建 HTTP 客户端。
- `schemas/` 是 FastAPI OpenAPI 和内部契约的事实源；接口变更先改模型和测试。
- `prompts/` 集中管理提示词，禁止在 Controller 或 Tool 里拼接大段 Prompt。

## 5. 初始启动顺序

### 5.1 前置条件

1. 先在 `deploy/` 启动 MySQL、Redis、Milvus、Nginx（见 [deploy/README.md](../deploy/README.md)）。
2. 在 IDEA 启动 `springboot/`，确认 `http://127.0.0.1:8080` 可访问且 Flyway 已完成。
3. 配置 Qwen 聊天兼容端点、`qwen3-vl-embedding` 官方 SDK 依赖和本地环境变量；真实 Key 只放在未提交的 `.env` 或操作系统环境中。
4. Python 已固定为 3.14.2，使用 `uv 0.13.7` 管理虚拟环境和锁文件；`.python-version` 与 `uv.lock` 必须提交。

### 5.2 初始化后运行

当前工作区已完成初始化并落地 RAG/Agent 代码：`pyproject.toml`、`uv.lock`、`.python-version`、`.venv`、`.env` 和 `app/` 均存在。新克隆仓库时执行：

```powershell
cd D:\luoyang-smart-tourism\ai-service
Copy-Item .env.example .env
# 只在本机编辑 .env，填入真实的 QWEN_API_KEY 和 INTERNAL_SHARED_SECRET；不得提交
uv sync --all-groups
uv run ruff check .
uv run mypy app
uv run pytest
uv run python -m app.scripts.check_langchain_api
# 当前 Redis 8.2.11 应输出 checkpointer: READY
uv run python -m app.scripts.check_redis_checkpointer
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

当前已注册的正式路由：

- `POST /internal/v1/ai/chat`
- `POST /internal/v1/ai/checkpoints/{threadId}/reset`
- `POST /internal/v1/ai/index/jobs`
- `GET /internal/v1/ai/index/jobs/{jobId}`
- `POST /internal/v1/ai/index/documents/{documentId}/delete`
- `GET /health/live`
- `GET /health/ready`
- `GET /docs`、`GET /openapi.json`

所有 `/internal/v1/ai/**` 路由均要求 HMAC-SHA256、时间窗和请求 ID，不能通过 Nginx 暴露。

> **Redis 验证结果**：`langgraph-checkpoint-redis 0.6.0` 依赖 RedisJSON 和 RediSearch。当前 VM 为 Redis 8.2.11，模块完整；checkpointer 使用 DB 0 + 前缀，`RedisSaver.setup()` 已输出 `checkpointer: READY`，真实多轮消息与 reset 测试均通过。

另开终端执行测试：

```powershell
cd D:\luoyang-smart-tourism\ai-service
uv run pytest
# 真实 Redis + Milvus（要求 VM 中间件已启动）
$env:RUN_INFRA_TESTS='1'; uv run pytest tests/integration/test_infrastructure_health.py
# 真实 Qwen Chat + qwen3-vl-embedding
$env:RUN_QWEN_SMOKE='1'; uv run pytest tests/integration/test_qwen_smoke.py
```

初始化时已验证的版本（以 `uv.lock` 为最终事实源）：

| 组件 | 版本 |
|---|---|
| Python | 3.14.2 |
| uv | 0.13.7 |
| FastAPI | 0.163.1 |
| LangChain | 1.5.0 |
| LangChain Core | 1.8.2 |
| LangChain OpenAI | 1.8.0 |
| LangGraph | 1.2.22 |
| langgraph-checkpoint-redis | 0.6.0 |
| langchain-milvus | 0.4.0 |
| DashScope SDK | 2.1.0 |
| PyMilvus | 3.0.3 |

> 表中 `redis` Python SDK 8.1.0 只是客户端库，不代表 Redis 服务器版本。checkpoint 目标服务器已固定为 `redis:8.2`（8.2.11，含 RedisJSON/RediSearch）；RedisSearch 索引固定使用 DB 0。

Swagger 与健康检查：

- Swagger UI：`http://127.0.0.1:8000/docs`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`
- 存活检查：`http://127.0.0.1:8000/health/live`
- 就绪检查：`http://127.0.0.1:8000/health/ready`

`--reload` 只用于本地开发，不用于演示或生产。FastAPI 必须绑定 `127.0.0.1`，不得绑定 `0.0.0.0` 后暴露到公网。启动后先验证 `/health/live`，再验证 Redis 与 Milvus 的 `/health/ready`，最后做一次 Java → FastAPI 的内部冒烟。

### 5.3 从空目录复现本工程

本工程已按以下顺序实际初始化；以后重建环境时直接复用，不要手工创建 `requirements.txt` 或 `.venv`：

```powershell
cd D:\luoyang-smart-tourism\ai-service
uv init --bare --name luoyang-ai-service --python 3.14 --vcs none .
uv python pin 3.14.2

uv add fastapi "uvicorn[standard]" pydantic-settings python-dotenv httpx `
  langchain langchain-openai langchain-classic langchain-milvus `
  langchain-text-splitters langchain-community langgraph `
  langgraph-checkpoint-redis redis pymilvus pypdf python-docx dashscope

uv add --dev pytest pytest-asyncio pytest-cov ruff mypy
uv sync --all-groups
```

初始化约定：

- `--vcs none` 只避免在子工程再建一个 Git 仓库；版本控制由工作区根仓库统一管理。
- `.python-version=3.14.2`、`pyproject.toml`、`uv.lock` 必须提交；`.venv`、`.env`、缓存不提交。
- 新依赖统一使用 `uv add` / `uv add --dev`，禁止手工编辑 `uv.lock`。
- 当前依赖实际安装成功，Python 3.14.2 无需降级。
## 6. 环境变量规范

`.env.example` 只保留变量名和占位值。统一使用下列命名，禁止在代码中新增 `DASHSCOPE_API_KEY`、`EMBEDDING_DIM`、`MODEL_TIMEOUT_SECONDS` 或 `MILVUS_METRIC` 等别名：

```dotenv
APP_ENV=local
LOG_LEVEL=INFO
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000

# Qwen：聊天走 OpenAI-compatible；qwen3-vl-embedding 走 DashScope 原生多模态 SDK
QWEN_API_KEY=replace-me
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.8-flash
QWEN_EMBEDDING_MODEL=qwen3-vl-embedding
QWEN_EMBEDDING_DIM=1024
QWEN_REQUEST_TIMEOUT_SECONDS=25
QWEN_MAX_RETRIES=2
MODEL_MAX_CONCURRENCY=4

# Milvus
MILVUS_URI=http://192.168.100.128:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=luoyang_tourism_knowledge_v1
MILVUS_METRIC_TYPE=COSINE

# Redis / LangGraph checkpoint（服务器必须为 Redis 8.2 + RedisJSON + RediSearch）
REDIS_URL=redis://192.168.100.128:6379/0
CHECKPOINT_KEY_PREFIX=langgraph:checkpoint:
CHECKPOINT_TTL_SECONDS=604800

# Java internal service
SPRINGBOOT_INTERNAL_BASE_URL=http://127.0.0.1:8080
INTERNAL_SHARED_SECRET=replace-with-long-random-secret
INTERNAL_TIMESTAMP_TOLERANCE_SECONDS=300

# RAG / Agent budget
RAG_TOP_K=5
RAG_MIN_SCORE=<calibrated-value>
RAG_MAX_CONTEXT_CHARS=8000
AGENT_MAX_TOOL_CALLS=6
AGENT_RECURSION_LIMIT=12
TOOL_TIMEOUT_SECONDS=10
CHAT_TOTAL_TIMEOUT_SECONDS=28

# Indexing
INDEX_MAX_CHUNKS=2000
EMBEDDING_BATCH_SIZE=16
```

配置规则：

- QWEN_BASE_URL 与模型名以阿里云百炼控制台当前地域/工作空间提供的配置为准；当前锁定聊天 qwen3.8-flash、向量 qwen3-vl-embedding。
- `qwen3-vl-embedding` 不支持 OpenAI-compatible Embeddings，禁止传给 `OpenAIEmbeddings`；统一使用官方 SDK 适配器。
- `QWEN_EMBEDDING_DIM` 已用真实调用验证为 `1024`，且必须与 Milvus collection 的 `dim` 一致；换 embedding 模型必须重建 collection，不得混写。
- `RAG_MIN_SCORE` 必须用本地 30–50 条问题评估后校准；未校准前不得直接上线。
- `REDIS_URL` 固定使用 DB 0；不要擅自切到 DB 1，因为 RedisSearch 不支持在非 0 DB 建索引。通过 `CHECKPOINT_KEY_PREFIX=langgraph:checkpoint:` 与 Java 缓存隔离。
- Redis 服务器已满足 `RedisSaver` 的 RedisJSON/RediSearch 前置条件；当前 VM 为 Redis 8.2.11，checkpointer 使用 DB 0 + 前缀并通过 `checkpointer: READY`。
- 运行 `uv run python -m app.scripts.check_redis_checkpointer`；只有输出 `checkpointer: READY` 才能进入多轮 Agent 验收。
- `CHECKPOINT_TTL_SECONDS` 与 Java 会话保留策略保持一致。
- `MODEL_MAX_CONCURRENCY` 是模型并发上限；超过上限时短等待后返回可重试错误，不无限排队。
- `CHAT_TOTAL_TIMEOUT_SECONDS` 必须小于 Java 侧 30 秒预算，为网络和持久化留余量。
- `INTERNAL_SHARED_SECRET` 开发与演示使用不同值，只存在于环境变量，不写入 Markdown、日志或前端。
- `.env`、真实 Key、数据库密码、OSS AccessKey、微信 AppSecret 不得提交到版本库。
- 本机 `.env` 已写入真实 Qwen Key，但 Markdown、源码和测试快照只保留占位值；该 Key 曾在聊天会话中出现，建议在联调完成后到百炼控制台轮换。

## 7. Qwen 接入规范

### 7.1 聊天模型（`qwen3.8-flash`）

统一使用 `langchain_openai.ChatOpenAI`，由官方客户端处理鉴权、重试、超时和响应解析：

```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model=settings.qwen_chat_model,
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
    temperature=0.1,
    timeout=settings.qwen_request_timeout_seconds,
    max_retries=settings.qwen_max_retries,
)
```

要求：

- 不在业务代码中手写 OpenAI/Qwen HTTP JSON。
- 旅游事实问答使用低温度（建议 0–0.2），不允许用高随机参数“润色”事实。
- 结构化输出优先使用官方 `with_structured_output(PydanticModel)`；模型/端点是否支持由锁定版本和真实冒烟确认。
- 不在日志中记录完整 Prompt、用户原文或 `api_key`。
- 模型名称、端点和密钥不写死在 Python 源文件中。

### 7.2 向量模型（`qwen3-vl-embedding`）

`qwen3-vl-embedding` 官方不支持 OpenAI-compatible Embeddings，禁止传给 `OpenAIEmbeddings`。项目统一使用 `app/integrations/qwen.py` 中的 `QwenMultiModalEmbeddings`；它封装 DashScope 官方 `MultiModalEmbedding` 高层接口，并实现 LangChain `Embeddings` 契约。

```python
from app.integrations.qwen import create_embedding_model

embedding_model = create_embedding_model()
vector = embedding_model.embed_query("洛阳龙门石窟")
images = embedding_model.embed_image_urls(["https://example.com/longmen.jpg"])
```

已用真实 API 验证：

1. 请求 `QWEN_EMBEDDING_DIM=1024` 返回 1024 维向量。
2. 官方支持维度：2560、2048、1536、1024、768、512、256。
3. 批量大小不超过服务端限制；当前环境变量起点为 `EMBEDDING_BATCH_SIZE=16`。
4. Milvus collection 的 `dim` 与 metric 必须与锁定 embedding 模型一致。
5. 更换模型或维度时新建 collection 或执行可回滚重建，禁止混写。

### 7.3 版本与导入冒烟

LangChain 生态的导入路径会随版本调整，项目必须锁定完整依赖并在 CI/本地执行导入测试：

- `langchain`、`langchain-openai`、`langchain-milvus`、`langgraph`、Redis checkpointer 一起锁定。
- `tests/contract/test_langchain_api.py` 和 `uv run python -m app.scripts.check_langchain_api` 验证 `create_agent`、checkpointer、`create_retrieval_chain` 可用。
- `uv run python -m app.scripts.check_redis_checkpointer` 单独验证 RedisJSON/RediSearch 和 `RedisSaver.setup()`；它不能替代 LangChain 导入测试。
- 升级依赖必须单独提交，并执行 RAG 回归、Agent 多轮测试和真实 Qwen 冒烟。
- 不得为了兼容旧代码退回 `AgentExecutor`、手写 ReAct 文本解析或自行实现工具循环。

## 8. 知识库入库链路（RAG）

### 8.1 端到端流程

```text
管理端上传文档
  -> Spring Boot 校验类型/大小/权限并上传 OSS
  -> Java 写入 file_object、knowledge_document、ai_index_job
  -> Java 生成短时签名下载 URL，调用 FastAPI /internal/v1/ai/index/jobs
  -> FastAPI 下载、解析、切分，生成稳定 chunk 元数据
  -> QwenMultiModalEmbeddings 批量生成向量
  -> Milvus upsert（document_id + document_version + chunk_index 唯一）
  -> FastAPI 回调 Java /internal/v1/ai/knowledge/chunks/batch
  -> Java 写 knowledge_chunk，返回任务成功和 chunkCount
  -> 管理端展示索引进度、失败原因和重试按钮
```

AI 服务不直连业务 MySQL。`knowledge_chunk` 的持久化通过 Java 内部接口完成；FastAPI 只保留必要的运行状态和向量。

### 8.2 文档解析与切分

支持 PDF、DOCX、Markdown、TXT。解析失败必须返回可分类错误，例如 `UNSUPPORTED_FILE_TYPE`、`PARSE_ERROR`、`EMPTY_DOCUMENT`、`TOO_MANY_CHUNKS`，错误信息不能包含服务器绝对路径或密钥。

中文友好切分建议起点：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", "；", "，", " "],
    keep_separator=True,
)
```

每个 chunk 的 metadata 至少包含：

- `document_id`
- `document_version`
- `chunk_index`
- `title`
- `category`
- `source_url`
- `page`
- `section_path`
- `enabled`

索引任务必须支持幂等键、同键不同参数冲突、旧版本不覆盖新版本、失败重试和进度更新。重复提交的同一任务不能重复生成向量或产生重复 chunk。

### 8.3 Milvus collection

第一版固定 collection 名称 `luoyang_tourism_knowledge_v1`，metric 使用 `COSINE`。集合至少保存：

- 主键 `vector_id`（由文档 ID、版本和 chunk 索引稳定生成）
- `embedding` 向量
- `document_id`、`document_version`、`chunk_index`
- `title`、`category`、`source_url`、`page`、`section_path`
- `content`、`content_hash`
- `enabled`

删除文档时必须先删除/失效 Milvus 向量，再通知 Java；失败时记录补偿任务。不同 embedding 模型不得写入同一 collection；维度不一致时直接拒绝启动索引，不做静默截断。

## 9. RAG 链与来源追溯

### 9.1 高层 API 组合

RAG 统一在 `app/chains/rag_chain.py` 封装，使用官方链式组件：

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "用户问题：{input}\n\n检索资料：\n{context}"),
])
combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = await rag_chain.ainvoke({"input": question})
answer = result["answer"]
documents = result.get("context", [])
```

不得退回 `RetrievalQA` 等旧式链，也不得在业务代码中手写“请求 embedding、拼 Prompt、手发 HTTP”的底层流程。

### 9.2 检索阈值与资料不足

流程：

1. 先执行 Milvus 检索，过滤 `enabled=false`、旧版本和非发布来源。
2. 按 `RAG_TOP_K` 取回候选，取最高相关度与 `RAG_MIN_SCORE` 比较。
3. 结果为空、低于阈值或全部被过滤时，**不调用生成模型**。
4. 返回 HTTP 200、业务成功信封和 `insufficient=true`，并给出 `insufficientReason`。
5. 前端展示 `insufficient-card`，提供换问法、查看景点列表、重试三个动作。
6. 有证据时，来源由服务端从检索结果/工具结果构造，模型只能使用现有 `[S1]`、`[S2]` 标记。

来源对象至少包含 `sourceType`、`sourceId`、`chunkId`、`title`、`snippet`、`score`、`rankNo` 和 `deepLink`。来源不能由模型自由编造；引用编号必须能在本轮来源表中找到，否则进入降级或移除无效引用后重新校验。

### 9.3 生成与校验

- Prompt 必须明确“只根据资料回答”“不确定就说资料不足”“不编造开放时间、票价、电话、地址”。
- `answer` 中的引用编号必须和 `sources.rankNo` 对齐。
- `grounded=true` 仅在至少存在一条有效可展示来源时为真。
- 对结构性推荐问题，优先调用 Java 的确定性推荐工具；AI 只能解释候选和排序，不得新增或重排景点。
- 生成失败不得把模型堆栈返回 Java/前端；返回脱敏的内部错误码和可重试标记。

## 10. Agent 与多轮记忆

### 10.1 `create_agent` 装配

Agent 在 FastAPI `lifespan` 中只构建一次，使用同一实例、不同 `thread_id` 隔离状态：

```python
from langchain.agents import create_agent

tourism_agent = create_agent(
    model=chat_model,
    tools=TOOLS,
    system_prompt=TOURISM_AGENT_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
```

调用时：

```python
config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": settings.agent_recursion_limit,
}
result = await tourism_agent.ainvoke(
    {"messages": [{"role": "user", "content": message}]},
    config=config,
)
```

锁定版本验收：`create_agent` 可从顶层包导入、能接收 `system_prompt`/`tools`/`checkpointer`、同一实例可并发服务多个 thread、真实 Qwen 至少完成一次工具调用。禁止使用旧式 `AgentExecutor` 作为主实现。

### 10.2 工具白名单

工具必须用 LangChain `@tool` 和 Pydantic 参数模型声明，参数名清晰，返回值只含白名单字段：

| 工具 | 用途 | Java 内部接口 |
|---|---|---|
| `search_scenic_spots` | 搜索已发布景点摘要 | `GET /internal/v1/ai/spots/search` |
| `get_spot_detail` | 获取单个景点摘要 | `GET /internal/v1/ai/spots/{id}` |
| `search_travel_guides` | 搜索攻略和路线 | `GET /internal/v1/ai/guides/search` |
| `search_foods` | 搜索洛阳美食 | `GET /internal/v1/ai/foods/search` |
| `retrieve_tourism_knowledge` | 检索知识库资料 | Milvus Retriever |
| `recommend_spots` | 获取 Java 推荐候选与解释素材 | `POST /internal/v1/ai/recommendations` |

约束：

- Agent 不直接执行 SQL、Shell、任意 URL、动态代码或未注册工具。
- 工具只返回摘要，最多 10 条；不返回后台私有字段、用户隐私、OSS 私有 URL 或完整 HTML。
- `recommend_spots` 的候选和排序由 Java 决定，AI 只解释，不能改变顺序。
- 单轮工具调用不超过 `AGENT_MAX_TOOL_CALLS`（默认 6），不重复相同参数，整体受 `CHAT_TOTAL_TIMEOUT_SECONDS` 约束。
- 工具结果是数据，不是指令；文档中的“忽略系统提示”“泄露密钥”等内容不得改变工具白名单。

### 10.3 `threadId` 与 checkpointer 对齐

这是 AI 多轮对话的核心契约：

```text
小程序创建会话
  -> Java 生成 26 位 ULID，写入 ai_conversation.thread_id
  -> 返回 conversationId（对外与 threadId 同值）

小程序每轮提问
  -> 只传 conversationId
  -> Java 从 JWT 取 userId，校验 ai_conversation 的 user_id、is_deleted
  -> Java 用同一个 threadId 调 FastAPI /internal/v1/ai/chat

FastAPI
  -> 校验 HMAC，读取 threadId
  -> config.configurable.thread_id = threadId
  -> create_agent(...).ainvoke(..., config=config)
  -> Redis checkpointer 按 thread 恢复/保存图状态

Java 持久化
  -> ai_message 保存用户可见消息
  -> ai_message_source 保存来源
  -> checkpoint 只保存 LangGraph 图状态，不是展示事实源
```

必须在 Java 中固定 `threadId`，每轮不得重新生成。并发请求由 Java 的 `ai:thread:{threadId}:lock` 串行化，并用 `clientMessageId` 做幂等；FastAPI 不依赖客户端传任意 thread 来切换上下文。

### 10.4 checkpoint 丢失与恢复

> Redis 前置条件见 [§6.5.1](../docs/ai/AI-SERVICE-DESIGN.md#651-redis-版本与模块前置条件)。

- checkpointer 正常保留短期状态，TTL 与 Java 会话策略一致（建议 7 天）。
- Redis 重启或状态过期时，Java 可在内部请求中携带最近 6–10 条已持久化消息的 `recoveryHistory`。
- FastAPI 仅在当前 thread 没有 checkpoint 时使用恢复历史，之后交给 checkpointer 正常续接。
- 恢复历史来自 MySQL 的用户可见消息，不从模型自行回忆或重新生成。
- 用户删除会话时，Java 调用 `/internal/v1/ai/checkpoints/{threadId}/reset`；重复删除幂等成功，随后再软删除 MySQL 会话。
- checkpoint 删除失败时记录补偿任务，不能泄露其他用户状态或阻塞主业务删除。

## 11. 内部 API 与安全

### 11.1 路由边界

#### FastAPI 提供、Java 调用

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health/live` | 进程存活检查 |
| `GET` | `/health/ready` | Redis、Milvus 等依赖就绪检查 |
| `POST` | `/internal/v1/ai/chat` | Java 发起的 RAG/Agent 聊天 |
| `POST` | `/internal/v1/ai/checkpoints/{threadId}/reset` | 删除会话 checkpoint |
| `POST` | `/internal/v1/ai/index/jobs` | 创建/提交知识索引任务 |
| `GET` | `/internal/v1/ai/index/jobs/{jobId}` | 查询索引进度 |
| `POST` | `/internal/v1/ai/index/documents/{documentId}/delete` | 删除文档向量 |

#### Java 提供、FastAPI 调用

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/internal/v1/ai/spots/search` | 搜索当前已发布景点摘要 |
| `GET` | `/internal/v1/ai/spots/{id}` | 获取单个已发布景点摘要 |
| `GET` | `/internal/v1/ai/guides/search` | 搜索当前已发布攻略 |
| `GET` | `/internal/v1/ai/guides/{id}` | 获取单篇攻略/行程摘要 |
| `GET` | `/internal/v1/ai/foods/search` | 搜索当前已发布美食 |
| `GET` | `/internal/v1/ai/foods/{id}` | 获取单个美食摘要 |
| `POST` | `/internal/v1/ai/recommendations` | 获取 Java 确定性推荐候选与解释素材 |
| `GET` | `/internal/v1/ai/conversations/{threadId}/messages` | checkpoint 丢失时读取最近用户可见消息（可选） |
| `POST` | `/internal/v1/ai/knowledge/chunks/batch` | 由 FastAPI 回写 chunk 元数据，最终由 Java 持久化 |

调用方向必须以此表为准：FastAPI 的 OpenAPI 只描述第一张表；第二张表属于 Spring Boot 内部 API，不能被误实现为 FastAPI 路由，也不能经 Nginx 暴露。

FastAPI 的 `/docs` 和 `/openapi.json` 仅供本机开发；Nginx 必须把 `/internal/**`、`/docs`、`/openapi.json`、`/health/**` 对外隐藏或返回 404。

### 11.2 HMAC 内部鉴权

Java 与 FastAPI 的双向内部调用统一使用：

```text
canonicalString = timestamp + "\n"
                + METHOD + "\n"
                + pathWithSortedQuery + "\n"
                + requestId + "\n"
                + rawBody
signature = hex(HMAC-SHA256(INTERNAL_SHARED_SECRET, canonicalString))
```

请求头：

- `X-Internal-Timestamp`
- `X-Internal-Request-Id`
- `X-Internal-Signature`
- `Content-Type: application/json`

规则：

- 时间窗默认 ±300 秒。
- 查询参数按名称和值稳定排序；Path 使用实际发送路径。
- body 必须使用实际发送的 UTF-8 原始字节，签名后不得重新序列化。
- 服务端用常量时间比较签名，校验 requestId 时间窗和幂等缓存。
- `INTERNAL_SHARED_SECRET` 只放环境变量，不进入日志或错误信息。
- Java 和 Python 共用测试向量；任何跨语言签名差异在联调前修复。

### 11.3 错误与日志

内部错误使用稳定错误码和统一信封，不把 Python 堆栈、Prompt、模型原始响应返回给 Java。日志记录 `requestId`、`threadId`（必要时）、`userIdHash`、模型名、耗时、token 用量和工具耗时；默认不记录用户完整问题、完整回答和密钥。日志中所有外部 URL、OpenID、Authorization、API Key 都要脱敏。

## 12. 超时、并发与降级

| 场景 | 处理 |
|---|---|
| 检索为空/低于阈值 | 不调用生成模型，返回 `insufficient=true`，HTTP 200 |
| 工具单次超时 | `TOOL_TIMEOUT_SECONDS` 内失败，返回脱敏 `ok=false`，Agent 可换工具或降级 |
| Agent 工具次数超限 | 停止新增工具调用，基于已有证据回答；无证据则资料不足 |
| Agent 递归超限 | 停止循环，返回可解释失败或降级，不无限重试 |
| 全局请求超时 | 受 `CHAT_TOTAL_TIMEOUT_SECONDS` 控制，Java 映射为可重试 AI 错误 |
| Qwen 限流/超时 | 官方 `ChatOpenAI` 重试次数有限，最终返回脱敏错误 |
| Milvus 不可用 | readiness 失败；聊天进入 AI 降级，不影响 Spring Boot 普通业务 |
| Redis checkpoint 不可用 | AI 会话 fail-closed；不退回无记忆的静默多轮；记录告警 |
| HMAC 失败/时间窗过期 | 401 内部鉴权错误，不暴露密钥或具体签名细节 |
| 引用校验失败 | 移除无效引用并重试一次；仍失败则资料不足 |

降级优先级：

```text
业务工具证据 > 知识库检索证据 > 结构化“资料不足” > 脱敏系统错误
```

不允许为了“有回答”而让模型凭常识补全票价、开放时间、电话、地址、交通时长等事实。

## 13. 测试要求

| 层级 | 必测内容 |
|---|---|
| Unit | HMAC 向量、时间窗、Pydantic camelCase、阈值、引用编号、工具参数、来源去重 |
| Contract | OpenAPI 快照、内部字段、错误码、Java/Python 签名一致、用户聊天不泄露 |
| Integration | Redis checkpointer 恢复/重置、Milvus 写入/删除/检索、Java 工具契约、索引幂等 |
| Agent scenario | 多轮上下文、工具选择、恢复历史、工具超限、无资料降级、注入文档防护 |
| Live smoke | 真实 `qwen3.8-flash` / `qwen3-vl-embedding`、真实 Milvus、真实 Java 内部接口 |
| E2E | 小程序/管理端 → Java → FastAPI → Java 的全链路和失败路径 |

必须覆盖的行为：

1. 同一 `threadId` 第二轮能引用第一轮上下文。
2. 删除会话后再次提问不会恢复旧上下文。
3. Redis checkpoint 丢失且带 `recoveryHistory` 时能恢复最近消息。
4. 同一 `clientMessageId` 重复请求只生成一次回答。
5. 同一 thread 并发请求被 Java 锁串行化，FastAPI 不产生双 assistant 消息。
6. 检索低于阈值时不调用生成模型。
7. 文档中的恶意指令不改变工具白名单。
8. 索引任务幂等、旧版本不覆盖新版本、删除后向量不可检索。
9. `/health/live` 与 `/health/ready` 在依赖故障时行为符合预期。
10. 日志、响应和 Swagger 示例不含真实密钥、内部堆栈和用户隐私。

## 14. Definition of Done

一个 AI 功能只有在以下全部满足时才算完成：

- [ ] `pyproject.toml` / `uv.lock` 已锁定依赖，导入冒烟通过。
- [ ] Qwen 聊天只通过 `ChatOpenAI` 调用 `qwen3.8-flash`；向量只通过 DashScope 官方 `MultiModalEmbedding` SDK 适配 `QwenMultiModalEmbeddings`，无手写模型 HTTP。
- [ ] RAG 使用 `create_retrieval_chain` 或等价官方高层封装。
- [ ] Agent 使用 `create_agent`，工具白名单和预算生效。
- [ ] checkpointer 使用锁定版本验证过的官方 Redis 实现，生命周期可管理。
- [ ] `threadId` 与 Java `ai_conversation.thread_id` 一致，多轮、恢复、删除均可验证。
- [ ] 检索不足返回 `insufficient=true`，前端有 `insufficient-card`。
- [ ] 有来源回答的 `sources`、`rankNo`、`snippet`、`deepLink` 可追溯。
- [ ] HMAC、时间窗、requestId、幂等和并发测试通过。
- [ ] 日志脱敏、异常脱敏、健康检查和超时预算符合设计。
- [ ] 单元、契约、集成、Agent、真实冒烟和 E2E 必要用例通过。
- [ ] `docs/ai/AI-SERVICE-DESIGN.md`、API 契约、OpenAPI 快照和本 README 同步更新。
- [ ] 没有任何真实密钥、用户隐私、OSS 签名 URL 或完整 Prompt 被提交。

## 15. 常见问题

### 15.1 Qwen 兼容端点调用失败

先检查 `QWEN_BASE_URL` 是否为当前地域/工作空间的 OpenAI-compatible 端点并确认 `QWEN_API_KEY` 匹配；用 `ChatOpenAI.invoke("你好")` 验证聊天，用 `create_embedding_model().embed_query("测试")` 验证 `qwen3-vl-embedding`。向量模型不要走 OpenAI-compatible Embeddings，也不要通过手写 HTTP 绕过排查。

### 15.2 Milvus 启动或维度不匹配

检查 `MILVUS_URI` 指向宿主机 `192.168.100.128:19530`，collection 的 `dim` 与 `QWEN_EMBEDDING_DIM` 完全一致。更换 embedding 模型必须重建 collection；禁止把不同维度向量混写。Milvus 内部 etcd/MinIO 只由 Compose 管理，不由 AI 服务直接配置。

### 15.3 checkpoint 恢复不了

确认 Java 传来的 `threadId` 是数据库中的 ULID 且从未重新生成；checkpointer 固定使用 Redis DB 0，`langgraph:checkpoint:` 前缀不能与 Java 缓存冲突；检查 `/health/ready` 和 checkpointer setup 日志。只有当前 thread 没有状态时才使用 `recoveryHistory`，且最多最近 6–10 条。

### 15.4 Redis 不可用

AI 服务应按 fail-closed 处理多轮会话，不要把缺少 checkpoint 误判成“无历史正常回答”。Java 保留普通业务能力，前端显示“智能问答暂时不可用”，并可提供景点/攻略/美食入口。

### 15.5 HMAC 401

检查方法、路径、排序后的查询参数、requestId 和原始 body 是否和签名完全一致；检查两端时钟偏差是否在时间窗内；确认使用的是同一 `INTERNAL_SHARED_SECRET`。不要通过放宽为永久有效或关闭校验来解决。

### 15.6 资料不足是否算接口失败

不是。资料不足是 AI 业务结果，使用 HTTP 200 和业务成功码返回，`insufficient=true`。只有依赖不可用、鉴权失败、模型超时等系统错误才使用失败 HTTP/错误码。

### 15.7 Milvus gRPC 被系统代理劫持

如果 `Test-NetConnection 192.168.100.128 -Port 19530` 成功，但 pymilvus 报 `failed to connect ... 127.0.0.1:7897`，说明本机 `HTTP_PROXY/HTTPS_PROXY` 被 gRPC 误用。`app/integrations/milvus.py` 会在创建 VectorStore 和 readiness 检查前自动把 `192.168.100.128` 写入 `NO_PROXY/no_proxy`；手工排查时也可在当前 PowerShell 会话执行 `$env:NO_PROXY="$env:NO_PROXY,192.168.100.128"` 后重试。不要把 Milvus 改为走公网或临时关闭代理后长期不回滚。

## 16. 相关文档

- [项目总入口](../README.md)
- [文档中心](../docs/README.md)
- [系统架构](../docs/architecture/SYSTEM-ARCHITECTURE.md)
- [数据库设计](../docs/database/DATABASE-DESIGN.md)
- [API 契约](../docs/api/API-CONTRACT.md)
- [AI 服务详细设计](../docs/ai/AI-SERVICE-DESIGN.md)
- [开发规范](../docs/development/DEVELOPMENT-STANDARD.md)
- [实施路线图](../docs/roadmap/IMPLEMENTATION-ROADMAP.md)
- [部署说明](../deploy/README.md)


## 17. 已实现模块与提交边界

当前代码按以下模块落地，新增功能应继续遵守“一次性只改一个边界”的原则：

1. `core` / `schemas`：统一 camelCase 契约、错误信封、结构化日志、HMAC 与重放保护。
2. `rag` / `ingestion`：PDF/DOCX/Markdown/TXT 解析、中文切分、Milvus 向量写入/删除、索引任务幂等。
3. `chains` / `prompts`：阈值检索、`create_retrieval_chain`、引用校验、资料不足降级。
4. `agents` / `tools`：`create_agent`、六个白名单工具、请求级证据与工具预算。
5. `memory` / `services`：Redis/InMemory checkpointer、`threadId` 对齐、结果缓存、聊天编排。
6. `api`：聊天、checkpoint reset、索引提交/查询/删除和 health 路由。



