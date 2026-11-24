# AI 服务（学习友好版）

> 面向本科毕业设计：保留 AI 问答、RAG、Milvus、Agent、MinerU、MCP 和索引任务，但不再在 `app` 下拆 13 个子包。
> 为什么这样改见 [LEARNER-REFACTOR.md](LEARNER-REFACTOR.md)，逐文件讲解见 [CODE-GUIDE.md](CODE-GUIDE.md)。

## 1. 一句话理解

```text
Spring Boot（唯一业务入口）
  -> FastAPI（仅本机 127.0.0.1:8000）
       -> LangGraph Agent（create_agent + InMemorySaver）
            -> 六个本地业务工具
            -> RAG：MinerU -> 切分 -> DashScopeEmbeddings -> Milvus
            -> 可选 MCP：12306 车票查询
```

FastAPI 不负责公网 JWT、业务 MySQL 和 OSS 权限。景点、攻略、美食、推荐和聊天记录的最终事实都在 Java/MySQL。

## 2. 平铺目录

```text
ai-service/
├─ app/
│  ├─ config.py          环境变量、常量、日志、ULID、内部 token
│  ├─ errors.py          统一业务异常
│  ├─ schemas.py         Pydantic 请求/响应模型
│  ├─ llm.py             ChatOpenAI + DashScopeEmbeddings
│  ├─ rag.py             MinerU、切分、Milvus、create_retrieval_chain
│  ├─ springboot.py      Java 内部接口客户端
│  ├─ tools.py           六个本地工具、工具预算、MCP 包装
│  ├─ agent.py           create_agent、系统提示词、引用校验
│  ├─ chat.py            一次问答的主流程
│  ├─ indexer.py         下载、索引任务、MinerU 入库流水线
│  ├─ runtime.py         启动时组装依赖
│  └─ main.py            FastAPI 路由和异常处理
├─ tests/
├─ .env
├─ .env.example
├─ pyproject.toml
├─ CODE-GUIDE.md
└─ LEARNER-REFACTOR.md
```

没有 `agents/`、`chains/`、`core/`、`services/`、`rag/`、`schemas/` 等子包。

## 3. 启动

```powershell
cd D:\luoyang-smart-tourism\ai-service
uv sync --all-groups
uv run pytest
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

地址：

- `http://127.0.0.1:8000/health/live`
- `http://127.0.0.1:8000/health/ready`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## 4. 必填配置

```text
QWEN_API_KEY
QWEN_BASE_URL
QWEN_CHAT_MODEL
QWEN_EMBEDDING_MODEL=text-embedding-v4
MILVUS_URI
MILVUS_COLLECTION=luoyang_tourism_knowledge_v2
SPRINGBOOT_INTERNAL_BASE_URL
INTERNAL_TOKEN
```

PDF/DOCX/图片索引还需要：

```text
MINERU_API_KEY
```

可选 MCP：

```text
MCP_12306_ENABLED=true
```

MCP 默认关闭。启用后会通过 `npx -y 12306-mcp` 启动子进程；没有 Node/npm 或网络时，本地 RAG 和六个业务工具仍可正常工作。

## 5. 七个内部接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/internal/v1/ai/chat` | AI 问答 |
| POST | `/internal/v1/ai/checkpoints/{threadId}/reset` | 删除会话 checkpoint |
| POST | `/internal/v1/ai/index/jobs` | 提交索引任务 |
| GET | `/internal/v1/ai/index/jobs/{jobId}` | 查询索引进度 |
| POST | `/internal/v1/ai/index/documents/{documentId}/delete` | 删除文档向量 |
| GET | `/health/live` | 进程存活 |
| GET | `/health/ready` | Milvus / Agent 就绪 |

内部接口请求头：

```http
X-Internal-Token: <INTERNAL_TOKEN>
```

## 6. 两条重要说明

### 6.1 更换 Embedding 必须重建 collection

原模型 `qwen3-vl-embedding` 改为学习资料中的 `text-embedding-v4`，collection 从 `luoyang_tourism_knowledge_v1` 改为 `luoyang_tourism_knowledge_v2`。旧向量不能直接复用。

### 6.2 MCP 是可选增强

12306 使用第三方开源项目 `Joooook/12306-mcp`。它提供实时余票查询，但不能保证长期可用。MCP 的结果会显示查询时间，并提示“实时数据仅供参考，请以 12306 官方页面为准”。

## 7. 测试

```powershell
uv run ruff check app tests
uv run mypy app
uv run pytest -q
```

真实外部依赖按需打开：

```powershell
$env:RUN_INFRA_TESTS='1'     # Milvus
$env:RUN_QWEN_SMOKE='1'      # 千问 Chat/Embedding
$env:RUN_MINERU_SMOKE='1'    # MinerU
$env:RUN_MCP_TESTS='1'       # 12306 MCP
```

## 8. 相关文档

- [化简决策](LEARNER-REFACTOR.md)
- [全方位代码讲解](CODE-GUIDE.md)
- [项目总入口](../README.md)
- [系统架构](../docs/architecture/SYSTEM-ARCHITECTURE.md)
- [AI 服务设计](../docs/ai/AI-SERVICE-DESIGN.md)
