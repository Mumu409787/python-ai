# python-ai

Python + AI 方向的 RAG 检索问答系统。与 Java 项目 [`java-ai-rag`](../java-ai-rag)
功能对齐，并在其基础上补了 **Rerank 重排** 与 **Ragas 评测** 两块。

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 包管理 | uv | 依赖锁在 `uv.lock` |
| 解释器 | Python 3.14 | 备用 3.12 用 `uv python install` |
| 对话模型 | DeepSeek `deepseek-flash` | 推理模型，会先输出思维链 |
| 向量模型 | 硅基流动 `BAAI/bge-m3` | 1024 维 |
| 向量库 | Qdrant | 端口 6335（Java 项目占 6333） |
| 元数据库 | PostgreSQL 17 + pgvector | 端口 5432 |
| 缓存 | Redis 7.4 | 端口 6380，存会话记忆 |
| Web | FastAPI + sse-starlette | 流式用 SSE |
| 编排 | LangChain（仅 LLM 封装） | 检索走原生 `qdrant-client` |
| 评测 | Ragas 0.3.1 | 见 `evaluation/` |

## 快速开始

```bash
# 1) 起依赖服务
docker compose up -d && docker compose ps     # 三个 healthy

# 2) 配置密钥
cp .env.example .env                          # 填 DeepSeek 与硅基流动的 key

# 3) 自检
uv run python scripts/check_env.py            # 模型 / 向量 / rerank / Qdrant 连通性
uv run pytest                                 # 单元测试，不需要 key

# 4) 起服务
uv run uvicorn python_ai.main:app --port 8000

# 5) 端到端验证（另开终端）
uv run python scripts/smoke_test.py           # 22 项
```

## 接口

与 Java 项目逐字对齐，同一前端可同时对接两边。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 依赖健康检查 |
| POST | `/api/documents` | 上传入库（字段名 `file`，上限 20MB，SHA-256 去重） |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{id}` | 删除（连同向量） |
| POST | `/api/chat?mode=RAG\|CHAT` | 非流式问答 |
| POST/GET | `/api/chat/stream` | SSE 流式 |

**统一响应**：`{"code":0,"message":"成功","data":{...}}`，业务错误返回 HTTP 200 + `code != 0`。

**SSE 事件协议**（无 `[DONE]` 哨兵）：

```
event: retrieval     data: [{...引用列表 JSON 数组...}]
event: message       data: 裸文本片段          ← 逐 token，不是 JSON
event: done          data: {"sessionId":"...","costMs":123}
event: error         data: 中文错误文案
```

## 与 Java 项目的差异

| 方面 | java-ai-rag | 本项目 |
|---|---|---|
| 向量写入时机 | 在 `@Transactional` 内 → 事务回滚会留孤儿向量（其 README 已列为已知缺陷） | **事务外**写入，任何一步失败都不产生孤儿向量 |
| 同步问答检索 | 检索两次（手动拿引用 + 框架内部再检索） | **只检索一次**，命中同时用于引用和 prompt |
| 删除向量 | 先从 chunk 表查 `vector_id` 再逐个删 | **Qdrant 原生 payload 过滤**，一步删净（同时保留 `vector_id` 供对账） |
| 重排 | 无 | **Rerank**（`bge-reranker-v2-m3`） |
| 评测 | 无 | **Ragas** 四项指标 |
| 切片分隔符 | 只按换行/空格（中文会被硬切） | 补入中文句末标点 |
| 拒答阈值 | 固定 `min-score=0.6` | 实测标定 `0.45` + **相对**截断（见下） |

## 检索参数是怎么定的

`retrieval_score_threshold = 0.45` **不是拍脑袋写的**，是在 `data/samples/star-link-spec.md`
上实测标定的：

| 查询类型 | 向量分数（最高） |
|---|---|
| 不相关（火锅店 / 天气 / 爬虫） | 0.31 / 0.38 / 0.39 |
| 相关（认证 / 限流 / 灰度） | 0.58 / 0.60 / 0.53 |

分界线在 0.45 附近。**换语料必须重新标定** —— 这个值随领域、文档风格、切片粒度都会变。

> 顺带发现：Java 侧固定用 0.6，在这份语料上会把「认证用的 Token」（0.578）直接漏检。

**Rerank 不做绝对阈值过滤**，只负责排序。原因是 `relevance_score` 与余弦相似度
不是同一量纲，实测同一批候选能落在 `0.0 ~ 0.96`：初版设 `min_score=0.2`，
正好把 0.195 的合法命中挡成了拒答。现在改成**相对 top1 的比例截断**
（`rerank_relative_cutoff=0.1`），既能裁掉高分问题的拖尾噪声，又不会误杀低分问题。

## 评测结果（Ragas）

18 题评测集，完整报告见 `evaluation/report.md`：

| 指标 | 分数 | 含义 |
|---|---|---|
| faithfulness | **0.9280** | 回答是否完全基于检索到的资料（有无幻觉） |
| answer_relevancy | **0.9283** | 回答与问题的相关程度 |
| context_precision | **1.0000** | 检索到的片段里有多少是真正有用的 |
| context_recall | **1.0000** | 标准答案需要的信息，有多少被检索到 |

复现：`uv run python -m evaluation.run_ragas`（约 150 秒，会调用较多 LLM）。
原始分数落在 `evaluation/raw_scores.csv` —— 调整检索参数后可以离线重算，不必重跑评测。

## 目录

```
src/python_ai/
  config.py       配置（pydantic-settings）
  errors.py       业务异常与错误码
  schemas.py      DTO + 统一响应体（对外 camelCase）
  clients.py      LLM / Embedding / Qdrant / Redis 客户端装配
  prompts.py      RAG_SYSTEM 等提示词
  parsing.py      按扩展名解析（PDF 用 pypdf，文本强制 UTF-8）
  chunking.py     递归切分 800/150 + SHA-256
  embedding.py    批量向量化（每批 32，不传 dimensions）
  vectorstore.py  Qdrant 封装（建集合 / 写入 / 检索 / 按文档删除）
  retrieval.py    向量召回 → Rerank → 相对截断
  llm.py          LangChain 对话模型 + 流式
  memory.py       Redis 会话记忆
  ingest.py       入库 / 删除编排
  rag.py          问答编排（含 SSE 事件流）
  repository.py   数据访问
  db/             SQLAlchemy 模型与会话
  api/            FastAPI 路由
  main.py         应用入口与全局异常处理

evaluation/       Ragas 评测集与脚本
scripts/          check_env.py（连通性自检）/ smoke_test.py（端到端）
data/samples/     测试语料
```

## 踩过的坑

- **FastAPI 默认的 OpenAPI 3.1 会让 Swagger UI 的文件上传框失效** —— Pydantic v2 把上传字段
  生成为 `{"type": "string", "contentMediaType": "application/octet-stream"}`，而
  **Swagger UI 只认 OpenAPI 3.0 的 `{"format": "binary"}`**。认不出来，它就把上传框渲染成
  一个**普通文本框**（左边还标着 `string`），用户往里填什么都发不出真文件，服务端一律返回
  **422 Validation Error**。接口本身没问题，是文档页渲染错了，非常隐蔽。
  修法见 `main.py` 的 `_custom_openapi()`：把 openapi 版本降到 3.0.3 并改写字段。
  > 注：给 `FastAPI(openapi_version="3.0.3")` 传参**没用** —— 版本号变了，但 Pydantic 生成的
  > 字段描述不变，必须用 `app.openapi = _custom_openapi` 后处理 schema。

- **Swagger UI 默认从 jsdelivr CDN 拉 JS/CSS，国内直连不通** —— 实测状态码 `000`，
  挂代理也要 12 秒，`/docs` 会白屏。已把 `swagger-ui-bundle.min.js`（1.3MB）与
  `swagger-ui.min.css` 下载到 `src/python_ai/static/`（取自国内镜像 bootcdn），
  并在 `main.py` 里关掉 FastAPI 自带的 `docs_url`、自建离线可用的 `/docs`。
  **升级 swagger-ui 版本时，这两个文件要一起换。**

- **ruff 0.16.8 的 formatter 会把 `except (A, B):` 改写成 `except A, B:`**（Python 2 语法，
  直接产生无效代码），而它自己的 linter 不报错。已规避：不写元组 `except`。
- **`deepseek-flash` 是推理模型**：先吐 `reasoning_content` 再吐正文，两者共享 `max_tokens`。
  给 16 会被思维链吃光、正文为空。LangChain 的 `ChatOpenAI` 已自动过滤掉思维链，
  流式只需跳过空 chunk。
- **ragas 0.3.1 依赖已被移除的 `langchain_community.chat_models.vertexai`**。
  0.4.x 又和 `openai>=3.x` 冲突（它依赖的 `instructor` 要求 `openai<3`），所以只能
  用 0.3.1 + 把 `langchain-community` 钉在 `<0.4` + 装 `pillow`。
- **`bge-m3` 不能传 `dimensions`**，传了返回 400 / code 20015。
- **Qdrant 集合维度必须在启动时真实探测**：非 OpenAI 官方模型自报的维度是错的
  （会报 1536，实际 1024），拿错的值建集合会导致后续写入全部失败。

## 下一步

- [ ] 混合检索（BM25 + 向量，用 `rank_bm25`）
- [ ] 查询改写（Java 侧有实现但默认关，可直接移植）
- [ ] 会话记忆接 LangGraph Checkpointer
- [ ] Langfuse 接入（依赖已装，未接线）
- [ ] Alembic 迁移替代 `create_all`
- [ ] 写一个 MCP Server 暴露知识库检索
