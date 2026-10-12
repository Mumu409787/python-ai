"""MCP server：把 RAG 能力暴露给任意 MCP 客户端。

**定位**：这是项目的对外接口层之一，与 `api/`（FastAPI 路由）、CLI 平级。
它**不实现任何业务逻辑**，只做「把服务层的结果翻译成 MCP 工具返回值」这一件事：
    - 检索        → `retrieval.retrieve()`
    - 入库 / 列表 → `ingest.ingest_document()` / `ingest.list_documents()`
以后改业务逻辑只改服务层，本文件不用动（这正是分层的目的）。

**为什么不塞进 Claude Code 的专属目录**：放在项目仓库里，它才属于项目本身。
任何 MCP 客户端（Claude Code / Cursor / Codex / Claude Desktop）都能连它，
而不是某一个工具的配件。

本地调试：`uv run --directory D:/Projects/python-ai python -m python_ai.mcp_server`
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from starlette.concurrency import run_in_threadpool

from . import ingest
from .clients import close_clients
from .config import get_settings
from .db.session import dispose_db, init_db
from .errors import BizError
from .retrieval import retrieve
from .vectorstore import ensure_collection

# ⚠️ stdio 传输下**绝对不能往 stdout 写东西**，那会破坏 MCP 协议帧。
# basicConfig 默认输出到 stderr，所以是安全的 —— 别改成 stdout 的 handler。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class _Dependencies:
    db_ready: bool = False
    vectors_ready: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_dependencies = _Dependencies()


async def _ensure_dependencies(*, vectors: bool = False) -> None:
    """首次业务调用才初始化；并发调用共用结果，失败的阶段下次可重试。"""
    async with _dependencies.lock:
        if not _dependencies.db_ready:
            await init_db()
            _dependencies.db_ready = True
        if vectors and not _dependencies.vectors_ready:
            await ensure_collection()
            _dependencies.vectors_ready = True


@asynccontextmanager
async def lifespan(_: MCPServer):
    """启动只注册工具；业务依赖延迟初始化，退出时释放已使用的资源。"""
    global _dependencies
    _dependencies = _Dependencies()
    try:
        yield
    finally:
        try:
            await dispose_db()
        finally:
            try:
                await close_clients()
            finally:
                _dependencies = _Dependencies()


server = MCPServer(
    name="python-ai-rag",
    lifespan=lifespan,
    instructions=(
        "访问本机 python-ai 项目的 RAG 知识库。"
        "search_knowledge_base 做知识库检索（可选 BM25 混合召回 + Rerank）；"
        "list_documents 列出已入库的文档；"
        "ingest_file 把本地文件灌进知识库。"
    ),
)

# 单条命中返回时的截断长度 —— 片段太长会白占调用方的上下文窗口
_SNIPPET_CHARS = 800


@server.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在本地 RAG 知识库中做语义检索，返回最相关的文档片段。

    适用场景：需要基于自己入库的资料回答问题、核对事实、查证细节时。

    Args:
        query: 检索问题，自然语言即可（与知识库文档同语种时效果最好）。
        top_k: 最多返回几条，默认 5。
    """
    if not query.strip() or len(query) > 4000:
        return "检索失败：问题不能为空且不能超过 4000 个字符"
    if not 1 <= top_k <= 100:
        return "检索失败：top_k 必须在 1 到 100 之间"
    try:
        await _ensure_dependencies(vectors=True)
        hits = await retrieve(query, top_k=top_k)
    except BizError as exc:
        return f"检索失败：{exc.message}"

    if not hits:
        return "没有命中任何内容（无相关向量或关键词匹配，或知识库为空）。"

    blocks = [f"命中 {len(hits)} 条："]
    for index, hit in enumerate(hits, 1):
        snippet = hit.content
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS] + "…"
        blocks.append(
            f"\n[{index}] {hit.score_kind} 分数 {hit.score:.4f}｜文档「{hit.doc_name or '未知'}」"
            f"（doc_id={hit.doc_id}，第 {hit.chunk_index} 片）\n{snippet}"
        )
    return "\n".join(blocks)


@server.tool()
async def list_documents() -> str:
    """列出知识库中已入库的文档（按入库时间倒序）。"""
    try:
        await _ensure_dependencies()
        docs = await ingest.list_documents()
    except Exception as exc:
        return f"查询文档列表失败：{exc}"

    if not docs:
        return "知识库中还没有任何文档。"

    lines = [f"共 {len(docs)} 个文档："]
    lines.extend(
        f"- [{doc.id}] {doc.doc_name}｜{doc.chunk_count} 片｜{doc.status}｜{doc.created_at}"
        for doc in docs
    )
    return "\n".join(lines)


@server.tool()
async def ingest_file(path: str) -> str:
    """把本地文件灌进 RAG 知识库（解析 → 切片 → 向量化 → 写 Qdrant 与 Postgres）。

    适用场景：用户让你把某份资料加进知识库时。

    Args:
        path: 本地文件的**绝对路径**，支持 .md / .txt / .pdf。
              内容相同的文件会自动跳过（按 SHA-256 秒传去重），不会重复向量化。
    """
    try:
        file = Path(path).expanduser()
        if not file.is_absolute():
            return "入库失败：请提供文件的绝对路径"
        if not file.is_file():
            return f"文件不存在或不是普通文件：{file}"
        limit = get_settings().max_upload_bytes

        def read_bounded() -> bytes:
            with file.open("rb") as stream:
                return stream.read(limit + 1)

        data = await run_in_threadpool(read_bounded)
        await _ensure_dependencies(vectors=True)
        result = await ingest.ingest_document(file.name, data)
    except BizError as exc:
        return f"入库失败：{exc.message}"
    except Exception as exc:
        return f"入库失败：{exc}"

    if result.duplicated:
        return f"内容已存在，跳过重复入库（doc_id={result.doc_id}，{result.chunk_count} 片）。"
    return (
        f"入库成功：doc_id={result.doc_id}｜{result.doc_name}｜"
        f"{result.chunk_count} 片｜耗时 {result.cost_ms} ms"
    )


def main() -> None:
    """以 stdio 运行 —— 由 MCP 客户端按需拉起本进程。"""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
