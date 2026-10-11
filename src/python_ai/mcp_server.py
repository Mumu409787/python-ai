"""MCP server：把 RAG 检索能力暴露给任意 MCP 客户端。

**定位**：这是项目的对外接口层之一，与 `api/`（FastAPI 路由）、CLI 平级。
它**不实现任何检索逻辑** —— 业务逻辑的唯一归属地仍是 `retrieval.py` / `rag.py`，
本文件只做「把检索结果翻译成 MCP 工具返回值」这一件事。以后改检索只改下层，
这里不用动（这正是分层的目的）。

**为什么不塞进 Claude Code 的专属目录**：放在项目仓库里，它才属于项目本身。
任何 MCP 客户端（Claude Code / Cursor / Codex / Claude Desktop）都能连它，
而不是某一个工具的配件。

本地调试：`uv run --directory D:/Projects/python-ai python -m python_ai.mcp_server`

注册到 Claude Code：见本文件末尾 docstring 或 README。
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from .db.session import session_scope
from .errors import BizError
from .repository import list_documents as _fetch_documents
from .retrieval import retrieve

# ⚠️ stdio 传输下**绝对不能往 stdout 写东西**，那会破坏 MCP 协议帧。
# basicConfig 默认输出到 stderr，所以是安全的 —— 别改成 stdout 的 handler。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

server = MCPServer(
    name="python-ai-rag",
    instructions=(
        "访问本机 python-ai 项目的 RAG 知识库。"
        "search_knowledge_base 做语义检索（向量召回 + Rerank），"
        "list_documents 列出已入库的文档。"
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
    try:
        hits = await retrieve(query, top_k=top_k)
    except BizError as exc:
        return f"检索失败：{exc.message}"

    if not hits:
        return "没有命中任何内容（均低于相关度阈值，或知识库为空）。"

    blocks = [f"命中 {len(hits)} 条："]
    for index, hit in enumerate(hits, 1):
        snippet = hit.content
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS] + "…"
        blocks.append(
            f"\n[{index}] 相关度 {hit.score:.4f}｜文档「{hit.doc_name or '未知'}」"
            f"（doc_id={hit.doc_id}，第 {hit.chunk_index} 片）\n{snippet}"
        )
    return "\n".join(blocks)


@server.tool()
async def list_documents(limit: int = 50) -> str:
    """列出知识库中已入库的文档（按入库时间倒序）。

    适用场景：想先知道知识库里都有哪些资料时。

    Args:
        limit: 最多返回多少条，默认 50。
    """
    try:
        async with session_scope() as session:
            docs = await _fetch_documents(session, limit=limit)
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


def main() -> None:
    """以 stdio 运行 —— 由 MCP 客户端按需拉起本进程。"""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
