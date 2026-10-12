"""MCP 工具注册、输入校验和独立启动的回归，不连接真实模型。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from python_ai import mcp_server
from python_ai.errors import BizError, ResultCode


@pytest.fixture(autouse=True)
def isolated_dependencies(monkeypatch):
    monkeypatch.setattr(mcp_server, "_dependencies", mcp_server._Dependencies())
    monkeypatch.setattr(mcp_server, "init_db", AsyncMock())
    monkeypatch.setattr(mcp_server, "ensure_collection", AsyncMock())


async def test_mcp_tools_registered():
    tools = await mcp_server.server.list_tools()
    assert {tool.name for tool in tools} == {
        "search_knowledge_base",
        "list_documents",
        "ingest_file",
    }
    search = next(tool for tool in tools if tool.name == "search_knowledge_base")
    assert "query" in search.input_schema["properties"]


@pytest.mark.parametrize("query", [" ", "x" * 4001])
async def test_mcp_invalid_query_skips_retrieval(monkeypatch, query):
    retrieve = AsyncMock()
    monkeypatch.setattr(mcp_server, "retrieve", retrieve)
    assert "检索失败" in await mcp_server.search_knowledge_base(query)
    retrieve.assert_not_awaited()


async def test_mcp_requires_absolute_file_path():
    assert "绝对路径" in await mcp_server.ingest_file("relative.md")


async def test_mcp_ingest_bounds_file_read(monkeypatch, tmp_path):
    file = tmp_path / "large.md"
    file.write_bytes(b"abcdef")
    ingest = AsyncMock(side_effect=BizError(ResultCode.PARAM_INVALID, "文件过大"))
    monkeypatch.setattr(mcp_server, "get_settings", lambda: SimpleNamespace(max_upload_bytes=3))
    monkeypatch.setattr(mcp_server.ingest, "ingest_document", ingest)
    assert "文件过大" in await mcp_server.ingest_file(str(file))
    ingest.assert_awaited_once_with("large.md", b"abcd")


async def test_mcp_lifespan_and_discovery_do_not_initialize_dependencies(monkeypatch):
    calls = {}
    for name in ("init_db", "ensure_collection", "dispose_db", "close_clients"):
        calls[name] = AsyncMock()
        monkeypatch.setattr(mcp_server, name, calls[name])
    async with mcp_server.lifespan(mcp_server.server):
        await mcp_server.server.list_tools()
        calls["init_db"].assert_not_awaited()
        calls["ensure_collection"].assert_not_awaited()
        calls["close_clients"].assert_not_awaited()
    calls["dispose_db"].assert_awaited_once()
    calls["close_clients"].assert_awaited_once()


async def test_mcp_first_call_failure_still_cleans(monkeypatch):
    monkeypatch.setattr(mcp_server, "init_db", AsyncMock(side_effect=RuntimeError("offline")))
    dispose, close = AsyncMock(), AsyncMock()
    monkeypatch.setattr(mcp_server, "dispose_db", dispose)
    monkeypatch.setattr(mcp_server, "close_clients", close)
    with pytest.raises(RuntimeError, match="offline"):
        async with mcp_server.lifespan(mcp_server.server):
            await mcp_server._ensure_dependencies()
    dispose.assert_awaited_once()
    close.assert_awaited_once()


async def test_mcp_document_listing_only_initializes_db_once(monkeypatch):
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_server.ingest, "list_documents", listing)
    await mcp_server.list_documents()
    await mcp_server.list_documents()
    mcp_server.init_db.assert_awaited_once()
    mcp_server.ensure_collection.assert_not_awaited()
    assert listing.await_count == 2


async def test_mcp_concurrent_searches_initialize_vectors_once(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialize_vectors():
        entered.set()
        await release.wait()

    mcp_server.ensure_collection.side_effect = initialize_vectors
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_server, "retrieve", retrieve)
    first = asyncio.create_task(mcp_server.search_knowledge_base("first"))
    second = asyncio.create_task(mcp_server.search_knowledge_base("second"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        retrieve.assert_not_awaited()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    finally:
        release.set()
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
    mcp_server.init_db.assert_awaited_once()
    mcp_server.ensure_collection.assert_awaited_once()
    assert retrieve.await_count == 2


async def test_mcp_vector_initialization_failure_can_retry(monkeypatch):
    mcp_server.ensure_collection.side_effect = [
        BizError(ResultCode.EMBEDDING_FAILED, "暂时不可用"),
        None,
    ]
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_server, "retrieve", retrieve)
    assert "暂时不可用" in await mcp_server.search_knowledge_base("question")
    retrieve.assert_not_awaited()
    assert "没有命中" in await mcp_server.search_knowledge_base("question")
    mcp_server.init_db.assert_awaited_once()
    assert mcp_server.ensure_collection.await_count == 2
    retrieve.assert_awaited_once()


@pytest.mark.parametrize("top_k", [0, 101])
async def test_mcp_invalid_top_k_skips_initialization(top_k):
    assert "top_k" in await mcp_server.search_knowledge_base("question", top_k)
    mcp_server.init_db.assert_not_awaited()
    mcp_server.ensure_collection.assert_not_awaited()


async def test_mcp_ingestion_initializes_vectors_before_service(monkeypatch, tmp_path):
    file = tmp_path / "document.md"
    file.write_text("test", encoding="utf-8")

    async def ingest_document(*args):
        mcp_server.init_db.assert_awaited_once()
        mcp_server.ensure_collection.assert_awaited_once()
        return SimpleNamespace(duplicated=True, doc_id=1, chunk_count=1)

    monkeypatch.setattr(mcp_server.ingest, "ingest_document", ingest_document)
    assert "跳过重复入库" in await mcp_server.ingest_file(str(file))


async def test_mcp_database_initialization_failure_can_retry(monkeypatch):
    mcp_server.init_db.side_effect = [RuntimeError("offline"), None]
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_server.ingest, "list_documents", listing)
    assert "offline" in await mcp_server.list_documents()
    listing.assert_not_awaited()
    await mcp_server.list_documents()
    assert mcp_server.init_db.await_count == 2
    mcp_server.ensure_collection.assert_not_awaited()
    listing.assert_awaited_once()
