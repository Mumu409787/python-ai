"""可选服务联调：真实 PG/Qdrant/Redis，模型替身，无 API 费用。

PowerShell: $env:RUN_SERVICE_TESTS='1'; uv run pytest tests/test_services.py
使用独立 Qdrant 集合，只创建和清理本次测试数据。
"""

import asyncio
import os
from uuid import uuid4

import httpx
import pytest

from python_ai import ingest, lexical, memory, rag, repository, vectorstore
from python_ai.clients import close_clients, qdrant, redis_client
from python_ai.config import get_settings
from python_ai.db.session import dispose_db, document_lock, init_db, session_scope
from python_ai.errors import BizError, ResultCode
from python_ai.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SERVICE_TESTS") != "1", reason="需显式开启真实服务联调"
)


async def test_real_services_document_chat_and_concurrency(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "memory_backend", "redis")
    token = uuid4().hex
    collection = f"test_python_ai_{token}"
    sid = f"test-session-{token}"
    monkeypatch.setattr(settings, "qdrant_collection", collection)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    # 本组假模型不执行付费改写，改写编排另由离线回归覆盖。
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    doc_ids = []
    entered, resume = asyncio.Event(), asyncio.Event()
    pending = None

    async def embed(texts):
        entered.set()
        await resume.wait()
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def query(text):
        return [1.0, 0.0, 0.0, 0.0]

    async def dimension():
        return 4

    async def generate(*args):
        return "根据资料，测试回答。[1]"

    async def stream(*args):
        yield "测试"
        yield "回答。[1]"

    monkeypatch.setattr(ingest, "embed_texts", embed)
    monkeypatch.setattr(vectorstore, "embed_query", query)
    monkeypatch.setattr(vectorstore, "probe_dimension", dimension)
    monkeypatch.setattr(rag, "generate_text", generate)
    monkeypatch.setattr(rag, "stream_text", stream)

    try:
        await init_db()
        await vectorstore.ensure_collection()
        async with document_lock(token * 2):
            with pytest.raises(BizError) as exc:
                async with document_lock(token * 2):
                    pytest.fail("相同内容的数据库锁应互斥")
            assert exc.value.code == ResultCode.DOC_ALREADY_EXISTS

        data = f"联调文档 {token}。测试回答的来源。".encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = (await client.get("/api/health")).json()["data"]
            assert all(health[key] for key in ("postgres", "qdrant", "redis"))

            pending = asyncio.create_task(ingest.ingest_document(f"{token}.md", data))
            await asyncio.wait_for(entered.wait(), 10)
            # PROCESSING/PENDING 记录不能被另一个上传或删除清掉。
            async with session_scope() as session:
                doc = await repository.find_document_by_hash(session, ingest.sha256_hex(data))
                assert doc is not None
                doc_ids.append(doc.id)
            assert not await lexical.bm25_recall(token, top_k=100)
            with pytest.raises(BizError) as exc:
                await ingest.delete_document(doc.id)
            assert exc.value.code == ResultCode.DOC_ALREADY_EXISTS
            duplicate = (
                await client.post("/api/documents", files={"file": ("copy.md", data)})
            ).json()
            assert duplicate["code"] == ResultCode.DOC_ALREADY_EXISTS
            resume.set()
            first = await asyncio.wait_for(pending, 10)
            assert not first.duplicated
            assert await vectorstore.count_by_doc(first.doc_id) == first.chunk_count
            lexical_hits = await lexical.bm25_recall(token, top_k=100)
            assert any(hit.doc_id == first.doc_id for hit in lexical_hits)
            duplicate = (
                await client.post("/api/documents", files={"file": ("copy.md", data)})
            ).json()
            assert duplicate["data"]["duplicated"]
            assert duplicate["data"]["docId"] == first.doc_id

            response = (
                await client.post("/api/chat", json={"message": "测试回答", "sessionId": sid})
            ).json()
            assert response["code"] == 0 and response["data"]["hits"]
            assert response["data"]["hits"][0]["docId"] == first.doc_id
            assert await memory.load_history(sid) == [("测试回答", "根据资料，测试回答。[1]")]
            sse = await client.post(
                "/api/chat/stream", json={"message": "测试回答", "sessionId": sid}
            )
            assert "event: retrieval" in sse.text and "event: done" in sse.text
            assert "event: error" not in sse.text
            assert len(await memory.load_history(sid)) == 2
            assert await redis_client().ttl(f"chat:session:{sid}") > 0

            deleted = (await client.delete(f"/api/documents/{first.doc_id}")).json()
            assert deleted["code"] == 0
            assert await vectorstore.count_by_doc(first.doc_id) == 0
            assert not await lexical.bm25_recall(token, top_k=100)
            async with session_scope() as session:
                assert await repository.get_document(session, first.doc_id) is None
    finally:
        resume.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        try:
            for doc_id in doc_ids:
                async with session_scope() as session:
                    doc = await repository.get_document(session, doc_id)
                    if doc is not None:
                        await repository.delete_document(session, doc)
            if await qdrant().collection_exists(collection):
                await qdrant().delete_collection(collection)
            await redis_client().delete(f"chat:session:{sid}")
        finally:
            await dispose_db()
            await close_clients()
