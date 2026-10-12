"""已有功能的异常、并发和接口回归；外部服务全部替身化。"""

from contextlib import asynccontextmanager
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client.models import Distance, VectorParams
from sqlalchemy.engine import make_url

from python_ai import chunking, clients, embedding, ingest, llm, memory, rag, retrieval, vectorstore
from python_ai.api import documents
from python_ai.config import Settings
from python_ai.db import session as db_session
from python_ai.db.models import DocStatus
from python_ai.errors import BizError, ResultCode
from python_ai.main import app, lifespan
from python_ai.parsing import parse_document
from python_ai.prompts import REFUSAL_TEXT
from python_ai.schemas import ChatMode, ChatRequest
from python_ai.vectorstore import SearchHit


@asynccontextmanager
async def fake_scope(*args, **kwargs):
    yield SimpleNamespace(flush=AsyncMock())


def hit(score=0.8, doc_id=1):
    return SearchHit(score, doc_id, "a.md", 1, 0, "资料正文")


@pytest.mark.parametrize("message", [" ", "\n\t", "　"])
def test_blank_question_rejected(message):
    with pytest.raises(ValidationError):
        ChatRequest(message=message)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"session_max_turns": 0},
        {"session_ttl_seconds": 0},
        {"embedding_batch_size": -1},
        {"retrieval_top_k": 0},
        {"rerank_relative_cutoff": 1.1},
        {"max_upload_bytes": 0},
    ],
)
def test_invalid_configuration_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_database_password_special_characters():
    settings = Settings(_env_file=None, postgres_password="p@ss:/?#%")
    assert make_url(settings.postgres_dsn).password == "p@ss:/?#%"


def test_utf8_bom_removed():
    assert parse_document("a.md", b"\xef\xbb\xbfhello") == "hello"


def test_explicit_zero_chunk_overlap_is_respected():
    assert chunking.split_text("abcdefghij", chunk_size=4, chunk_overlap=0) == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_pdf_page_extraction_error_is_business_error(monkeypatch):
    page = MagicMock()
    page.extract_text.side_effect = RuntimeError("bad page")
    monkeypatch.setattr("python_ai.parsing.PdfReader", lambda _: SimpleNamespace(pages=[page]))
    with pytest.raises(BizError) as exc:
        parse_document("a.pdf", b"bad pdf")
    assert exc.value.code == ResultCode.PARAM_INVALID


async def test_embedding_restores_index_order(monkeypatch):
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[1.0]),
            ]
        )
    )
    monkeypatch.setattr(
        embedding,
        "embedding_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    assert await embedding.embed_texts(["a", "b"]) == [[1.0], [2.0]]
    assert "dimensions" not in create.call_args.kwargs


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(0, [1.0])],
        [(0, [1.0]), (0, [2.0])],
        [(0, []), (1, [])],
        [(0, [1.0]), (1, [1.0, 2.0])],
        [(0, [float("nan")]), (1, [1.0])],
    ],
)
async def test_invalid_embedding_response_rejected(monkeypatch, rows):
    create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in rows])
    )
    monkeypatch.setattr(
        embedding,
        "embedding_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    with pytest.raises(BizError) as exc:
        await embedding.embed_texts(["a", "b"])
    assert exc.value.code == ResultCode.EMBEDDING_FAILED


@pytest.mark.parametrize("content", ["", "  ", []])
async def test_empty_llm_answer_rejected(monkeypatch, content):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=content)))
    monkeypatch.setattr(llm, "chat_model", lambda: model)
    with pytest.raises(BizError) as exc:
        await llm.generate_text("system", "question")
    assert exc.value.code == ResultCode.LLM_CALL_FAILED


async def test_empty_stream_rejected(monkeypatch):
    async def chunks(*args):
        yield SimpleNamespace(content="")

    monkeypatch.setattr(llm, "chat_model", lambda: SimpleNamespace(astream=chunks))
    with pytest.raises(BizError):
        _ = [token async for token in llm.stream_text("s", "q")]


@pytest.mark.parametrize(
    "error,code",
    [
        (RuntimeError("429 rate limit secret"), ResultCode.LLM_RATE_LIMITED),
        (TimeoutError(), ResultCode.LLM_TIMEOUT),
        (RuntimeError("secret credential"), ResultCode.LLM_CALL_FAILED),
    ],
)
async def test_llm_errors_are_mapped(monkeypatch, error, code):
    monkeypatch.setattr(
        llm, "chat_model", lambda: SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
    )
    with pytest.raises(BizError) as exc:
        await llm.generate_text("s", "q")
    assert exc.value.code == code
    assert "secret" not in exc.value.message


@pytest.fixture
def rag_deps(monkeypatch):
    retrieve = AsyncMock(return_value=[])
    generate = AsyncMock(return_value="正文")
    append = AsyncMock()
    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(rag, "generate_text", generate)
    monkeypatch.setattr(rag, "load_history", AsyncMock(return_value=[]))
    monkeypatch.setattr(rag, "append_turn", append)
    return retrieve, generate, append


async def test_empty_rag_refuses_without_llm(rag_deps):
    retrieve, generate, append = rag_deps
    response = await rag.answer("未知问题", session_id="s-1")
    assert response.answer == REFUSAL_TEXT and response.hits == []
    retrieve.assert_awaited_once()
    generate.assert_not_awaited()
    append.assert_awaited_once_with("s-1", "未知问题", REFUSAL_TEXT)


async def test_empty_rag_stream_sends_retrieval_and_done(rag_deps):
    events = [e async for e in rag.stream_answer("未知问题", session_id="s-1")]
    assert [e[0] for e in events] == ["retrieval", "message", "done"]
    assert events[0][1] == "[]" and events[1][1] == REFUSAL_TEXT


async def test_chat_mode_skips_retrieval(rag_deps):
    retrieve, generate, _ = rag_deps
    response = await rag.answer("你好", mode=ChatMode.CHAT)
    assert not response.used_rag and response.answer == "正文"
    retrieve.assert_not_awaited()
    generate.assert_awaited_once()


async def test_failed_stream_never_saves_partial_history(monkeypatch, rag_deps):
    async def broken(*args):
        yield "部分正文"
        raise BizError(ResultCode.LLM_TIMEOUT, "模型超时")

    monkeypatch.setattr(rag, "stream_text", broken)
    events = [e async for e in rag.stream_answer("你好", mode=ChatMode.CHAT)]
    assert [e[0] for e in events] == ["message", "error"]
    rag_deps[2].assert_not_awaited()


@pytest.mark.parametrize("top_k", [0, -1, 101])
async def test_invalid_retrieval_count_rejected(top_k):
    with pytest.raises(BizError) as exc:
        await retrieval.retrieve("q", top_k=top_k)
    assert exc.value.code == ResultCode.PARAM_INVALID


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": 0.5}] * 2,
        [{"index": -1, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0.5}],
    ],
)
async def test_bad_rerank_response_falls_back(monkeypatch, results):
    # 这组专门验证旧向量候选的回退阈值；混合检索由 test_hybrid.py 独立覆盖。
    monkeypatch.setattr(retrieval, "bm25_recall", AsyncMock(return_value=[]))
    monkeypatch.setattr(retrieval, "vector_recall", AsyncMock(return_value=[hit(), hit(0.2)]))
    response = MagicMock(status_code=200)
    response.json.return_value = {"results": results}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(retrieval.httpx, "AsyncClient", lambda **kw: client)
    hits = await retrieval.retrieve("q")
    assert len(hits) == 1 and hits[0].score == 0.8


@pytest.fixture
def ingest_deps(monkeypatch):
    monkeypatch.setattr(ingest, "session_scope", fake_scope)
    monkeypatch.setattr(ingest, "document_lock", fake_scope)
    doc = SimpleNamespace(id=1, doc_name="a.md", chunk_count=1, status=DocStatus.DONE)
    mocks = {}
    for name, result in [
        ("find_document_by_hash", None),
        ("create_document", doc),
        ("add_chunks", [SimpleNamespace(id=1, vector_id="v", chunk_index=0)]),
        ("mark_status", None),
        ("delete_document", None),
    ]:
        mocks[name] = AsyncMock(return_value=result)
        monkeypatch.setattr(ingest.repo, name, mocks[name])
    for name, result in [
        ("embed_texts", [[1.0]]),
        ("upsert_points", None),
        ("delete_by_doc", None),
    ]:
        mocks[name] = AsyncMock(return_value=result)
        monkeypatch.setattr(ingest, name, mocks[name])
    return mocks, doc


async def test_duplicate_upload_does_not_embed(ingest_deps):
    mocks, doc = ingest_deps
    mocks["find_document_by_hash"].return_value = doc
    response = await ingest.ingest_document("a.md", b"hello")
    assert response.duplicated and response.doc_id == 1
    mocks["embed_texts"].assert_not_awaited()
    mocks["delete_by_doc"].assert_not_awaited()


async def test_retry_cleans_vectors_before_deleting_record(ingest_deps):
    mocks, doc = ingest_deps
    doc.status = DocStatus.FAILED
    mocks["find_document_by_hash"].return_value = doc
    events = []
    mocks["delete_by_doc"].side_effect = lambda *a: events.append("vectors")
    mocks["delete_document"].side_effect = lambda *a: events.append("record")
    await ingest.ingest_document("a.md", b"hello")
    assert events == ["vectors", "record"]


async def test_retry_cleanup_failure_preserves_old_record(ingest_deps):
    mocks, doc = ingest_deps
    doc.status = DocStatus.PENDING
    mocks["find_document_by_hash"].return_value = doc
    mocks["delete_by_doc"].side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await ingest.ingest_document("a.md", b"hello")
    mocks["delete_document"].assert_not_awaited()
    mocks["embed_texts"].assert_not_awaited()


@pytest.mark.parametrize("stage", ["upsert_points", "mark_status"])
async def test_ingest_failure_compensates_vectors(ingest_deps, stage):
    mocks, _ = ingest_deps
    if stage == "mark_status":
        mocks[stage].side_effect = [RuntimeError("original failure"), None]
    else:
        mocks[stage].side_effect = RuntimeError("original failure")
    with pytest.raises(RuntimeError, match="original failure"):
        await ingest.ingest_document("a.md", b"hello")
    mocks["delete_by_doc"].assert_awaited_once_with(1)
    assert mocks["mark_status"].call_args.args[2] == DocStatus.FAILED


async def test_cleanup_failure_keeps_original_exception(ingest_deps):
    mocks, _ = ingest_deps
    mocks["upsert_points"].side_effect = RuntimeError("original failure")
    mocks["delete_by_doc"].side_effect = RuntimeError("cleanup failure")
    with pytest.raises(RuntimeError, match="original failure"):
        await ingest.ingest_document("a.md", b"hello")


async def test_document_lock_rejects_concurrent_operation(monkeypatch):
    session = SimpleNamespace(scalar=AsyncMock(return_value=False))

    @asynccontextmanager
    async def scope():
        yield session

    monkeypatch.setattr(db_session, "session_scope", scope)
    with pytest.raises(BizError) as exc:
        async with db_session.document_lock("f" * 64):
            pytest.fail("should not acquire")
    assert exc.value.code == ResultCode.DOC_ALREADY_EXISTS
    assert session.scalar.call_args.args[1]["key"] == -1


async def test_search_excludes_failed_or_deleted_documents(monkeypatch):
    points = [
        SimpleNamespace(score=0.9, payload={"doc_id": i, "content": "text"}) for i in (1, 2, 3)
    ]
    monkeypatch.setattr(vectorstore, "embed_query", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(
        vectorstore,
        "qdrant",
        lambda: SimpleNamespace(
            query_points=AsyncMock(return_value=SimpleNamespace(points=points))
        ),
    )
    monkeypatch.setattr(vectorstore, "session_scope", fake_scope)
    monkeypatch.setattr(vectorstore, "done_document_ids", AsyncMock(return_value={2}))
    assert [h.doc_id for h in await vectorstore.search("q")] == [2]


@pytest.mark.parametrize(
    "params",
    [
        {},
        VectorParams(size=8, distance=Distance.COSINE),
        VectorParams(size=4, distance=Distance.DOT),
    ],
)
async def test_incompatible_collection_fails_without_destructive_advice(monkeypatch, params):
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        get_collection=AsyncMock(
            return_value=SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=params))
            )
        ),
    )
    monkeypatch.setattr(vectorstore, "qdrant", lambda: client)
    monkeypatch.setattr(vectorstore, "probe_dimension", AsyncMock(return_value=4))
    with pytest.raises(BizError) as exc:
        await vectorstore.ensure_collection()
    assert exc.value.code == ResultCode.VECTOR_STORE_UNAVAILABLE
    assert "down -v" not in exc.value.message


async def test_memory_writes_trim_and_ttl_in_one_transaction(monkeypatch):
    monkeypatch.setattr(memory.get_settings(), "memory_backend", "redis")
    pipe = MagicMock()
    pipe.execute = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = pipe
    client = MagicMock()
    client.pipeline.return_value = context
    monkeypatch.setattr(memory, "redis_client", lambda: client)
    await memory.append_turn("s-1", "q", "a")
    client.pipeline.assert_called_once_with(transaction=True)
    pipe.rpush.assert_called_once()
    pipe.ltrim.assert_called_once()
    pipe.expire.assert_called_once()
    pipe.execute.assert_awaited_once()


async def test_startup_failure_releases_resources(monkeypatch):
    init = AsyncMock(side_effect=RuntimeError("startup failed"))
    dispose, close = AsyncMock(), AsyncMock()
    monkeypatch.setattr("python_ai.main.init_db", init)
    monkeypatch.setattr("python_ai.main.dispose_db", dispose)
    monkeypatch.setattr("python_ai.main.close_clients", close)
    with pytest.raises(RuntimeError, match="startup failed"):
        async with lifespan(app):
            pass
    dispose.assert_awaited_once()
    close.assert_awaited_once()


async def test_close_clients_does_not_create_unused_clients(monkeypatch):
    for name in ("chat_client", "embedding_client", "chat_model", "qdrant", "redis_client"):

        @lru_cache(maxsize=1)
        def unused():
            pytest.fail("shutdown must not construct unused clients")

        monkeypatch.setattr(clients, name, unused)
    await clients.close_clients()


async def test_close_clients_closes_all_open_connections(monkeypatch):
    opened = []
    for name in ("chat_client", "embedding_client", "qdrant", "redis_client", "chat_model"):
        obj = SimpleNamespace(
            close=AsyncMock(),
            aclose=AsyncMock(),
            root_async_client=SimpleNamespace(close=AsyncMock()),
            root_client=SimpleNamespace(close=MagicMock()),
        )
        factory = lru_cache(maxsize=1)(lambda value=obj: value)
        monkeypatch.setattr(clients, name, factory)
        factory()
        opened.append((name, obj, factory))
    opened[0][1].close.side_effect = RuntimeError("close failed")
    await clients.close_clients()
    for name, obj, factory in opened:
        assert factory.cache_info().currsize == 0
        if name == "chat_model":
            obj.root_async_client.close.assert_awaited_once()
            obj.root_client.close.assert_called_once()
        elif name == "redis_client":
            obj.aclose.assert_awaited_once()
        else:
            obj.close.assert_awaited_once()


async def test_upload_is_bounded_and_closed(monkeypatch):
    file = SimpleNamespace(
        filename="a.md", read=AsyncMock(return_value=b"hello"), close=AsyncMock()
    )
    service = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(documents, "ingest_document", service)
    await documents.upload(file)
    file.read.assert_awaited_once_with(documents.get_settings().max_upload_bytes + 1)
    file.close.assert_awaited_once()


async def test_http_validation_and_sse_error_contract(monkeypatch):
    async def broken(*args, **kwargs):
        raise BizError(ResultCode.VECTOR_STORE_UNAVAILABLE, "向量库不可用")
        yield

    monkeypatch.setattr("python_ai.api.chat.stream_answer", broken)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path, payload in [
            ("/api/chat", {"message": " "}),
            ("/api/chat?mode=bad", {"message": "hello"}),
        ]:
            response = await client.post(path, json=payload)
            assert response.status_code == 200 and response.json()["code"] == 1001
        response = await client.get("/api/chat/stream", params={"message": " "})
        assert response.json()["code"] == 1001
        response = await client.post("/api/chat/stream", json={"message": "q"})
        assert "text/event-stream" in response.headers["content-type"]
        assert "event: error" in response.text and "向量库不可用" in response.text
        assert "event: done" not in response.text
