"""BM25/RRF 离线行为测试；不连接数据库、不请求模型。"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from python_ai import lexical, retrieval
from python_ai.config import Settings
from python_ai.vectorstore import SearchHit


def hit(chunk_id, content="正文", score=0.8):
    return SearchHit(score, 1, "a.md", chunk_id, chunk_id, content)


def test_tokenization_preserves_codes_paths_and_chinese():
    terms = lexical.tokenize("错误码42901，接口 /v1/tasks，Refresh Token，苍鹭服务")
    assert {"42901", "/v1/tasks", "refresh", "token", "苍鹭", "服务"} <= set(terms)
    assert lexical.tokenize("ＡＢＣ１２３") == ["abc123"]


def test_single_chunk_and_common_term_remain_searchable():
    assert lexical.rank_chunks("Token", [hit(1, "Token 有效期是两小时")], 5)[0].chunk_id == 1
    chunks = [hit(1, "Token 访问"), hit(2, "Token 刷新")]
    assert len(lexical.rank_chunks("Token", chunks, 5)) == 2


def test_bm25_exact_code_and_unrelated_query():
    chunks = [hit(1, "错误码 42901 是请求限流"), hit(2, "错误码 50001 是系统错误")]
    assert [h.chunk_id for h in lexical.rank_chunks("42901", chunks, 5)] == [1]
    assert lexical.rank_chunks("火锅美食", chunks, 5) == []
    assert lexical.rank_chunks("？！", chunks, 5) == []
    assert lexical.rank_chunks("42901", [], 5) == []


def test_rrf_deduplicates_and_uses_rank_instead_of_score():
    a, b, c = hit(1, score=0.8), hit(2, score=0.7), hit(3, score=900)
    fused = retrieval.fuse_rrf([a, b, a], [b, c])
    assert [h.chunk_id for h in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert all(h.score_kind == "rrf" for h in fused)
    assert a.score == 0.8  # 融合不修改原始候选。


@pytest.fixture
def config(monkeypatch):
    settings = Settings(_env_file=None, rerank_enabled=False)
    monkeypatch.setattr(retrieval, "get_settings", lambda: settings)
    return settings


async def test_hybrid_rescues_bm25_only_hit(config, monkeypatch):
    monkeypatch.setattr(retrieval, "vector_recall", AsyncMock(return_value=[]))
    monkeypatch.setattr(retrieval, "bm25_recall", AsyncMock(return_value=[hit(2, "42901")]))
    result = await retrieval.retrieve("42901", top_k=1)
    assert len(result) == 1 and result[0].chunk_id == 2 and result[0].score_kind == "rrf"
    assert result[0].score < config.retrieval_score_threshold


async def test_rerank_failure_keeps_bm25_rescue(config, monkeypatch):
    config.rerank_enabled = True
    monkeypatch.setattr(retrieval, "vector_recall", AsyncMock(return_value=[]))
    monkeypatch.setattr(retrieval, "bm25_recall", AsyncMock(return_value=[hit(2, "42901")]))
    monkeypatch.setattr(retrieval, "_rerank", AsyncMock(side_effect=TimeoutError()))
    assert (await retrieval.retrieve("42901"))[0].chunk_id == 2


async def test_disabling_hybrid_does_not_access_bm25(config, monkeypatch):
    config.hybrid_enabled = False
    dense, bm25 = AsyncMock(return_value=[hit(1)]), AsyncMock()
    monkeypatch.setattr(retrieval, "vector_recall", dense)
    monkeypatch.setattr(retrieval, "bm25_recall", bm25)
    assert (await retrieval.retrieve("问题"))[0].score_kind == "vector"
    bm25.assert_not_awaited()


async def test_bm25_failure_falls_back_but_vector_failure_surfaces(config, monkeypatch):
    monkeypatch.setattr(retrieval, "vector_recall", AsyncMock(return_value=[hit(1)]))
    monkeypatch.setattr(retrieval, "bm25_recall", AsyncMock(side_effect=RuntimeError()))
    assert (await retrieval.retrieve("问题"))[0].score_kind == "vector"
    monkeypatch.setattr(retrieval, "vector_recall", AsyncMock(side_effect=TimeoutError("vector")))
    with pytest.raises(TimeoutError, match="vector"):
        await retrieval.retrieve("问题")


async def test_lexical_refreshes_snapshot_and_revalidates_deleted_doc(monkeypatch):
    @asynccontextmanager
    async def scope():
        yield SimpleNamespace()

    rows = AsyncMock(side_effect=[[(1, "a.md", 1, 0, "42901 限流")], []])
    monkeypatch.setattr(lexical, "session_scope", scope)
    monkeypatch.setattr(lexical, "searchable_chunks", rows)
    monkeypatch.setattr(lexical, "done_document_ids", AsyncMock(return_value={1}))
    assert await lexical.bm25_recall("42901", top_k=5)
    assert await lexical.bm25_recall("42901", top_k=5) == []
    rows.side_effect = None
    rows.return_value = [(1, "a.md", 1, 0, "42901 限流")]
    monkeypatch.setattr(lexical, "done_document_ids", AsyncMock(return_value=set()))
    assert await lexical.bm25_recall("42901", top_k=5) == []


@pytest.mark.parametrize(
    "fields",
    [
        {"bm25_candidate_k": 0},
        {"bm25_candidate_k": 101},
        {"hybrid_rrf_k": 0},
        {"query_rewrite_timeout": 0},
        {"query_rewrite_history_turns": 11},
    ],
)
def test_invalid_new_settings_rejected(fields):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **fields)
