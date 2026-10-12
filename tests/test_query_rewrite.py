"""改写行为、回退及流式/非流式编排回归；无真实模型请求。"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from python_ai import query_rewrite, rag
from python_ai.config import Settings
from python_ai.schemas import ChatMode, QueryInfo
from python_ai.vectorstore import SearchHit


@pytest.fixture
def deps(monkeypatch):
    settings = Settings(_env_file=None, query_rewrite_enabled=True)
    generate = AsyncMock(return_value='{"query":"Refresh Token 的有效期是多久？"}')
    monkeypatch.setattr(query_rewrite, "get_settings", lambda: settings)
    monkeypatch.setattr(query_rewrite, "generate_text", generate)
    return settings, generate


async def test_no_history_and_disabled_do_not_call_model(deps):
    settings, generate = deps
    assert (await query_rewrite.rewrite_query("有效期？", [])).rewrite_status == "no_history"
    settings.query_rewrite_enabled = False
    assert (
        await query_rewrite.rewrite_query("有效期？", [("q", "a")])
    ).rewrite_status == "disabled"
    generate.assert_not_awaited()


async def test_rewrites_followup_with_bounded_history(deps):
    settings, generate = deps
    settings.query_rewrite_history_turns = 1
    history = [("不应出现", "旧历史"), ("Refresh Token 有效期？", "a" * 1000)]
    result = await query_rewrite.rewrite_query("那刷新用的呢？", history)
    assert result.rewrite_status == "rewritten"
    payload = json.loads(generate.call_args.args[1])
    assert len(payload["history"]) == 1 and len(payload["history"][0][1]) == 600
    assert payload["question"] == "那刷新用的呢？"


@pytest.mark.parametrize(
    "raw",
    [
        "不是JSON",
        "[]",
        '{"query":""}',
        '{"query":123}',
        '{"query":"42901 如何处理？"}',  # 新增了不存在的错误码。
        '{"query":"问题一\\n问题二"}',
    ],
)
async def test_invalid_model_output_falls_back(deps, raw):
    deps[1].return_value = raw
    result = await query_rewrite.rewrite_query("那它呢？", [("有效期？", "两小时")])
    assert result.rewrite_status == "fallback" and result.retrieval_query == "那它呢？"


async def test_preserves_explicit_identifier_and_no_history_injection(deps):
    deps[1].return_value = '{"query":"错误码 50001 怎么处理？"}'
    result = await query_rewrite.rewrite_query("42901 怎么处理？", [("50001", "系统错误")])
    assert result.rewrite_status == "fallback"


async def test_negation_is_preserved_and_unchanged_is_valid(deps):
    deps[1].return_value = '{"query":"哪些请求可以重试？"}'
    question = "那哪些请求不能重试？"
    assert (
        await query_rewrite.rewrite_query(question, [("重试", "规则")])
    ).rewrite_status == "fallback"
    deps[1].return_value = json.dumps({"query": question}, ensure_ascii=False)
    assert (
        await query_rewrite.rewrite_query(question, [("重试", "规则")])
    ).rewrite_status == "unchanged"


async def test_rewrite_exception_and_timeout_fall_back(deps):
    settings, generate = deps
    generate.side_effect = RuntimeError("provider secret")
    assert (
        await query_rewrite.rewrite_query("那它呢？", [("q", "a")])
    ).rewrite_status == "fallback"

    async def slow(*args):
        await asyncio.sleep(1)

    settings.query_rewrite_timeout = 0.001
    generate.side_effect = slow
    assert (
        await query_rewrite.rewrite_query("那它呢？", [("q", "a")])
    ).rewrite_status == "fallback"


@pytest.mark.parametrize("streaming", [False, True])
async def test_rag_uses_rewritten_query_but_saves_original_question(monkeypatch, streaming):
    history = [("访问 Token 呢？", "两小时")]
    info = QueryInfo(
        original_question="那刷新用的呢？",
        retrieval_query="Refresh Token 有效期？",
        rewrite_status="rewritten",
    )
    retrieve = AsyncMock(return_value=[SearchHit(0.8, 1, "a.md", 2, 0, "七天")])
    save, rewrite = AsyncMock(), AsyncMock(return_value=info)
    monkeypatch.setattr(rag, "load_history", AsyncMock(return_value=history))
    monkeypatch.setattr(rag, "rewrite_query", rewrite)
    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(rag, "append_turn", save)
    generate = AsyncMock(return_value="七天。[1]")
    monkeypatch.setattr(rag, "generate_text", generate)

    async def stream(system, user, previous):
        assert "那刷新用的呢？" in user and previous == history
        yield "七天。[1]"

    monkeypatch.setattr(rag, "stream_text", stream)
    if streaming:
        events = [
            event async for event in rag.stream_answer(info.original_question, session_id="s")
        ]
        assert [name for name, _ in events] == ["retrieval", "message", "done"]
        assert json.loads(events[-1][1])["queryInfo"]["retrievalQuery"] == info.retrieval_query
    else:
        response = await rag.answer(info.original_question, session_id="s")
        assert response.query_info == info
        assert "那刷新用的呢？" in generate.call_args.args[1]
    retrieve.assert_awaited_once_with(info.retrieval_query)
    rewrite.assert_awaited_once_with(info.original_question, history)
    save.assert_awaited_once_with("s", info.original_question, "七天。[1]")


async def test_chat_never_rewrites_even_with_history(monkeypatch):
    rewrite = AsyncMock()
    monkeypatch.setattr(rag, "rewrite_query", rewrite)
    monkeypatch.setattr(rag, "load_history", AsyncMock(return_value=[("q", "a")]))
    monkeypatch.setattr(rag, "generate_text", AsyncMock(return_value="你好"))
    monkeypatch.setattr(rag, "append_turn", AsyncMock())
    response = await rag.answer("你好", mode=ChatMode.CHAT)
    assert response.query_info is None
    rewrite.assert_not_awaited()
