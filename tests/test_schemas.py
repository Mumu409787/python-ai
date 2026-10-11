"""DTO、统一响应体与模式解析。不碰网络、不需要 API Key。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from python_ai.schemas import ChatMode, ChatRequest, Result, RetrievalHit


def test_fields_serialize_as_camel_case() -> None:
    """对外字段必须是 camelCase —— 前端要能和 Java 项目共用。"""
    request = ChatRequest(message="你好", sessionId="s-1")
    payload = request.model_dump(by_alias=True)

    assert payload["sessionId"] == "s-1"
    assert "session_id" not in payload


def test_internal_construction_accepts_snake_case() -> None:
    assert ChatRequest(message="hi", session_id="s-2").session_id == "s-2"


def test_retrieval_hit_serializes_camel_case() -> None:
    hit = RetrievalHit(chunk_id=1, doc_id=2, doc_name="a.md", content="x", score=0.9)
    payload = hit.model_dump(by_alias=True)

    assert payload == {
        "chunkId": 1,
        "docId": 2,
        "docName": "a.md",
        "content": "x",
        "score": 0.9,
    }


def test_message_is_validated() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 4001)


def test_result_ok_and_fail() -> None:
    ok = Result.ok({"a": 1})
    assert ok.code == 0 and ok.message == "成功" and ok.data == {"a": 1}

    bad = Result.fail(1002, "文档不存在")
    assert bad.code == 1002 and bad.data is None
    assert bad.model_dump(by_alias=True)["message"] == "文档不存在"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ChatMode.RAG),
        ("", ChatMode.RAG),
        ("   ", ChatMode.RAG),
        ("rag", ChatMode.RAG),
        ("RAG", ChatMode.RAG),
        (" chat ", ChatMode.CHAT),
    ],
)
def test_chat_mode_parse(raw: str | None, expected: ChatMode) -> None:
    assert ChatMode.parse(raw) is expected


def test_chat_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="未知的对话模式"):
        ChatMode.parse("AGENT")
