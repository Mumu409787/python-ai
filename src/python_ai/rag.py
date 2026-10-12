"""问答编排。对应 Java 的 `RagAssistant` + `ChatService`。

**与 Java 侧的一处改进**：Java 的同步 RAG 会检索两次 —— 一次手动检索拿引用给前端，
一次框架内部再检索喂 prompt（它自己的 README 列为已知取舍）。
这里只检索一次，命中同时用于「返回引用」和「组装 prompt」，embedding 开销减半。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from .classification import classify
from .errors import friendly_llm_message
from .llm import generate_text, stream_text
from .memory import append_turn, load_history
from .prompts import CHAT_SYSTEM, RAG_SYSTEM, REFUSAL_TEXT, build_rag_user_prompt
from .query_rewrite import rewrite_query
from .retrieval import build_contexts, retrieve, to_dtos
from .schemas import ChatMode, ChatResponse, DonePayload


def new_session_id() -> str:
    return f"s-{uuid4()}"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def _answer(
    question: str,
    *,
    session_id: str | None = None,
    mode: ChatMode = ChatMode.RAG,
) -> ChatResponse:
    """非流式问答。"""
    started = time.perf_counter()
    sid = session_id or new_session_id()
    use_rag = mode is ChatMode.RAG

    history = await load_history(sid) if use_rag else await load_history(sid, mode="CHAT")
    query_info = await rewrite_query(question, history) if use_rag else None
    hits = await retrieve(query_info.retrieval_query) if query_info else []

    if use_rag:
        system, user = RAG_SYSTEM, build_rag_user_prompt(question, build_contexts(hits))
    else:
        system, user = CHAT_SYSTEM, question

    text = REFUSAL_TEXT if use_rag and not hits else await generate_text(system, user, history)
    saved = (
        await append_turn(sid, question, text)
        if use_rag
        else await append_turn(sid, question, text, mode="CHAT")
    )

    return ChatResponse(
        session_id=sid,
        answer=text,
        hits=to_dtos(hits),
        cost_ms=_elapsed_ms(started),
        used_rag=use_rag,
        query_info=query_info,
        memory_saved=saved is not False,
    )


async def _stream_answer(
    question: str,
    *,
    session_id: str | None = None,
    mode: ChatMode = ChatMode.RAG,
) -> AsyncIterator[tuple[str, str]]:
    """流式问答，产出 `(事件名, 数据)` 序列。

    事件协议与 Java 侧逐字一致：
        retrieval(JSON 数组) → message(裸文本) × N → done(JSON)
    **没有 `[DONE]` 哨兵**，结束信号就是 `done` 事件 + 连接关闭。
    """
    started = time.perf_counter()
    sid = session_id or new_session_id()
    use_rag = mode is ChatMode.RAG

    history = await load_history(sid) if use_rag else await load_history(sid, mode="CHAT")
    query_info = await rewrite_query(question, history) if use_rag else None
    hits = await retrieve(query_info.retrieval_query) if query_info else []
    if use_rag:
        payload = [dto.model_dump(by_alias=True) for dto in to_dtos(hits)]
        yield "retrieval", json.dumps(payload, ensure_ascii=False)

    if use_rag:
        system, user = RAG_SYSTEM, build_rag_user_prompt(question, build_contexts(hits))
    else:
        system, user = CHAT_SYSTEM, question

    collected: list[str] = []
    try:
        if use_rag and not hits:
            collected.append(REFUSAL_TEXT)
            yield "message", REFUSAL_TEXT
        else:
            async for token in stream_text(system, user, history):
                collected.append(token)
                yield "message", token
    except Exception as exc:
        # 已经吐了一部分也照样报错收尾 —— 不静默截断
        yield "error", friendly_llm_message(exc)
        return

    text = "".join(collected)
    saved = (
        await append_turn(sid, question, text)
        if use_rag
        else await append_turn(sid, question, text, mode="CHAT")
    )
    done = DonePayload(
        session_id=sid,
        cost_ms=_elapsed_ms(started),
        query_info=query_info,
        memory_saved=saved is not False,
    )
    yield "done", done.model_dump_json(by_alias=True)


async def answer(question: str, *, session_id=None, mode=ChatMode.RAG) -> ChatResponse:
    from .config import get_settings
    from .observability import observation, trace_link

    sid = session_id or new_session_id()
    with observation(
        "chat.answer",
        session_id=sid,
        metadata={"mode": str(mode)},
        input=question
        if get_settings().langfuse_capture_content
        else {"questionChars": len(question)},
    ) as span:
        result = await _answer(question, session_id=sid, mode=mode)
        result.question_category = classify(question)
        span.update(
            metadata={"mode": str(mode), "questionCategory": result.question_category.model_dump()}
        )
        result.trace_id = span.trace_id
        result.trace_url = trace_link(span.trace_id)
        span.update(
            output=result.answer
            if get_settings().langfuse_capture_content
            else {
                "answerChars": len(result.answer),
                "hitCount": len(result.hits),
                "memorySaved": result.memory_saved,
            }
        )
        return result


async def stream_answer(question: str, *, session_id=None, mode=ChatMode.RAG):
    from .config import get_settings
    from .observability import observation, trace_link

    sid = session_id or new_session_id()
    with observation(
        "chat.stream",
        session_id=sid,
        metadata={"mode": str(mode)},
        input=question
        if get_settings().langfuse_capture_content
        else {"questionChars": len(question)},
    ) as span:
        async for event, data in _stream_answer(question, session_id=sid, mode=mode):
            if event == "done":
                payload = json.loads(data)
                payload["questionCategory"] = classify(question).model_dump(by_alias=True)
                payload.update(traceId=span.trace_id, traceUrl=trace_link(span.trace_id))
                span.update(output={"completed": True, "memorySaved": payload.get("memorySaved")})
                data = json.dumps(payload, ensure_ascii=False)
            elif event == "error":
                span.update(level="ERROR", status_message="generation_failed")
            yield event, data
