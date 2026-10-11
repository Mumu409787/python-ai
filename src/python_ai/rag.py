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

from .errors import friendly_llm_message
from .llm import generate_text, stream_text
from .memory import append_turn, load_history
from .prompts import CHAT_SYSTEM, RAG_SYSTEM, build_rag_user_prompt
from .retrieval import build_contexts, retrieve, to_dtos
from .schemas import ChatMode, ChatResponse, DonePayload


def new_session_id() -> str:
    return f"s-{uuid4()}"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def answer(
    question: str,
    *,
    session_id: str | None = None,
    mode: ChatMode = ChatMode.RAG,
) -> ChatResponse:
    """非流式问答。"""
    started = time.perf_counter()
    sid = session_id or new_session_id()
    use_rag = mode is ChatMode.RAG

    hits = await retrieve(question) if use_rag else []
    history = await load_history(sid)

    if use_rag:
        system, user = RAG_SYSTEM, build_rag_user_prompt(question, build_contexts(hits))
    else:
        system, user = CHAT_SYSTEM, question

    text = await generate_text(system, user, history)
    await append_turn(sid, question, text)

    return ChatResponse(
        session_id=sid,
        answer=text,
        hits=to_dtos(hits),
        cost_ms=_elapsed_ms(started),
        used_rag=use_rag,
    )


async def stream_answer(
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

    hits = await retrieve(question) if use_rag else []
    if hits:
        payload = [dto.model_dump(by_alias=True) for dto in to_dtos(hits)]
        yield "retrieval", json.dumps(payload, ensure_ascii=False)

    history = await load_history(sid)
    if use_rag:
        system, user = RAG_SYSTEM, build_rag_user_prompt(question, build_contexts(hits))
    else:
        system, user = CHAT_SYSTEM, question

    collected: list[str] = []
    try:
        async for token in stream_text(system, user, history):
            collected.append(token)
            yield "message", token
    except Exception as exc:
        # 已经吐了一部分也照样报错收尾 —— 不静默截断
        yield "error", friendly_llm_message(exc)
        return

    text = "".join(collected)
    await append_turn(sid, question, text)
    done = DonePayload(session_id=sid, cost_ms=_elapsed_ms(started))
    yield "done", done.model_dump_json(by_alias=True)
