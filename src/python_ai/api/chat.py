"""对话接口。对应 Java 的 `ChatController`。

SSE 端点返回的是**裸事件流**，不包 `Result` —— 与 Java 侧一致
（流式场景下每帧包一层 code/message 反而累赘）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from ..errors import BizError, ResultCode, friendly_llm_message
from ..rag import answer, new_session_id, stream_answer
from ..schemas import ChatMode, ChatRequest, ChatResponse, Result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _parse_mode(raw: str | None) -> ChatMode:
    try:
        return ChatMode.parse(raw)
    except ValueError as exc:
        raise BizError(ResultCode.PARAM_INVALID, str(exc)) from exc


@router.post("", response_model=Result[ChatResponse])
async def chat(
    request: ChatRequest,
    mode: str = Query("RAG", description="CHAT 或 RAG"),
) -> Result[ChatResponse]:
    """非流式问答。"""
    response = await answer(
        request.message,
        session_id=request.session_id,
        mode=_parse_mode(mode),
    )
    return Result.ok(response)


@router.post("/stream")
async def chat_stream_post(
    request: ChatRequest,
    mode: str = Query("RAG", description="CHAT 或 RAG"),
) -> EventSourceResponse:
    return _sse_response(request.message, request.session_id, _parse_mode(mode))


@router.get("/stream")
async def chat_stream_get(
    message: str = Query(min_length=1, max_length=4000),
    sessionId: str | None = Query(None),  # noqa: N803 —— 保持与 Java 侧的参数名一致
    mode: str = Query("RAG"),
) -> EventSourceResponse:
    return _sse_response(message, sessionId, _parse_mode(mode))


def _sse_response(message: str, session_id: str | None, mode: ChatMode) -> EventSourceResponse:
    """把 `(事件名, 数据)` 序列转成 SSE 帧。

    事件协议与 Java 侧逐字一致：`retrieval` / `message` / `done` / `error`。
    `message` 的 data 是**裸文本**，其余是 JSON 字符串。
    """
    return EventSourceResponse(_event_stream(message, session_id, mode))


async def _event_stream(
    message: str, session_id: str | None, mode: ChatMode
) -> AsyncIterator[dict[str, str]]:
    sid = session_id or new_session_id()
    try:
        async for event, data in stream_answer(message, session_id=sid, mode=mode):
            yield {"event": event, "data": data}
    except BizError as exc:
        # 检索阶段就失败的情况（如 Qdrant 不可用）—— 也要走 error 事件，
        # 而不是让连接直接断掉
        logger.warning("流式问答失败: %s", exc)
        yield {"event": "error", "data": exc.message}
    except Exception as exc:  # noqa: BLE001 —— 兜底，保证客户端能收到结束信号
        logger.exception("流式问答出现未预期异常")
        yield {"event": "error", "data": friendly_llm_message(exc)}
