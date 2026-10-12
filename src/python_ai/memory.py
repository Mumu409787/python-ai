"""默认 LangGraph/PostgreSQL 完整轮次记忆；Redis 为可显式选择的旧实现。"""

from __future__ import annotations

import json
import logging

from .clients import redis_client
from .config import get_settings
from .observability import traced

logger = logging.getLogger(__name__)


def _key(session_id: str) -> str:
    return f"chat:session:{session_id}"


async def _load_redis(session_id: str) -> list[tuple[str, str]]:
    """读取最近的 (问题, 回答) 列表，按时间正序。"""
    try:
        raw = await redis_client().lrange(_key(session_id), -get_settings().session_max_turns, -1)
    except Exception as exc:
        logger.warning("读取会话记忆失败，按无记忆处理：%s", exc)
        return []

    turns: list[tuple[str, str]] = []
    for item in raw:
        try:
            data = json.loads(item)
        except json.JSONDecodeError:
            continue
        # 不用 `except (A, B)` 元组写法：ruff 0.16.8 的 formatter 会把它
        # 改写成 Python 2 的 `except A, B:`（无效语法），改用显式判断规避
        if isinstance(data, dict) and "q" in data and "a" in data:
            turns.append((str(data["q"]), str(data["a"])))
    return turns


async def _append_redis(session_id: str, question: str, answer: str) -> bool:
    """追加一轮问答，并修剪长度与刷新 TTL。"""
    settings = get_settings()
    key = _key(session_id)
    try:
        client = redis_client()
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps({"q": question, "a": answer}, ensure_ascii=False))
            pipe.ltrim(key, -settings.session_max_turns, -1)
            pipe.expire(key, settings.session_ttl_seconds)
            await pipe.execute()
        return True
    except Exception as exc:
        logger.warning("写入会话记忆失败（不影响本轮回答）：%s", exc)
        return False


@traced("load_history")
async def load_history(session_id: str, *, mode: str = "RAG") -> list[tuple[str, str]]:
    """默认从 PostgreSQL Checkpointer 读取；Redis 仅作显式回滚选项。"""
    if get_settings().memory_backend == "redis":
        return await _load_redis(f"{mode}:{session_id}" if mode != "RAG" else session_id)
    from .checkpoint_memory import load

    try:
        return await load(session_id, mode)
    except Exception as exc:
        logger.warning("读取检查点失败，按无记忆处理：%s", type(exc).__name__)
        return []


@traced("append_turn")
async def append_turn(session_id: str, question: str, answer: str, *, mode: str = "RAG") -> bool:
    if get_settings().memory_backend == "redis":
        return await _append_redis(
            f"{mode}:{session_id}" if mode != "RAG" else session_id, question, answer
        )
    from .checkpoint_memory import append

    try:
        await append(session_id, question, answer, mode)
        return True
    except Exception as exc:
        logger.warning("保存检查点失败：%s", type(exc).__name__)
        return False
