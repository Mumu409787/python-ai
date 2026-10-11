"""会话记忆。对应 Java 的 `RedisChatMemoryStore`，但简化成「最近 N 轮问答」。

存 Redis List，每轮一个 JSON 元素；只保留最近 `session_max_turns` 轮，
整体带 TTL。Redis 挂了就降级成「无记忆」，不影响单轮问答。
"""

from __future__ import annotations

import json
import logging

from .clients import redis_client
from .config import get_settings

logger = logging.getLogger(__name__)


def _key(session_id: str) -> str:
    return f"chat:session:{session_id}"


async def load_history(session_id: str) -> list[tuple[str, str]]:
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


async def append_turn(session_id: str, question: str, answer: str) -> None:
    """追加一轮问答，并修剪长度与刷新 TTL。"""
    settings = get_settings()
    key = _key(session_id)
    try:
        client = redis_client()
        await client.rpush(key, json.dumps({"q": question, "a": answer}, ensure_ascii=False))
        await client.ltrim(key, -settings.session_max_turns, -1)
        await client.expire(key, settings.session_ttl_seconds)
    except Exception as exc:
        logger.warning("写入会话记忆失败（不影响本轮回答）：%s", exc)
