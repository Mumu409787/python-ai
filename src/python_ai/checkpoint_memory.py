"""完成问答的 LangGraph 状态；同步 PG saver 在线程池运行，兼容 Windows 事件循环。

只保留最新检查点：读取、删除旧检查点、写入完整受限历史在同一 PG 事务内完成。
这是会话记忆持久化，不提供中断后恢复生成或历史时间旅行。
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from typing import TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None
SCHEMA = "python_ai_memory"
_init_lock = threading.Lock()


class MemoryState(TypedDict):
    turns: list[list[str]]
    expires_at: float


def _completed(state: MemoryState):
    return state


def _graph(saver):
    builder = StateGraph(MemoryState)
    builder.add_node("completed_turn", _completed)
    builder.add_edge(START, "completed_turn")
    builder.add_edge("completed_turn", END)
    return builder.compile(checkpointer=saver)


def thread_id(sid: str, mode: str = "RAG") -> str:
    return hashlib.sha256(f"{mode}\0{sid}".encode()).hexdigest()


def _lock_key(tid):
    return int.from_bytes(bytes.fromhex(tid[:16]), "big", signed=True)


def _create_pool():
    global _pool
    if _pool is not None:
        return
    s = get_settings()
    candidate = ConnectionPool(
        kwargs=dict(
            host=s.postgres_host,
            port=s.postgres_port,
            dbname=s.postgres_db,
            user=s.postgres_user,
            password=s.postgres_password,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=5,
        ),
        min_size=1,
        max_size=5,
        timeout=10,
        open=False,
    )
    try:
        candidate.open(wait=True, timeout=10)
        with candidate.connection() as connection:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            connection.execute(f"SET search_path TO {SCHEMA}, public")
            PostgresSaver(connection).setup()
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {SCHEMA}.sessions "
                "(thread_id text PRIMARY KEY, expires_at double precision NOT NULL)"
            )
        _pool = candidate
    except BaseException:
        candidate.close()
        raise


def _initialize():
    with _init_lock:
        _create_pool()


def _operate(sid, mode, turn=None, delete=False):
    _initialize()
    tid = thread_id(sid, mode)
    with _pool.connection() as connection:
        # SET per checkout: every connection, including newly grown pool slots.
        connection.execute(f"SET search_path TO {SCHEMA}, public")
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(tid),))
            saver = PostgresSaver(connection)
            config = {"configurable": {"thread_id": tid}}
            graph = _graph(saver)
            if delete:
                saver.delete_thread(tid)
                connection.execute("DELETE FROM sessions WHERE thread_id=%s", (tid,))
                return []
            state = graph.get_state(config).values
            if state and state.get("expires_at", 0) <= time.time():
                saver.delete_thread(tid)
                connection.execute("DELETE FROM sessions WHERE thread_id=%s", (tid,))
                state = {}
            turns = state.get("turns", []) if state else []
            if turn is not None:
                settings = get_settings()
                turns = (turns + [list(turn)])[-settings.session_max_turns :]
                expires = time.time() + settings.session_ttl_seconds
                # Atomic replacement avoids accumulating per-turn checkpoint snapshots.
                saver.delete_thread(tid)
                graph.invoke({"turns": turns, "expires_at": expires}, config, durability="exit")
                connection.execute(
                    "INSERT INTO sessions VALUES (%s,%s) ON CONFLICT(thread_id) "
                    "DO UPDATE SET expires_at=excluded.expires_at",
                    (tid, expires),
                )
            return [tuple(t) for t in turns]


def _cleanup():
    if _pool is None:
        return
    with _pool.connection() as connection:
        connection.execute(f"SET search_path TO {SCHEMA}, public")
        expired = connection.execute(
            "SELECT thread_id FROM sessions WHERE expires_at < %s LIMIT 100", (time.time(),)
        ).fetchall()
        for row in expired:
            tid = row["thread_id"]
            with connection.transaction():
                locked = connection.execute(
                    "SELECT pg_try_advisory_xact_lock(%s) AS locked", (_lock_key(tid),)
                ).fetchone()["locked"]
                if not locked:
                    continue
                current = connection.execute(
                    "SELECT expires_at FROM sessions WHERE thread_id=%s", (tid,)
                ).fetchone()
                if current and current["expires_at"] < time.time():
                    PostgresSaver(connection).delete_thread(tid)
                    connection.execute("DELETE FROM sessions WHERE thread_id=%s", (tid,))


async def initialize():
    await asyncio.to_thread(_initialize)
    await asyncio.to_thread(_cleanup)


async def load(sid, mode="RAG"):
    return await asyncio.to_thread(_operate, sid, mode)


async def append(sid, question, answer, mode="RAG"):
    await asyncio.to_thread(_operate, sid, mode, (question, answer))


async def forget(sid, mode="RAG"):
    await asyncio.to_thread(_operate, sid, mode, delete=True)


async def cleanup():
    await asyncio.to_thread(_cleanup)


async def close():
    global _pool
    if _pool is not None:
        await asyncio.to_thread(_pool.close)
    _pool = None
