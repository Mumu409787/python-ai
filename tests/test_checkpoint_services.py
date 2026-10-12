"""真实 PostgreSQL 检查点验收；不调用模型，独立 session，用后清理。"""

import asyncio
import os
from uuid import uuid4

import pytest

from python_ai import checkpoint_memory as cp
from python_ai.config import get_settings

pytestmark = pytest.mark.skipif(os.getenv("RUN_SERVICE_TESTS") != "1", reason="需开启服务联调")


async def test_durable_isolated_bounded_concurrent_history(monkeypatch):
    sid = "checkpoint-qa-" + uuid4().hex
    monkeypatch.setattr(get_settings(), "session_max_turns", 3)
    try:
        await cp.initialize()
        await cp.append(sid, "访问令牌", "两小时")
        await cp.close()
        # 关闭连接池并重新打开，状态仍来自数据库，而非进程缓存。
        assert await cp.load(sid) == [("访问令牌", "两小时")]
        assert await cp.load(sid, "CHAT") == []
        await cp.append(sid, "你好", "你好", "CHAT")
        await asyncio.gather(*(cp.append(sid, f"q{i}", f"a{i}") for i in range(6)))
        history = await cp.load(sid)
        assert len(history) == 3 and len(set(history)) == 3
        assert await cp.load(sid, "CHAT") == [("你好", "你好")]
        assert await cp.load(sid + "-other") == []
        with cp._pool.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) AS n FROM python_ai_memory.checkpoints WHERE thread_id=%s",
                    (cp.thread_id(sid),),
                ).fetchone()["n"]
                == 1
            )
    finally:
        await cp.forget(sid)
        await cp.forget(sid, "CHAT")
        await cp.close()


async def test_replacement_rollback_and_expiry(monkeypatch):
    sid = "checkpoint-qa-" + uuid4().hex
    try:
        await cp.append(sid, "q", "a")
        original = cp._graph

        def broken(saver):
            graph = original(saver)

            def fail(*a, **kw):
                raise RuntimeError("simulated save failure")

            graph.invoke = fail
            return graph

        with monkeypatch.context() as patch:
            patch.setattr(cp, "_graph", broken)
            with pytest.raises(RuntimeError, match="simulated"):
                await cp.append(sid, "new", "lost")
        assert await cp.load(sid) == [("q", "a")]
        with cp._pool.connection() as conn:
            conn.execute(
                "UPDATE python_ai_memory.sessions SET expires_at=0 WHERE thread_id=%s",
                (cp.thread_id(sid),),
            )
        await cp.cleanup()
        assert await cp.load(sid) == []
    finally:
        await cp.forget(sid)
        await cp.close()
