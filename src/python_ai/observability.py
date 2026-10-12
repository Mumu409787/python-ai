"""Langfuse v4：嵌套调用链、可选正文、错误隔离；没有配置时不建客户端。"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import contextmanager
from functools import lru_cache, wraps

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def tracing_client():
    s = get_settings()
    if not s.langfuse_enabled or not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        base_url=s.langfuse_host or "https://cloud.langfuse.com",
        timeout=5,
    )


class Observation:
    def __init__(self, span=None):
        self.span = span

    @property
    def trace_id(self):
        return self.span.trace_id if self.span else None

    def update(self, **kwargs):
        if self.span:
            try:
                self.span.update(**kwargs)
            except Exception as exc:
                logger.warning("更新调用记录失败：%s", type(exc).__name__)


@contextmanager
def observation(name: str, *, session_id=None, **kwargs):
    manager, propagation = None, None
    obs = Observation()
    try:
        client = tracing_client()
        if client:
            manager = client.start_as_current_observation(name=name, **kwargs)
            obs = Observation(manager.__enter__())
            if session_id:
                from hashlib import sha256

                from langfuse import propagate_attributes

                value = str(session_id)
                if not value.isascii() or len(value) > 200:
                    value = sha256(value.encode()).hexdigest()
                propagation = propagate_attributes(session_id=value)
                propagation.__enter__()
    except Exception as exc:
        logger.warning("调用记录不可用：%s", type(exc).__name__)
    try:
        yield obs
    except BaseException as exc:
        obs.update(
            level="ERROR" if isinstance(exc, Exception) else "WARNING",
            status_message=type(exc).__name__,
        )
        raise
    finally:
        for context in (propagation, manager):
            if context:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:
                    logger.warning("结束调用记录失败：%s", type(exc).__name__)


def traced(name: str, *, as_type="span"):
    def decorate(function):
        def context():
            options = {"as_type": as_type}
            if as_type == "generation":
                options["model"] = get_settings().deepseek_model
            return observation(name, **options)

        if inspect.isasyncgenfunction(function):

            @wraps(function)
            async def stream(*args, **kwargs):
                with context():
                    async for item in function(*args, **kwargs):
                        yield item

            return stream

        @wraps(function)
        async def call(*args, **kwargs):
            with context():
                return await function(*args, **kwargs)

        return call

    return decorate


def generation_usage(usage):
    try:
        client = tracing_client()
        if client and usage:
            client.update_current_generation(
                usage_details={
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                }
            )
    except Exception as exc:
        logger.warning("记录 token 失败：%s", type(exc).__name__)


def trace_link(trace_id):
    try:
        client = tracing_client()
        return client.get_trace_url(trace_id=trace_id) if client and trace_id else None
    except Exception:
        return None


async def close_tracing():
    if tracing_client.cache_info().currsize:
        client = tracing_client()
        if client:
            await asyncio.to_thread(client.shutdown)
    tracing_client.cache_clear()
