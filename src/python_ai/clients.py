"""外部客户端的装配与生命周期。

对应 Java 的 `ChatModelConfig` / `EmbeddingModelConfig` / `VectorStoreConfig`。

用 `lru_cache` 做进程级单例：同一进程内到处 `chat_client()` 拿到的都是同一个实例，
连接池才不会被反复重建。进程退出前调用 `close_clients()` 释放。
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def chat_client() -> AsyncOpenAI:
    """DeepSeek 的原始 SDK 客户端（用于非 LangChain 场景，如直接诊断）。"""
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        timeout=s.chat_timeout,
        max_retries=s.chat_max_retries,
    )


@lru_cache(maxsize=1)
def embedding_client() -> AsyncOpenAI:
    """硅基流动的 embedding 客户端。

    刻意用原生 SDK 而不是 `OpenAIEmbeddings`：后者的 `check_embedding_ctx_length`
    会先做 tokenize，还有可能自动补 `dimensions`，两者都会让 bge-m3 报 400。
    """
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.siliconflow_api_key,
        base_url=s.siliconflow_base_url,
        timeout=s.embedding_timeout,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def chat_model() -> ChatOpenAI:
    """LangChain 的对话模型封装，供 RAG 编排使用。

    ⚠️ deepseek-flash 是推理模型，`max_tokens` 由思维链和正文共享。
    """
    s = get_settings()
    return ChatOpenAI(
        model=s.deepseek_model,
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        temperature=s.chat_temperature,
        max_tokens=s.chat_max_tokens,
        timeout=s.chat_timeout,
        max_retries=s.chat_max_retries,
    )


@lru_cache(maxsize=1)
def qdrant() -> AsyncQdrantClient:
    """Qdrant 客户端。注意端口是 6335（Java 项目占的是 6333）。"""
    return AsyncQdrantClient(url=get_settings().qdrant_url)


@lru_cache(maxsize=1)
def redis_client() -> Redis:
    """Redis 客户端（会话记忆用，DB 0）。"""
    return Redis.from_url(
        get_settings().redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


async def close_clients() -> None:
    """释放连接。在 FastAPI 的 lifespan 退出阶段调用。"""
    pending = []
    try:
        for factory in (chat_client, embedding_client, qdrant):
            if factory.cache_info().currsize:
                pending.append(factory().close())
        if redis_client.cache_info().currsize:
            pending.append(redis_client().aclose())
        if chat_model.cache_info().currsize:
            model = chat_model()
            pending.append(model.root_async_client.close())
            try:
                model.root_client.close()
            except Exception:
                logger.exception("关闭同步对话客户端失败")
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("关闭客户端失败：%s", result)
    finally:
        for factory in (chat_client, embedding_client, chat_model, qdrant, redis_client):
            factory.cache_clear()
