"""健康检查。对应 Java 的 Actuator `/actuator/health`，这里简化成一个自建端点。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from sqlalchemy import text

from ..clients import qdrant, redis_client
from ..config import get_settings
from ..db.session import session_scope
from ..schemas import Result

router = APIRouter(prefix="/api", tags=["health"])


async def _check_postgres() -> bool:
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_qdrant() -> bool:
    try:
        await qdrant().get_collections()
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        return bool(await redis_client().ping())
    except Exception:
        return False


@router.get("/health", response_model=Result[dict])
async def health() -> Result[dict]:
    settings = get_settings()
    pg, qd, rd = await asyncio.gather(_check_postgres(), _check_qdrant(), _check_redis())
    payload = {
        "projectVariant": "current",
        "postgres": pg,
        "qdrant": qd,
        "redis": rd,
        "collection": settings.qdrant_collection,
        "chatModel": settings.deepseek_model,
        "embeddingModel": settings.embedding_model,
        "rerankEnabled": settings.rerank_enabled,
        "hybridEnabled": settings.hybrid_enabled,
        "queryRewriteEnabled": settings.query_rewrite_enabled,
        "memoryBackend": settings.memory_backend,
        "classifierEnabled": settings.classifier_enabled,
        "langfuseConfigured": bool(
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        ),
    }
    return Result.ok(payload)
