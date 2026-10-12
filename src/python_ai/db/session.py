"""数据库引擎与会话。

对应 Java 的 `MybatisPlusConfig` + Spring 的 DataSource 自动装配。
懒初始化：只有真正用到数据库时才建连接池，跑纯单元测试不必起 PostgreSQL。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import get_settings
from ..errors import BizError, ResultCode

_engine: AsyncEngine | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_settings().postgres_dsn,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def session_factory() -> async_sessionmaker[AsyncSession]:
    global _factory
    if _factory is None:
        _factory = async_sessionmaker(engine(), expire_on_commit=False)
    return _factory


async def init_db() -> None:
    """只检查迁移版本；修改表结构必须显式运行迁移命令。"""
    from .migrations import assert_current

    async with engine().begin() as conn:
        await conn.run_sync(assert_current)


async def dispose_db() -> None:
    """关闭连接池，进程退出时调用。"""
    global _engine, _factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _factory = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """事务边界：正常退出提交，抛异常回滚。

    ⚠️ 注意 Qdrant 的写入不在这个事务里 —— 参见 `ingest` 里的说明。
    """
    async with session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def document_lock(content_hash: str) -> AsyncIterator[None]:
    """跨进程互斥同内容的上传和删除；事务退出自动释放锁。

    使用非阻塞事务锁，避免等待上传的请求耗尽连接池。业务写入仍用独立事务。
    """
    key = int.from_bytes(bytes.fromhex(content_hash[:16]), "big", signed=True)
    async with session_scope() as session:
        acquired = await session.scalar(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": key}
        )
        if not acquired:
            raise BizError(
                ResultCode.DOC_ALREADY_EXISTS, "相同内容的文档正在处理，请稍后重试", retryable=True
            )
        yield
