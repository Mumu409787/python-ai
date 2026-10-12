"""复用应用配置；口令不写入 ini，不记录 DSN。仅管理 public 的业务表。"""

import asyncio

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from python_ai.config import get_settings
from python_ai.db.migrations import include_object
from python_ai.db.models import Base

config = context.config


def run(connection):
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = create_async_engine(get_settings().postgres_dsn, poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=get_settings().postgres_dsn,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
elif config.attributes.get("connection") is not None:
    run(config.attributes["connection"])
else:
    asyncio.run(online())
