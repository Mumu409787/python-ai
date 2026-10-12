"""隔离临时数据库验证迁移；不会在主库执行降级。"""

import os
from uuid import uuid4

import psycopg
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from psycopg import sql
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from alembic import command
from python_ai.config import get_settings
from python_ai.db.migrations import assert_current, configuration, include_object, upgrade
from python_ai.db.models import Base

pytestmark = pytest.mark.skipif(os.getenv("RUN_SERVICE_TESTS") != "1", reason="需开启服务联调")


@pytest.fixture
def isolated_database():
    settings = get_settings()
    name = "test_migrations_" + uuid4().hex
    admin = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        autocommit=True,
    )
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    engine = create_engine(
        make_url(settings.postgres_dsn).set(drivername="postgresql+psycopg", database=name)
    )
    try:
        yield engine
    finally:
        engine.dispose()
        assert name.startswith("test_migrations_")
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


def test_empty_database_upgrade_downgrade_upgrade(isolated_database):
    with isolated_database.begin() as connection:
        with pytest.raises(RuntimeError, match="迁移版本不匹配"):
            assert_current(connection)
        upgrade(connection)
        assert_current(connection)
        assert not compare_metadata(
            MigrationContext.configure(connection, opts={"include_object": include_object}),
            Base.metadata,
        )
        config = configuration()
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
        assert "document" not in inspect(connection).get_table_names()
        command.upgrade(config, "head")
        assert_current(connection)


def test_existing_database_adoption_requires_matching_schema(isolated_database):
    with isolated_database.begin() as connection:
        Base.metadata.create_all(connection)
        with pytest.raises(RuntimeError, match="先备份"):
            upgrade(connection)
        connection.execute(text("ALTER TABLE document DROP COLUMN error_msg"))
        with pytest.raises(RuntimeError, match="不一致"):
            upgrade(connection, adopt_existing=True)
        connection.execute(text("ALTER TABLE document ADD COLUMN error_msg TEXT"))
        upgrade(connection, adopt_existing=True)
        assert_current(connection)
