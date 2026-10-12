"""显式执行迁移；应用启动仅检查版本，绝不隐式建业务表。"""

from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from alembic import command

from ..config import PROJECT_ROOT
from .models import Base


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table":
        return name in Base.metadata.tables and obj.schema in {None, "public"}
    return True


def configuration():
    return Config(str(PROJECT_ROOT / "alembic.ini"))


def expected_head():
    return ScriptDirectory.from_config(configuration()).get_current_head()


def assert_current(connection):
    current = MigrationContext.configure(connection).get_current_revision()
    if current != expected_head():
        raise RuntimeError(
            f"数据库迁移版本不匹配：当前 {current or '未接管'}，需要 {expected_head()}。"
            "请在项目目录的 PowerShell 运行 .\\.venv\\Scripts\\python.exe "
            "scripts/migrate_db.py；原有 create_all 数据库首次接管请先备份并添加 --adopt-existing。"
        )


def upgrade(connection, *, adopt_existing=False):
    config = configuration()
    config.attributes["connection"] = connection
    current = MigrationContext.configure(connection).get_current_revision()
    tables = set(inspect(connection).get_table_names()) & set(Base.metadata.tables)
    if current is None and tables:
        if not adopt_existing:
            raise RuntimeError("发现未接管的业务表；先备份，再使用 --adopt-existing 核对并接管")
        diffs = compare_metadata(
            MigrationContext.configure(
                connection,
                opts={
                    "include_object": include_object,
                    "compare_type": True,
                },
            ),
            Base.metadata,
        )
        only_page_addition = all(
            isinstance(d, tuple)
            and d[0] == "add_column"
            and d[2] == "chunk"
            and d[3].name == "source_page"
            for d in diffs
        )
        if diffs and not only_page_addition:
            raise RuntimeError(f"现有表与初始版本不一致，拒绝接管：{diffs!r}")
        # Two known schema-only snapshots; future revisions must really run.
        adopted = "0001_initial" if diffs else "0002_source_page"
        command.stamp(config, adopted)
    command.upgrade(config, "head")
