"""记录 PDF/图片来源页码，旧片段保留 NULL，无需重建索引。"""

import sqlalchemy as sa

from alembic import op

revision = "0002_source_page"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chunk", sa.Column("source_page", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("chunk", "source_page")
