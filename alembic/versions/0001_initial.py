"""知识库业务表初始版本。Checkpoint 私有 schema 由对应 saver 管理。"""

import sqlalchemy as sa

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "document",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("doc_name", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_document_content_hash", "document", ["content_hash"], unique=True)
    op.create_table(
        "chunk",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "doc_id", sa.Integer(), sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("vector_id", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
    )
    op.create_index("ix_chunk_doc_id", "chunk", ["doc_id"])


def downgrade():
    op.drop_table("chunk")
    op.drop_table("document")
