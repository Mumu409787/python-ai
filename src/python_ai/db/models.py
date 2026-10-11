"""SQLAlchemy 2.0 声明式模型。

对应 Java 的 `DocumentDO` / `ChunkDO`（MySQL 那两张表）。
Java 侧还有 conversation / message / retrieval_log / ingest_task 四张表，
这里只复刻与 RAG 链路直接相关的两张。

分工：**PostgreSQL 存「关系与状态」，Qdrant 存「向量与原文」**。
chunk 表里的 `vector_id` 是与 Qdrant 的对照键，用于对账和排障。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocStatus:
    """文档处理状态（对齐 Java 的 PENDING/PROCESSING/DONE/FAILED）。"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_name: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str | None] = mapped_column(String(32))
    # SHA-256 十六进制，用于整份文件的秒传去重
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_size: Mapped[int | None]
    chunk_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(16), default=DocStatus.PENDING)
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int]
    # Qdrant 里的 point id（UUID 字符串），删除与对账用
    vector_id: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")
