"""数据访问层。

对应 Java 的 `DocumentMapper` / `ChunkMapper`。写成接受 `session` 的普通函数，
事务边界由调用方（`ingest.py`）控制 —— 需要跨事务编排的地方能看得更清楚。
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import Chunk, DocStatus, Document


async def find_document_by_hash(session: AsyncSession, content_hash: str) -> Document | None:
    result = await session.execute(select(Document).where(Document.content_hash == content_hash))
    return result.scalar_one_or_none()


async def create_document(
    session: AsyncSession,
    *,
    doc_name: str,
    source_type: str | None,
    content_hash: str,
    file_size: int,
) -> Document:
    doc = Document(
        doc_name=doc_name,
        source_type=source_type,
        content_hash=content_hash,
        file_size=file_size,
        chunk_count=0,
        status=DocStatus.PENDING,
    )
    session.add(doc)
    await session.flush()  # 拿到自增 id
    return doc


async def add_chunks(session: AsyncSession, doc_id: int, texts: list[str]) -> list[Chunk]:
    """写入切片。`vector_id` 在这里生成 —— 后续写 Qdrant 要用它做 point id。"""
    chunks = [
        Chunk(
            doc_id=doc_id,
            chunk_index=index,
            vector_id=str(uuid4()),
            content=text,
        )
        for index, text in enumerate(texts)
    ]
    session.add_all(chunks)
    await session.flush()  # 拿到自增 id
    return chunks


async def get_document(session: AsyncSession, doc_id: int) -> Document | None:
    return await session.get(Document, doc_id)


async def list_documents(session: AsyncSession, *, limit: int = 100) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.id.desc()).limit(limit))
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, doc: Document) -> None:
    """删文档。chunk 靠 `cascade` 一起删。"""
    await session.delete(doc)


async def delete_chunks_of(session: AsyncSession, doc_id: int) -> None:
    await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))


async def mark_status(
    session: AsyncSession,
    doc_id: int,
    status: str,
    *,
    chunk_count: int | None = None,
    error_msg: str | None = None,
) -> None:
    doc = await session.get(Document, doc_id)
    if doc is None:
        return
    doc.status = status
    doc.error_msg = error_msg
    if chunk_count is not None:
        doc.chunk_count = chunk_count


async def count_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Document))
    return int(result.scalar_one())
