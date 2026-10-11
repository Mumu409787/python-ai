"""入库 / 删除编排。对应 Java 的 `DocumentService`。

**与 Java 侧最重要的一处结构差异**：Java 把「写 Qdrant」放在 `@Transactional` 里，
它自己的 README 把「Qdrant 写完但事务回滚 → 孤儿向量」列为已知缺陷。
这里改成 **事务外写向量**：

    事务1（建记录 + 存切片，status=PENDING）→ 提交
      ↓
    事务外（向量化 + 写 Qdrant）           ← 失败了 DB 里留下 FAILED 记录可查
      ↓
    事务2（标记 DONE）

这样任何一步失败都不会产生「数据库没有、向量库却有」的孤儿向量。
代价是多一次事务往返，换来的是可对账的状态机。
"""

from __future__ import annotations

import time

from . import repository as repo
from .chunking import sha256_hex, split_text
from .config import get_settings
from .db.models import DocStatus, Document
from .db.session import session_scope
from .embedding import embed_texts
from .errors import BizError, ResultCode
from .parsing import parse_document, source_type_of
from .schemas import DocumentOut, IngestResult
from .vectorstore import VectorPoint, delete_by_doc, upsert_points


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        doc_name=doc.doc_name,
        source_type=doc.source_type,
        content_hash=doc.content_hash,
        file_size=doc.file_size,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_msg=doc.error_msg,
        created_at=doc.created_at,
    )


async def ingest_document(filename: str, data: bytes) -> IngestResult:
    """把一份上传文件灌进知识库。"""
    started = time.perf_counter()
    settings = get_settings()

    # ── 校验 ──
    if not data:
        raise BizError(ResultCode.FILE_EMPTY, "上传的文件是空的")
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // 1024 // 1024
        raise BizError(ResultCode.PARAM_INVALID, f"文件不能超过 {limit_mb} MB")

    doc_name = filename or "unnamed"
    content_hash = sha256_hex(data)

    # ── 解析 + 切分：放在任何写库动作之前，失败就什么都不留下 ──
    text = parse_document(doc_name, data)
    pieces = split_text(text)
    if not pieces:
        raise BizError(ResultCode.FILE_EMPTY, "文档切分后没有有效内容")

    # ── 事务1：建记录 + 存切片 ──
    async with session_scope() as session:
        existing = await repo.find_document_by_hash(session, content_hash)
        if existing is not None and existing.status == DocStatus.DONE:
            # 秒传：同样内容的文件不再重复向量化（省一次 embedding 开销）
            return IngestResult(
                doc_id=existing.id,
                doc_name=existing.doc_name,
                chunk_count=existing.chunk_count,
                cost_ms=_elapsed_ms(started),
                duplicated=True,
            )
        if existing is not None:
            # 上次失败（或中断在 PENDING）留下的残影 —— 内容指纹被它占着，
            # 不清掉的话这次插入会撞唯一约束
            await repo.delete_document(session, existing)
            await session.flush()

        doc = await repo.create_document(
            session,
            doc_name=doc_name,
            source_type=source_type_of(doc_name),
            content_hash=content_hash,
            file_size=len(data),
        )
        chunks = await repo.add_chunks(session, doc.id, pieces)
        doc_id = doc.id
        # (chunk_id, vector_id, chunk_index) —— 出事务前把要用的值取出来
        chunk_meta = [(c.id, c.vector_id, c.chunk_index) for c in chunks]

    # ── 事务外：向量化 + 写 Qdrant ──
    try:
        vectors = await embed_texts(pieces)
        points = [
            VectorPoint(
                vector_id=vector_id,
                vector=vector,
                doc_id=doc_id,
                doc_name=doc_name,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                content=content,
            )
            for (chunk_id, vector_id, chunk_index), vector, content in zip(
                chunk_meta, vectors, pieces, strict=True
            )
        ]
        await upsert_points(points)
    except Exception as exc:
        async with session_scope() as session:
            await repo.mark_status(session, doc_id, DocStatus.FAILED, error_msg=str(exc)[:500])
        raise

    # ── 事务2：标记完成 ──
    async with session_scope() as session:
        await repo.mark_status(
            session, doc_id, DocStatus.DONE, chunk_count=len(pieces), error_msg=None
        )

    return IngestResult(
        doc_id=doc_id,
        doc_name=doc_name,
        chunk_count=len(pieces),
        cost_ms=_elapsed_ms(started),
        duplicated=False,
    )


async def list_documents() -> list[DocumentOut]:
    async with session_scope() as session:
        docs = await repo.list_documents(session)
        return [_to_out(d) for d in docs]


async def delete_document(doc_id: int) -> None:
    """删除文档。

    顺序与 Java 一致：**先删向量，再删数据库**。
    反过来的话，一旦删库成功而删向量失败，就会留下检索得到、详情却查不到的孤儿向量。
    """
    async with session_scope() as session:
        doc = await repo.get_document(session, doc_id)
        if doc is None:
            raise BizError(ResultCode.DOC_NOT_FOUND, f"文档不存在：{doc_id}")

    await delete_by_doc(doc_id)

    async with session_scope() as session:
        doc = await repo.get_document(session, doc_id)
        if doc is not None:
            await repo.delete_document(session, doc)
