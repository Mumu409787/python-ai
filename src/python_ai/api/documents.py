"""文档管理接口。对应 Java 的 `DocumentController`。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from ..config import get_settings
from ..errors import BizError, ResultCode
from ..ingest import delete_document, ingest_document, list_documents
from ..schemas import DocumentOut, IngestResult, Result

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=Result[IngestResult])
async def upload(file: Annotated[UploadFile, File()]) -> Result[IngestResult]:
    """上传并入库。**字段名必须是 `file`**（与 Java 侧一致）。"""
    try:
        data = await file.read(get_settings().max_upload_bytes + 1)
    finally:
        await file.close()
    if not data:
        raise BizError(ResultCode.FILE_EMPTY, "上传的文件是空的")
    result = await ingest_document(file.filename or "unnamed", data)
    return Result.ok(result)


@router.get("", response_model=Result[list[DocumentOut]])
async def list_all() -> Result[list[DocumentOut]]:
    return Result.ok(await list_documents())


@router.delete("/{doc_id}", response_model=Result[None])
async def remove(doc_id: int) -> Result[None]:
    await delete_document(doc_id)
    return Result.ok()
