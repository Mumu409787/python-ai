"""DTO 与统一响应体。

对外字段一律 camelCase —— 与 Java 项目逐字一致，
这样同一个前端可以不加改动地同时对接两边（`sessionId` / `costMs` / `docId`…）。
内部代码仍然写 snake_case，靠 alias generator 转换。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,  # 允许用 snake_case 构造，方便内部调用
        from_attributes=True,
    )


class Result[T](CamelModel):
    """统一响应体，对应 Java 的 `Result<T>(code, message, data)`。

    `data` 为 None 时由 FastAPI 的 `response_model_exclude_none=True` 省略。
    """

    code: int = 0
    message: str = "成功"
    data: T | None = None

    @classmethod
    def ok(cls, data: T | None = None) -> Result[T]:
        return cls(code=0, message="成功", data=data)

    @classmethod
    def fail(cls, code: int, message: str) -> Result[T]:
        return cls(code=int(code), message=message, data=None)


class ChatMode(StrEnum):
    CHAT = "CHAT"
    RAG = "RAG"

    @classmethod
    def parse(cls, raw: str | None) -> ChatMode:
        """空值默认 RAG；非法值抛错——与 Java `ChatService.Mode.from()` 一致。"""
        if raw is None or not raw.strip():
            return cls.RAG
        try:
            return cls(raw.strip().upper())
        except ValueError:
            raise ValueError(f"未知的对话模式: {raw}") from None


class ChatRequest(CamelModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    use_rag: bool | None = None


class RetrievalHit(CamelModel):
    """一条检索命中。流式场景下 `score` 为 None（对齐 Java 的行为）。"""

    chunk_id: int | None = None
    doc_id: int | None = None
    doc_name: str | None = None
    content: str = ""
    score: float | None = None


class ChatResponse(CamelModel):
    session_id: str
    answer: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    cost_ms: int = 0
    used_rag: bool = True


class IngestResult(CamelModel):
    doc_id: int
    doc_name: str
    chunk_count: int
    cost_ms: int
    duplicated: bool = False


class DocumentOut(CamelModel):
    id: int
    doc_name: str
    source_type: str | None = None
    content_hash: str | None = None
    file_size: int | None = None
    chunk_count: int = 0
    status: str = "DONE"
    error_msg: str | None = None
    created_at: datetime | None = None


class DonePayload(CamelModel):
    """SSE `done` 事件的数据体。"""

    session_id: str
    cost_ms: int
