"""Qdrant 封装：建集合、写入、检索、按文档删除。

对应 Java 的 `QdrantCollectionInitializer` + `VectorStoreService`。

与 Java 侧的两点差异：
1. **原文也存进 payload**。Java 只存 doc_id/doc_name/chunk_id/chunk_index，
   拿到命中后还要回 MySQL 查 content；存进 payload 后检索一步到位。
2. **删除直接用 payload 过滤**。Java 要先从 chunk 表查出所有 vector_id 再删；
   Qdrant 原生支持按 payload 条件删除，少一次往返、也不会因为映射表不全而漏删。
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .clients import qdrant
from .config import get_settings
from .embedding import embed_query, probe_dimension
from .errors import BizError, ResultCode

# payload 键名 —— 前四个与 Java 的 ChunkMetadataKeys 保持一致
PAYLOAD_DOC_ID = "doc_id"
PAYLOAD_DOC_NAME = "doc_name"
PAYLOAD_CHUNK_ID = "chunk_id"
PAYLOAD_CHUNK_INDEX = "chunk_index"
PAYLOAD_CONTENT = "content"


@dataclass(slots=True)
class VectorPoint:
    vector_id: str
    vector: list[float]
    doc_id: int
    doc_name: str
    chunk_id: int
    chunk_index: int
    content: str


@dataclass(slots=True)
class SearchHit:
    score: float
    doc_id: int | None
    doc_name: str | None
    chunk_id: int | None
    chunk_index: int | None
    content: str

    @classmethod
    def from_scored_point(cls, point: object) -> SearchHit:
        payload = getattr(point, "payload", None) or {}
        return cls(
            score=float(getattr(point, "score", 0.0)),
            doc_id=payload.get(PAYLOAD_DOC_ID),
            doc_name=payload.get(PAYLOAD_DOC_NAME),
            chunk_id=payload.get(PAYLOAD_CHUNK_ID),
            chunk_index=payload.get(PAYLOAD_CHUNK_INDEX),
            content=payload.get(PAYLOAD_CONTENT, ""),
        )


def _doc_filter(doc_id: int) -> Filter:
    return Filter(must=[FieldCondition(key=PAYLOAD_DOC_ID, match=MatchValue(value=doc_id))])


async def ensure_collection() -> int:
    """启动自检：集合不存在就建，存在就校验维度。返回实测维度。

    维度是**真实调一次 embedding 探出来的**，不是读配置——
    非 OpenAI 官方模型自报的维度是错的（bge-m3 会报 1536，实际 1024），
    拿错的值建集合会导致后续写入全部失败。
    """
    s = get_settings()
    client = qdrant()
    name = s.qdrant_collection

    try:
        dim = await probe_dimension()
    except BizError:
        raise
    except Exception as exc:
        raise BizError(ResultCode.EMBEDDING_FAILED, f"维度探测失败：{exc}") from exc

    try:
        exists = await client.collection_exists(name)
    except Exception as exc:
        raise BizError(
            ResultCode.VECTOR_STORE_UNAVAILABLE,
            f"Qdrant 不可用（{s.qdrant_url}）：{exc}",
        ) from exc

    if not exists:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return dim

    info = await client.get_collection(name)
    existing = info.config.params.vectors.size
    if existing != dim:
        raise BizError(
            ResultCode.VECTOR_STORE_UNAVAILABLE,
            f"集合 {name} 的维度是 {existing}，与向量模型实测维度 {dim} 不一致。"
            f"修复：docker compose down -v && docker compose up -d（会清空已有向量，需重新入库）",
        )
    return dim


async def upsert_points(points: list[VectorPoint]) -> None:
    """批量写入。`vector_id` 由调用方生成，保证同一 chunk 重复入库时是覆盖而非新增。"""
    if not points:
        return

    s = get_settings()
    try:
        await qdrant().upsert(
            collection_name=s.qdrant_collection,
            points=[
                PointStruct(
                    id=p.vector_id,
                    vector=p.vector,
                    payload={
                        PAYLOAD_DOC_ID: p.doc_id,
                        PAYLOAD_DOC_NAME: p.doc_name,
                        PAYLOAD_CHUNK_ID: p.chunk_id,
                        PAYLOAD_CHUNK_INDEX: p.chunk_index,
                        PAYLOAD_CONTENT: p.content,
                    },
                )
                for p in points
            ],
        )
    except Exception as exc:
        raise BizError(
            ResultCode.VECTOR_STORE_WRITE_FAILED,
            f"向量写入失败：{exc}",
            retryable=True,
        ) from exc


async def search(
    query: str,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[SearchHit]:
    """向量检索。低于 `score_threshold` 的命中在 Qdrant 侧就被丢掉，不进 prompt。"""
    s = get_settings()
    vector = await embed_query(query)

    try:
        result = await qdrant().query_points(
            collection_name=s.qdrant_collection,
            query=vector,
            limit=top_k or s.retrieval_top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
    except Exception as exc:
        raise BizError(
            ResultCode.VECTOR_STORE_UNAVAILABLE,
            f"向量检索失败：{exc}",
        ) from exc

    return [SearchHit.from_scored_point(p) for p in result.points]


async def delete_by_doc(doc_id: int) -> None:
    """按 `doc_id` 删除该文档的全部向量。"""
    s = get_settings()
    try:
        await qdrant().delete(
            collection_name=s.qdrant_collection,
            points_selector=FilterSelector(filter=_doc_filter(doc_id)),
        )
    except Exception as exc:
        raise BizError(
            ResultCode.VECTOR_STORE_UNAVAILABLE,
            f"删除向量失败：{exc}",
        ) from exc


async def count_by_doc(doc_id: int) -> int:
    """统计该文档在向量库里的点数 —— 删除后校验「无孤儿向量」用。"""
    s = get_settings()
    result = await qdrant().count(
        collection_name=s.qdrant_collection,
        count_filter=_doc_filter(doc_id),
        exact=True,
    )
    return result.count


async def collection_count() -> int:
    """集合总点数。"""
    result = await qdrant().count(
        collection_name=get_settings().qdrant_collection,
        exact=True,
    )
    return result.count
