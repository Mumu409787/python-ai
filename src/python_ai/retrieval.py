"""检索层：向量召回 →（可选的）Rerank → 组装上下文。

对应 Java 的 `RetrievalService`。

**拒答的机制在这里**：`score_threshold` 由 Qdrant 在检索时生效，
低分命中在向量库侧就被丢掉，根本进不了 prompt。这不是「召回了再判断要不要答」，
而是「不够相关就当作没检索到」。真正的兜底是 prompt 里那句固定文案。
"""

from __future__ import annotations

import logging

import httpx

from .config import get_settings
from .errors import BizError, ResultCode
from .schemas import RetrievalHit
from .vectorstore import SearchHit, search

logger = logging.getLogger(__name__)


def truncate(text: str, limit: int) -> str:
    """单段注入上限，超出截断并加省略号（对齐 Java 的 maxSegmentChars）。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def vector_recall(
    query: str,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[SearchHit]:
    """纯向量召回。"""
    s = get_settings()
    return await search(
        query,
        top_k=top_k or s.retrieval_top_k,
        score_threshold=threshold if threshold is not None else s.retrieval_score_threshold,
    )


async def retrieve(question: str, *, top_k: int | None = None) -> list[SearchHit]:
    """对外入口：向量召回，必要时接 Rerank。"""
    s = get_settings()

    if not s.rerank_enabled:
        return await vector_recall(question, top_k=top_k)

    # 开了 rerank 就用宽松阈值多捞一些候选，再精排到 top_k
    candidates = await vector_recall(
        question,
        top_k=s.rerank_candidate_k,
        threshold=s.rerank_candidate_threshold,
    )
    if not candidates:
        return []

    try:
        ranked = await _rerank(question, candidates)
    except Exception as exc:
        # rerank 是增强项，挂了不该拖垮整个问答
        logger.warning("rerank 失败，回落到纯向量结果：%s", exc)
        return candidates[: top_k or s.retrieval_top_k]

    return _cut_tail(ranked, s.rerank_relative_cutoff)[: top_k or s.retrieval_top_k]


def _cut_tail(ranked: list[SearchHit], ratio: float) -> list[SearchHit]:
    """按 top1 的相对比例裁掉拖尾。

    rerank 常给出「一条高分 + 一串接近 0 的噪声」这种分布，
    噪声进了 prompt 既费 token 又可能干扰作答，所以要裁掉。
    但绝对阈值不可用（见 config 里的说明），只能相对着 top1 裁。
    """
    if not ranked:
        return []
    cutoff = ranked[0].score * ratio
    return [hit for hit in ranked if hit.score >= cutoff]


async def _rerank(question: str, candidates: list[SearchHit]) -> list[SearchHit]:
    """调硅基流动的 rerank 接口重排。

    实测的请求/响应契约（2026-10-11 验证，勿凭记忆改）：
        请求  {"model", "query", "documents": [...], "top_n", "return_documents"}
        响应  {"id", "meta", "results": [{"index", "document", "relevance_score"}]}
    `index` 指向请求里 `documents` 的下标。
    """
    s = get_settings()
    url = f"{s.siliconflow_base_url.rstrip('/')}/rerank"

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            url,
            headers={"Authorization": f"Bearer {s.siliconflow_api_key}"},
            json={
                "model": s.rerank_model,
                "query": question,
                "documents": [hit.content for hit in candidates],
                "top_n": len(candidates),
                "return_documents": False,
            },
        )

    if resp.status_code != 200:
        raise BizError(
            ResultCode.EMBEDDING_FAILED,
            f"rerank 返回 HTTP {resp.status_code}：{resp.text[:200]}",
            retryable=True,
        )

    results = resp.json().get("results") or []
    ranked: list[SearchHit] = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            continue
        hit = candidates[index]
        ranked.append(
            SearchHit(
                score=float(item.get("relevance_score", 0.0)),
                doc_id=hit.doc_id,
                doc_name=hit.doc_name,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                content=hit.content,
            )
        )

    ranked.sort(key=lambda h: h.score, reverse=True)
    return ranked


def build_contexts(hits: list[SearchHit]) -> list[str]:
    """转成注入 prompt 的文本块（带单段长度上限）。"""
    limit = get_settings().retrieval_max_segment_chars
    return [truncate(hit.content, limit) for hit in hits]


def to_dtos(hits: list[SearchHit]) -> list[RetrievalHit]:
    """转成对外 DTO。"""
    return [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            doc_name=hit.doc_name,
            content=hit.content,
            score=round(hit.score, 4),
        )
        for hit in hits
    ]
