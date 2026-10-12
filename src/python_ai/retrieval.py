"""检索层：向量/BM25 召回 → RRF 融合 →（可选的）Rerank → 上下文。

对应 Java 的 `RetrievalService`。

向量阈值由 Qdrant 过滤；BM25 仅返回有词项匹配的片段，不套用余弦阈值。
所有来源均为空时由 rag.py 固定拒答；有来源仍要求模型依据原文回答。
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace

import httpx

from .config import get_settings
from .errors import BizError, ResultCode
from .lexical import bm25_recall
from .observability import traced
from .schemas import RetrievalHit
from .vectorstore import SearchHit, search

logger = logging.getLogger(__name__)


def truncate(text: str, limit: int) -> str:
    """单段注入上限，超出截断并加省略号（对齐 Java 的 maxSegmentChars）。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


@traced("vector_recall")
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


@traced("retrieve")
async def retrieve(question: str, *, top_k: int | None = None) -> list[SearchHit]:
    """对外入口：可选 BM25/RRF 混合召回，必要时接 Rerank。"""
    s = get_settings()
    if top_k is not None and not 1 <= top_k <= 100:
        raise BizError(ResultCode.PARAM_INVALID, "top_k 必须在 1 到 100 之间")

    limit = top_k or s.retrieval_top_k
    candidate_k = max(limit, s.rerank_candidate_k if s.rerank_enabled else limit)
    if s.hybrid_enabled:
        candidate_k = max(candidate_k, s.bm25_candidate_k)
    threshold = s.rerank_candidate_threshold if s.rerank_enabled else s.retrieval_score_threshold
    if s.hybrid_enabled:
        dense, lexical = await asyncio.gather(
            vector_recall(question, top_k=candidate_k, threshold=threshold),
            bm25_recall(question, top_k=candidate_k),
            return_exceptions=True,
        )
        if isinstance(dense, BaseException):
            raise dense
        if isinstance(lexical, BaseException):
            if not isinstance(lexical, Exception):
                raise lexical
            logger.warning("BM25 失败，回退向量检索：%s", type(lexical).__name__)
            candidates = dense
        else:
            candidates = fuse_rrf(dense, lexical, k=s.hybrid_rrf_k) if lexical else dense
    else:
        candidates = await vector_recall(question, top_k=candidate_k, threshold=threshold)
    candidates = candidates[:candidate_k]
    if not candidates:
        return []
    if not s.rerank_enabled:
        return candidates[:limit]

    try:
        ranked = await _rerank(question, candidates)
    except Exception as exc:
        # rerank 是增强项，挂了不该拖垮整个问答
        logger.warning("rerank 失败，回落到纯向量结果：%s", exc)
        # RRF/BM25 不是余弦分数，不套用向量阈值，否则会删掉关键词补召回。
        return [
            h
            for h in candidates
            if h.score_kind != "vector" or h.score >= s.retrieval_score_threshold
        ][:limit]

    return _cut_tail(ranked, s.rerank_relative_cutoff)[: top_k or s.retrieval_top_k]


def fuse_rrf(*rankings: list[SearchHit], k: int = 60) -> list[SearchHit]:
    """每个结果列表对一个片段最多贡献一次；按排名融合，不相加原始分数。"""
    scores: dict[tuple, float] = {}
    hits: dict[tuple, SearchHit] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            key = (
                (hit.doc_id, hit.chunk_id)
                if hit.chunk_id is not None
                else (hit.doc_id, hit.chunk_index, hit.content)
            )
            if key in seen:
                continue
            seen.add(key)
            hits.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [
        replace(hits[key], score=score, score_kind="rrf")
        for key, score in sorted(scores.items(), key=lambda item: -item[1])
    ]


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


@traced("_rerank")
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
    if not isinstance(results, list) or len(results) != len(candidates):
        raise ValueError("rerank 响应数量不完整")
    ranked: list[SearchHit] = []
    seen: set[int] = set()
    for item in results:
        index = item.get("index")
        if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
            raise ValueError("rerank 响应索引无效或重复")
        score = float(item["relevance_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("rerank 响应分数无效")
        seen.add(index)
        hit = candidates[index]
        ranked.append(
            SearchHit(
                score=score,
                doc_id=hit.doc_id,
                doc_name=hit.doc_name,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                source_page=hit.source_page,
                content=hit.content,
                score_kind="rerank",
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
            score_kind=hit.score_kind,
            source_page=hit.source_page,
        )
        for hit in hits
    ]
