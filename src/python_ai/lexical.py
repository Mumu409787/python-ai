"""小型知识库的 BM25 召回：中文双字词项，保留错误码及接口路径。"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import replace

from rank_bm25 import BM25Plus

from .db.session import session_scope
from .observability import traced
from .repository import done_document_ids, searchable_chunks
from .vectorstore import SearchHit

_TERMS = re.compile(r"/?[a-z0-9_]+(?:[./:\-][a-z0-9_]+)*|[\u3400-\u9fff]+")
_HAN = re.compile(r"^[\u3400-\u9fff]+$")
_STOP = frozenset({"什么", "多少", "怎么", "如何", "是否", "这个", "那个", "请问", "一下"})


def tokenize(text: str) -> list[str]:
    """文档与问题共用确定性规则，无需下载词典；中文用相邻双字片段。"""
    tokens = []
    for term in _TERMS.findall(unicodedata.normalize("NFKC", text).casefold()):
        if _HAN.fullmatch(term):
            parts = [term] if len(term) == 1 else [term[i : i + 2] for i in range(len(term) - 1)]
            tokens.extend(part for part in parts if part not in _STOP)
        else:
            tokens.append(term)
    return tokens


def rank_chunks(query: str, chunks: list[SearchHit], top_k: int) -> list[SearchHit]:
    """CPU 部分可在线程中执行。delta=0 使无匹配分数为零，单片段也可检索。"""
    terms = list(dict.fromkeys(tokenize(query)))
    corpus = [(hit, tokenize(hit.content)) for hit in chunks]
    corpus = [(hit, tokens) for hit, tokens in corpus if tokens]
    if not terms or not corpus:
        return []
    # BM25Plus 的 IDF 保持正数，避免小语料中 Okapi 的零/负 IDF 漏掉真实匹配。
    scores = BM25Plus([tokens for _, tokens in corpus], delta=0).get_scores(terms)
    ranked = sorted(
        (
            replace(hit, score=float(score), score_kind="bm25")
            for (hit, _), score in zip(corpus, scores, strict=True)
            if score > 0
        ),
        key=lambda hit: (-hit.score, hit.chunk_id or 0),
    )
    return ranked[:top_k]


@traced("bm25_recall")
async def bm25_recall(query: str, *, top_k: int) -> list[SearchHit]:
    # 每次读取当前 DONE 片段，HTTP/MCP 多进程都可立即看到已提交的新增和删除。
    async with session_scope() as session:
        rows = await searchable_chunks(session)
    chunks = [
        SearchHit(0.0, *row[:5], score_kind="bm25", source_page=row[5] if len(row) > 5 else None)
        for row in rows
    ]
    hits = await asyncio.to_thread(rank_chunks, query, chunks, top_k)
    # 排序期间可能有删除提交，返回前再按元数据校验一次。
    async with session_scope() as session:
        ready = await done_document_ids(session, {h.doc_id for h in hits if h.doc_id is not None})
    return [hit for hit in hits if hit.doc_id in ready]
