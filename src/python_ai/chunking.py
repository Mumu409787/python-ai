"""文本切分与内容指纹。

对应 Java 的 `ChunkingService` + `DocumentSplitters.recursive`。

参数与 Java 侧对齐：**800 / 150 字符**（不是 token）。
唯一的有意差异是分隔符表——Java 只按换行和空格切，中文长句没有空格可切，
会被硬切在句子中间。这里补上中文标点作为分隔层级。
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings

# 从粗到细：先按空行分段，再按换行，然后中文句末标点，最后才硬切。
# 原版默认只有 ["\n\n", "\n", " ", ""]，对中文几乎等于硬切。
SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    "，",
    "、",
    " ",
    "",
]


def sha256_hex(data: bytes) -> str:
    """整份文件的内容指纹，用于秒传去重（对齐 Java 的 SHA-256 去重）。"""
    return hashlib.sha256(data).hexdigest()


def split_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """递归切分，返回非空的片段列表。"""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.chunk_size,
        chunk_overlap=chunk_overlap or s.chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


def estimate_chunk_count(text: str) -> int:
    """切分前的成本预估（对齐 Java `ChunkingService.estimateChunkCount`）。

    只做上限估计，实际片数以 `split_text` 结果为准。
    """
    size = get_settings().chunk_size
    if not text:
        return 0
    return max(1, -(-len(text) // size))  # 向上取整
