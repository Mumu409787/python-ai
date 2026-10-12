"""向量化。

用硅基流动的 OpenAI 兼容 `/embeddings`，原生 SDK 调用。

⚠️ 两个必须守住的点（Java 侧实测踩过）：
1. **绝不传 `dimensions`** —— bge-m3 维度固定 1024，传了返回 HTTP 400 / code 20015
2. **不要信模型自报的维度** —— 非 OpenAI 官方模型会给出错误的默认值（1536），
   要真实调一次拿 `len(embedding)`
"""

from __future__ import annotations

import math

from .clients import embedding_client
from .config import get_settings
from .errors import BizError, ResultCode


async def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """批量向量化，按 `batch_size` 分批串行调用。

    返回顺序与入参严格一一对应（按响应里的 `index` 归位，不依赖返回顺序）。
    """
    if not texts:
        return []

    s = get_settings()
    size = batch_size if batch_size is not None else s.embedding_batch_size
    if size <= 0:
        raise BizError(ResultCode.PARAM_INVALID, "向量批大小必须大于 0")
    client = embedding_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        try:
            # 刻意不传 dimensions
            resp = await client.embeddings.create(model=s.embedding_model, input=batch)
        except Exception as exc:
            raise BizError(
                ResultCode.EMBEDDING_FAILED,
                f"向量化失败（第 {start // size + 1} 批，共 {len(batch)} 条）：{exc}",
                retryable=True,
            ) from exc

        ordered = sorted(resp.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(batch))):
            raise BizError(ResultCode.EMBEDDING_FAILED, "向量响应数量或索引不完整", retryable=True)
        expected_dim = len(vectors[0]) if vectors else len(ordered[0].embedding)
        if not expected_dim or any(
            len(item.embedding) != expected_dim
            or not all(math.isfinite(value) for value in item.embedding)
            for item in ordered
        ):
            raise BizError(ResultCode.EMBEDDING_FAILED, "向量响应维度不一致或包含无效数值")
        vectors.extend(item.embedding for item in ordered)

    return vectors


async def embed_query(text: str) -> list[float]:
    """单条向量化（检索时用）。"""
    vectors = await embed_texts([text])
    return vectors[0]


async def probe_dimension() -> int:
    """真实探测一次维度 —— 建 Qdrant 集合必须用它，不能用配置里的假设值。"""
    vectors = await embed_texts(["dimension probe"])
    return len(vectors[0])
