"""把硅基流动的 bge-m3 包装成 Ragas 认得的 Embeddings 接口。

**为什么用同步 httpx 而不是复用 `python_ai.embedding`**：
Ragas 内部可能在工作线程里调用 embedding，而那些线程未必有 event loop，
`asyncio.run()` 在那里会直接抛 `RuntimeError`。同步调用最稳。

同样刻意不传 `dimensions`（bge-m3 传了会 400）。
"""

from __future__ import annotations

import httpx
from langchain_core.embeddings import Embeddings

from python_ai.config import get_settings


class SiliconFlowEmbeddings(Embeddings):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._client = httpx.Client(
            base_url=settings.siliconflow_base_url,
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
            timeout=60.0,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.post(
                "/embeddings",
                json={"model": self._model, "input": batch},  # 不传 dimensions
            )
            resp.raise_for_status()
            ordered = sorted(resp.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
