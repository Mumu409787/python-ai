"""环境自检：验证 .env 里的密钥与三个外部依赖是否真的能用。

    uv run python scripts/check_env.py

不需要启动 FastAPI，也不动数据库 —— 只想快速确认「配置对不对」时跑这个。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 让脚本无论从哪个目录调用都能读到项目根的 .env
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OK = "  [ok]"
BAD = "  [!!]"


async def check_chat() -> bool:
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
    )
    try:
        # max_tokens 不能太小：deepseek-flash 是推理模型，会先输出 reasoning_content
        # 再输出正文，给 16 会被思维链吃光、正文为空（finish_reason='length'）
        resp = await client.chat.completions.create(
            model=os.environ["DEEPSEEK_MODEL"],
            messages=[{"role": "user", "content": "回复两个字：正常"}],
            max_tokens=128,
            temperature=0,
        )
    except Exception as exc:
        print(f"{BAD} 对话模型调用失败：{type(exc).__name__}: {exc}")
        return False
    print(f"{OK} 对话模型 {os.environ['DEEPSEEK_MODEL']} → {resp.choices[0].message.content!r}")
    return True


async def check_embedding() -> bool:
    client = AsyncOpenAI(
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url=os.environ["SILICONFLOW_BASE_URL"],
    )
    try:
        # 刻意不传 dimensions —— bge-m3 传了会 400 / code 20015
        resp = await client.embeddings.create(
            model=os.environ["EMBEDDING_MODEL"],
            input=["维度探测"],
        )
    except Exception as exc:
        print(f"{BAD} 向量模型调用失败：{type(exc).__name__}: {exc}")
        return False

    dim = len(resp.data[0].embedding)
    expect = int(os.environ.get("EMBEDDING_DIM", "1024"))
    mark = OK if dim == expect else BAD
    print(f"{mark} 向量模型 {os.environ['EMBEDDING_MODEL']} → {dim} 维（.env 声明 {expect}）")
    return dim == expect


async def check_rerank() -> bool:
    """探测硅基流动 rerank 接口的真实请求/响应字段名（字段名不要凭记忆写）。"""
    url = f"{os.environ['SILICONFLOW_BASE_URL'].rstrip('/')}/rerank"
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "向量库用的是什么",
        "documents": [
            "本系统使用 Qdrant 作为向量数据库，距离度量是余弦相似度。",
            "今天天气不错，适合出门散步。",
        ],
        "top_n": 2,
        "return_documents": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                url,
                headers={"Authorization": f"Bearer {os.environ['SILICONFLOW_API_KEY']}"},
                json=payload,
            )
    except Exception as exc:
        print(f"{BAD} rerank 请求异常：{type(exc).__name__}: {exc}")
        return False

    if resp.status_code != 200:
        print(f"{BAD} rerank HTTP {resp.status_code}：{resp.text[:300]}")
        return False

    data = resp.json()
    print(f"{OK} rerank 原始响应键：{sorted(data.keys())}")
    results = data.get("results")
    if not results:
        print(f"{BAD} rerank 响应里没有 results：{str(data)[:300]}")
        return False
    print(f"{OK} rerank 单条结果字段：{sorted(results[0].keys())}")
    print(f"      → {results}")
    return True


async def check_qdrant() -> bool:
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            resp = await http.get(f"{os.environ['QDRANT_URL']}/collections")
        except Exception as exc:
            print(f"{BAD} Qdrant 连不上：{exc}")
            return False
    print(f"{OK} Qdrant {os.environ['QDRANT_URL']} → {resp.json()}")
    return True


async def main() -> int:
    print("── 外部依赖连通性 ──")
    results = [
        await check_chat(),
        await check_embedding(),
        await check_rerank(),
        await check_qdrant(),
    ]
    print()
    if all(results):
        print("全部通过 ✓")
        return 0
    print("有项目未通过，见上 ✗")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
