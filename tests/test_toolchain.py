"""工具链自检：不需要任何 API Key 就能跑通，用来确认环境完好。

uv run pytest
"""

import importlib.metadata as md

import pytest

EXPECTED = [
    "langchain",
    "langgraph",
    "langchain-openai",
    "openai",
    "fastapi",
    "uvicorn",
    "qdrant-client",
    "chromadb",
    "tiktoken",
    "httpx",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "langfuse",
    "ragas",
]


@pytest.mark.parametrize("pkg", EXPECTED)
def test_package_installed(pkg: str) -> None:
    assert md.version(pkg)


def test_openai_sdk_is_importable() -> None:
    import openai

    # DeepSeek / 硅基流动 都走这个 client，只是换了 base_url
    assert openai.OpenAI
    assert openai.AsyncOpenAI


def test_python_version_matches_project() -> None:
    import sys

    assert sys.version_info[:2] == (3, 14)


def test_chromadb_local_client_starts(tmp_path) -> None:
    """本地向量库能起来 —— 不用起 Docker 也能练手。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("smoke")
    col.add(ids=["1"], documents=["你好，世界"], embeddings=[[0.0] * 8])
    assert col.count() == 1

    client.delete_collection("smoke")
