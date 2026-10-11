r"""Python + AI 复刻项目：RAG 检索问答 + Rerank + 评测。

接口与 `D:\Projects\java-ai-rag` 对齐，便于对照学习。

    uv run python-ai          # 启动服务（等同 uvicorn python_ai.main:app）

包根**刻意不导入子模块** —— 让 `import python_ai.schemas` 这类轻量导入
不必连带加载 FastAPI / SQLAlchemy 全套，单元测试跑得更快。
"""

__all__: list[str] = []
