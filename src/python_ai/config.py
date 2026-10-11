"""配置：从 `.env` 读取，集中所有可调参数。

对应 Java 项目的 `AiProperties` + `RagProperties`，但用平铺字段 + 派生属性代替嵌套绑定，
读起来更直观，`.env` 里也能一眼看全。

约定：环境变量名 = 字段名的大写形式（如 `deepseek_api_key` ← `DEEPSEEK_API_KEY`）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/python_ai/config.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 对话模型：DeepSeek ──
    # ⚠️ deepseek-flash 是推理模型：先吐 reasoning_content 再吐正文，
    #    两者共享 max_tokens 额度，所以这个值给得比 Java 侧宽。
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    chat_temperature: float = 0.3
    chat_max_tokens: int = 4096
    chat_timeout: float = 60.0
    chat_max_retries: int = 2

    # ── 向量模型：硅基流动 bge-m3 ──
    # ⚠️ 不传 dimensions —— bge-m3 维度固定 1024，传了会 400 / code 20015
    siliconflow_api_key: str
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_timeout: float = 30.0

    # ── 向量库：Qdrant ──
    qdrant_url: str = "http://localhost:6335"
    qdrant_collection: str = "python_ai_docs"

    # ── 关系库：PostgreSQL（元数据）──
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres123456"
    postgres_db: str = "python_ai"

    # ── 缓存：Redis（会话记忆）──
    redis_url: str = "redis://localhost:6380/0"
    session_ttl_seconds: int = 3600
    session_max_turns: int = 10

    # ── 可观测：Langfuse（可选，不填则不上报）──
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # ── 切分参数（对齐 Java：800 / 150 字符，不是 token）──
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── 检索参数 ──
    retrieval_top_k: int = 5
    # 0.45 是在本项目样例语料（data/samples/star-link-spec.md）上实测标定的：
    # 不相关问题的最高分是 0.39，相关问题的最低分是 0.52，取中间值。
    # ⚠️ 换语料必须重新标定 —— 这个值随领域、文档风格、切片粒度都会变。
    # （Java 侧固定 0.6，在这份语料上会把「认证用的 Token」这种 0.578 的问题漏掉）
    retrieval_score_threshold: float = 0.45
    retrieval_max_segment_chars: int = 1200

    # ── Rerank（Java 侧没有，本项目新增）──
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 向量阶段捞候选 → rerank 精排 → 取 top_k
    rerank_candidate_k: int = 20
    # 与 retrieval_score_threshold 同源：过滤职责全部交给向量阶段
    rerank_candidate_threshold: float = 0.45
    # rerank 后按「相对比例」截断：只保留分数 >= top1 * cutoff 的命中。
    # 用相对值而非绝对值，是因为 relevance_score 的量纲不稳定 ——
    # 实测同一语料、同一批候选能落在 0.0 ~ 0.96，绝对阈值非常脆弱：
    # 初版设 0.2，正好把 0.195 的合法命中（「认证用的 Token」）挡成了拒答。
    # 相对截断则能同时处理高分问题（裁掉 0.05 的拖尾）和低分问题（保留全部有效命中）。
    rerank_relative_cutoff: float = 0.1
    # 拒答的最终兜底不在这里，而是 RAG_SYSTEM prompt —— 与 Java 侧设计一致。

    # ── 上传限制（对齐 Java：20MB）──
    max_upload_bytes: int = 20 * 1024 * 1024

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy async 用的 DSN。"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置。用函数而非模块级实例，测试时才好替换。"""
    return Settings()
