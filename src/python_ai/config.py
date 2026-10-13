"""配置：从 `.env` 读取，集中所有可调参数。

对应 Java 项目的 `AiProperties` + `RagProperties`，但用平铺字段 + 派生属性代替嵌套绑定，
读起来更直观，`.env` 里也能一眼看全。

约定：环境变量名 = 字段名的大写形式（如 `deepseek_api_key` ← `DEEPSEEK_API_KEY`）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

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
    chat_max_tokens: int = Field(default=4096, gt=0)
    chat_timeout: float = Field(default=60.0, gt=0)
    chat_max_retries: int = Field(default=2, ge=0)

    # ── 向量模型：硅基流动 bge-m3 ──
    # ⚠️ 不传 dimensions —— bge-m3 维度固定 1024，传了会 400 / code 20015
    siliconflow_api_key: str
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_timeout: float = Field(default=30.0, gt=0)

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
    session_ttl_seconds: int = Field(default=3600, gt=0)
    session_max_turns: int = Field(default=10, gt=0)
    memory_backend: str = "langgraph"

    # ── 可观测：Langfuse（可选，不填则不上报）──
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    langfuse_enabled: bool = True
    langfuse_capture_content: bool = False
    classifier_enabled: bool = True
    classifier_model_path: Path = PROJECT_ROOT / "training/artifacts/question_classifier.json"
    ocr_enabled: bool = True
    ocr_language: str = "chi_sim+eng"
    tesseract_path: str = ""
    ocr_tessdata_dir: Path = PROJECT_ROOT / ".tools/tessdata"
    ocr_timeout_seconds: float = Field(default=20, gt=0)
    ocr_total_timeout_seconds: float = Field(default=60, gt=0)
    ocr_max_pages: int = Field(default=30, ge=1)
    ocr_max_pixels: int = Field(default=20_000_000, ge=1)
    ocr_pdf_dpi: int = Field(default=200, ge=72, le=400)

    # ── 切分参数（对齐 Java：800 / 150 字符，不是 token）──
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # ── 检索参数 ──
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    # 0.45 是在本项目样例语料（data/samples/star-link-spec.md）上实测标定的：
    # 不相关问题的最高分是 0.39，相关问题的最低分是 0.52，取中间值。
    # ⚠️ 换语料必须重新标定 —— 这个值随领域、文档风格、切片粒度都会变。
    # （Java 侧固定 0.6，在这份语料上会把「认证用的 Token」这种 0.578 的问题漏掉）
    retrieval_score_threshold: float = Field(default=0.45, ge=0, le=1)
    retrieval_max_segment_chars: int = Field(default=1200, gt=0)

    # 本地 BM25 与向量按排名融合；小型知识库每次从 DONE 片段重建，避免陈旧缓存。
    hybrid_enabled: bool = True
    bm25_candidate_k: int = Field(default=20, ge=1, le=100)
    hybrid_rrf_k: int = Field(default=60, ge=1)

    # 只改写有历史的 RAG 追问；失败、超时、无效输出均使用原问题。
    query_rewrite_enabled: bool = True
    query_rewrite_timeout: float = Field(default=8.0, gt=0)
    query_rewrite_history_turns: int = Field(default=3, ge=1, le=10)

    # ── Rerank（Java 侧没有，本项目新增）──
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 向量阶段捞候选 → rerank 精排 → 取 top_k
    rerank_candidate_k: int = Field(default=20, ge=1, le=100)
    # 与 retrieval_score_threshold 同源：过滤职责全部交给向量阶段
    rerank_candidate_threshold: float = Field(default=0.45, ge=0, le=1)
    # rerank 后按「相对比例」截断：只保留分数 >= top1 * cutoff 的命中。
    # 用相对值而非绝对值，是因为 relevance_score 的量纲不稳定 ——
    # 实测同一语料、同一批候选能落在 0.0 ~ 0.96，绝对阈值非常脆弱：
    # 初版设 0.2，正好把 0.195 的合法命中（「认证用的 Token」）挡成了拒答。
    # 相对截断则能同时处理高分问题（裁掉 0.05 的拖尾）和低分问题（保留全部有效命中）。
    rerank_relative_cutoff: float = Field(default=0.1, ge=0, le=1)
    # 拒答的最终兜底不在这里，而是 RAG_SYSTEM prompt —— 与 Java 侧设计一致。

    # ── 上传限制（对齐 Java：20MB）──
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)

    # Trusted server-to-server QA; an empty key disables only the external ask endpoint.
    integration_api_key: SecretStr = SecretStr("")
    integration_timeout_seconds: float = Field(default=180.0, gt=0, le=600)

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        integration_key = self.integration_api_key.get_secret_value()
        if integration_key and (
            len(integration_key) < 32 or integration_key != integration_key.strip()
        ):
            raise ValueError("integration_api_key 至少 32 字符，且不能含首尾空白")
        if self.memory_backend not in {"langgraph", "redis"}:
            raise ValueError("memory_backend 必须是 langgraph 或 redis")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy async 用的 DSN。"""
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置。用函数而非模块级实例，测试时才好替换。"""
    return Settings()
