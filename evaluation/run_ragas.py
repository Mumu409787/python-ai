"""用 Ragas 评测本项目的 RAG 链路。

    uv run python -m evaluation.run_ragas

产出 `evaluation/report.md`，含四个指标的总分与逐题明细。
**这一步会真的调用大量 LLM**（每个指标每题至少一次），耗时几分钟、也会花钱。

⚠️ **导入顺序有硬约束，不要把这些 import 挪到文件顶部。**

`ragas/executor.py` 在**模块导入时**就执行 `nest_asyncio.apply()`，而 nest_asyncio 1.6.0
会 patch asyncio 的全局实现，与 Python 3.14 不兼容 —— patch 之后 `asyncio.timeout()`
拿不到当前 task，抛 `RuntimeError: Timeout should be used inside a task`：
asyncpg 建不了连接，ragas 自己的评分任务也会 **全部返回 nan**（表面看是跑完了的）。

`main()` 里因此按严格顺序来：
    跑完异步的样本收集 → `neutralize_nest_asyncio()` → 才 import ragas

"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from python_ai.clients import chat_model, close_clients
from python_ai.db.session import dispose_db, init_db
from python_ai.ingest import ingest_document
from python_ai.rag import answer
from python_ai.vectorstore import ensure_collection

from .embeddings import SiliconFlowEmbeddings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET_PATH = HERE / "dataset.jsonl"
REPORT_PATH = HERE / "report.md"
RAW_SCORES_PATH = HERE / "raw_scores.csv"
SAMPLE_DOC = ROOT / "data" / "samples" / "star-link-spec.md"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_MEANINGS = {
    "faithfulness": "回答是否完全基于检索到的资料（有无幻觉）",
    "answer_relevancy": "回答与问题的相关程度",
    "context_precision": "检索到的片段里有多少是真正有用的",
    "context_recall": "标准答案需要的信息，有多少被检索到",
}


def load_cases() -> list[dict[str, str]]:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def neutralize_nest_asyncio() -> None:
    """把 `nest_asyncio.apply()` 变成空操作 —— 必须在 import ragas 之前调用。

    `ragas/executor.py` 在**模块导入时**就执行了 `nest_asyncio.apply()`（第 16 行），
    而 nest_asyncio 1.6.0 会去 patch asyncio 的全局实现，它与 Python 3.14 不兼容：
    patch 之后 `asyncio.timeout()` 不再能拿到当前 task，表现为
    `RuntimeError: Timeout should be used inside a task` ——
    asyncpg 建连接会挂，ragas 自己的 72 个评分任务也会全部返回 nan。

    nest_asyncio 的用途是「在已经在跑的 event loop 里再嵌套跑一个」，
    我们这里是普通脚本、直接 `asyncio.run()`，根本不需要它，所以直接屏蔽掉。
    """
    import nest_asyncio

    nest_asyncio.apply = lambda *args, **kwargs: None  # type: ignore[method-assign]


async def _prepare() -> list[dict[str, object]]:
    """入库语料 + 逐题跑 RAG，收集 (问题, 回答, 命中上下文, 标准答案)。

    返回纯 dict —— 刻意不构造 `SingleTaskSample`，因为那需要先 import ragas。
    """
    await init_db()
    await ensure_collection()

    ingested = await ingest_document(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())
    print(
        f"语料：{SAMPLE_DOC.name} → {ingested.chunk_count} 片"
        f"{'（命中已有记录，跳过重复向量化）' if ingested.duplicated else ''}"
    )

    cases = load_cases()
    collected: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        response = await answer(case["question"])
        contexts = [hit.content for hit in response.hits]
        collected.append(
            {
                "user_input": case["question"],
                "retrieved_contexts": contexts,
                "response": response.answer,
                "reference": case["ground_truth"],
            }
        )
        print(f"  [{index:2d}/{len(cases)}] 命中 {len(contexts)} 条 | {case['question'][:34]}")

    await dispose_db()
    await close_clients()
    return collected


def write_report(
    scores: dict[str, float],
    counts: dict[str, int],
    rows: list[dict],
    elapsed: float,
    cases: int,
) -> None:
    lines = [
        "# RAG 评测报告（Ragas）",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 语料：`data/samples/{SAMPLE_DOC.name}`",
        f"- 评测集：`evaluation/dataset.jsonl`（{cases} 题）",
        "- 链路：向量召回（阈值过滤）→ Rerank 精排 → DeepSeek 生成",
        f"- 总耗时：{elapsed:.0f} 秒",
        "",
        "## 总分",
        "",
        "| 指标 | 分数 | 有效样本 | 含义 |",
        "|---|---|---|---|",
    ]
    for name in METRIC_NAMES:
        value = scores.get(name, float("nan"))
        valid = counts.get(name, 0)
        lines.append(f"| {name} | **{value:.4f}** | {valid}/{cases} | {METRIC_MEANINGS[name]} |")

    lines += ["", "## 逐题明细", ""]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for row in rows:
            cells = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(str(value).replace("|", "\\|")[:110])
            lines.append("| " + " | ".join(cells) + " |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()

    print("── 1/3 入库语料并跑 RAG ──")
    collected = asyncio.run(_prepare())

    # ⚠️ 顺序要紧：先屏蔽 nest_asyncio，再 import ragas。
    # 两者都必须在上面那段异步数据库操作之后，原因见模块 docstring。
    neutralize_nest_asyncio()

    print("\n── 2/3 导入 ragas 并评分（调用 LLM 较多，请耐心）──")
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    dataset = EvaluationDataset(samples=[SingleTurnSample(**sample) for sample in collected])
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=LangchainLLMWrapper(chat_model()),
        embeddings=LangchainEmbeddingsWrapper(SiliconFlowEmbeddings()),
        show_progress=True,
    )

    print("\n── 3/3 写报告 ──")
    frame = result.to_pandas()
    # 直接存原始分数：报告格式调整、或想换阈值重算时，不必再烧一遍 LLM
    frame.to_csv(RAW_SCORES_PATH, index=False)

    # 总分直接用 pandas 聚合，不走 ragas 的 `result[name]`（那个在这里只会拿到 nan）。
    # `.mean()` 默认跳过 NaN，所以个别评分失败不会毁掉整列。
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in METRIC_NAMES:
        if name in frame.columns:
            column = frame[name]
            counts[name] = int(column.notna().sum())
            scores[name] = float(column.mean())
        else:
            counts[name] = 0
            scores[name] = float("nan")

    detail_rows = [
        {
            "问题": row["user_input"],
            **{name: row.get(name) for name in METRIC_NAMES},
        }
        for _, row in frame.iterrows()
    ]

    elapsed = time.perf_counter() - started
    write_report(scores, counts, detail_rows, elapsed, len(collected))

    print(f"\n报告已写入：{REPORT_PATH}")
    for name in METRIC_NAMES:
        print(f"  {name:<20} {scores[name]:.4f}")
    print(f"总耗时 {elapsed:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
