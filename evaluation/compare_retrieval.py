"""只读召回对照：每题一次真实 embedding，无生成、重排、入库或删除。

PowerShell: uv run python -m evaluation.compare_retrieval
结果写入唯一新目录，不覆盖历史 Ragas 报告；小样本不代表总体质量。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from python_ai.clients import close_clients
from python_ai.config import get_settings
from python_ai.db.session import dispose_db, session_scope
from python_ai.lexical import bm25_recall
from python_ai.repository import searchable_chunks
from python_ai.retrieval import fuse_rrf, vector_recall

HERE = Path(__file__).resolve().parent


async def corpus_snapshot():
    async with session_scope() as session:
        rows = await searchable_chunks(session)
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    return rows, digest


async def run() -> Path:
    settings = get_settings()
    cases_path = HERE / "retrieval_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = HERE / "runs" / f"hybrid-{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    try:
        rows, fingerprint = await corpus_snapshot()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        metadata = {
            "runId": run_id,
            "codeRevision": revision,
            "workingTree": "may contain uncommitted changes; archive source with the run",
            "sourceSha256": {
                name: hashlib.sha256(
                    (HERE.parent / "src/python_ai" / name).read_bytes()
                ).hexdigest()
                for name in ("retrieval.py", "lexical.py", "vectorstore.py", "repository.py")
            },
            "corpusSha256": fingerprint,
            "chunks": len(rows),
            "datasetSha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "embeddingModel": settings.embedding_model,
            "collection": settings.qdrant_collection,
            "threshold": settings.retrieval_score_threshold,
            "candidateK": max(settings.bm25_candidate_k, settings.retrieval_top_k),
            "topK": settings.retrieval_top_k,
            "rrfK": settings.hybrid_rrf_k,
            "rerank": False,
            "generation": False,
            "writesBusinessData": False,
            "cost": "one real embedding request per case; token/currency amount unknown",
        }
        result = []
        for case in cases:
            item = dict(case)
            if case["expected"] and not any(case["expected"] in row[-1] for row in rows):
                item["error"] = "expected source missing from current corpus"
                result.append(item)
                continue
            try:
                dense = await vector_recall(
                    case["question"],
                    top_k=metadata["candidateK"],
                    threshold=settings.retrieval_score_threshold,
                )
                lexical = await bm25_recall(case["question"], top_k=metadata["candidateK"])
                mixed = fuse_rrf(dense, lexical, k=settings.hybrid_rrf_k) if lexical else dense
                for name, hits in (("vector", dense), ("hybrid", mixed)):
                    selected = hits[: settings.retrieval_top_k]
                    item[name] = {
                        "hit": any(case["expected"] in h.content for h in selected)
                        if case["expected"]
                        else not selected,
                        "chunks": [
                            {
                                "docId": h.doc_id,
                                "chunkId": h.chunk_id,
                                "score": h.score,
                                "scoreKind": h.score_kind,
                            }
                            for h in selected
                        ],
                    }
            except Exception as exc:
                item["error"] = type(exc).__name__
            result.append(item)
        _, final_fingerprint = await corpus_snapshot()
        metadata["corpusUnchanged"] = fingerprint == final_fingerprint
        (output / "results.json").write_text(
            json.dumps({"metadata": metadata, "cases": result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        valid = [item for item in result if "error" not in item]
        lines = [
            "# BM25 混合召回小样本对照",
            "",
            f"有效样本 {len(valid)}/{len(cases)}，异常/缺资料 {len(cases) - len(valid)}。",
            "仅比较召回；重排和生成均关闭。改写不参与本次对比。",
            f"期间语料未变化：{metadata['corpusUnchanged']}。",
            "命中表示前 K 个片段含人工指定原文；库外题检查零命中，不是模型正确率。",
            "",
            "| 问题 | 向量 | 混合 |",
            "| --- | --- | --- |",
        ]
        for item in result:
            if "error" in item:
                lines.append(f"| {item['question']} | 异常：{item['error']} | 未比较 |")
            else:
                lines.append(
                    f"| {item['question']} | {item['vector']['hit']} | {item['hybrid']['hit']} |"
                )
        lines += ["", "不能用此小样本宣称总体准确率提升；详见 results.json 的固定配置及语料指纹。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(output)
        return output
    finally:
        await dispose_db()
        await close_clients()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
