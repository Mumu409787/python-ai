"""端到端冒烟测试：对**运行中的服务**打一遍全部接口。

    # 终端 1
    uv run uvicorn python_ai.main:app --port 8000
    # 终端 2
    uv run python scripts/smoke_test.py

会真的调用模型 API 并读写 Qdrant/PG，跑完自动清理自己创建的文档。
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "star-link-spec.md"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))


def main() -> int:
    client = httpx.Client(timeout=120)

    print("── 服务可达性 ──")
    try:
        health = client.get(f"{BASE}/api/health").json()
    except httpx.ConnectError:
        print("  连不上服务，请先启动：uv run uvicorn python_ai.main:app --port 8000")
        return 1
    deps = health.get("data", {})
    check("health 返回 0", health.get("code") == 0)
    check(
        "PG / Qdrant / Redis 都可用",
        all(deps.get(k) for k in ("postgres", "qdrant", "redis")),
        str(deps),
    )

    print("\n── 清理历史数据 ──")
    for doc in client.get(f"{BASE}/api/documents").json().get("data", []):
        client.delete(f"{BASE}/api/documents/{doc['id']}")
    check("起始状态为空", client.get(f"{BASE}/api/documents").json()["data"] == [])

    print("\n── 上传与去重 ──")
    with SAMPLE.open("rb") as fh:
        first = client.post(f"{BASE}/api/documents", files={"file": (SAMPLE.name, fh)}).json()
    check("首次上传成功", first["code"] == 0, str(first.get("data")))
    doc_id = first["data"]["docId"]
    check("切出多个片段", first["data"]["chunkCount"] > 1, f"{first['data']['chunkCount']} 片")

    with SAMPLE.open("rb") as fh:
        second = client.post(f"{BASE}/api/documents", files={"file": (SAMPLE.name, fh)}).json()
    check(
        "重复上传被去重（秒传）", second["data"]["duplicated"] and second["data"]["docId"] == doc_id
    )

    print("\n── 参数校验 ──")
    check(
        "空 message 被拒",
        client.post(f"{BASE}/api/chat", json={"message": ""}).json()["code"] == 1001,
    )
    check(
        "非法 mode 被拒",
        client.post(f"{BASE}/api/chat?mode=XXX", json={"message": "你好"}).json()["code"] == 1001,
    )
    check(
        "不存在的文档被拒",
        client.delete(f"{BASE}/api/documents/999999").json()["code"] == 1002,
    )

    print("\n── RAG 问答 ──")
    hit = client.post(f"{BASE}/api/chat", json={"message": "Token 的有效期是多久？"}).json()
    check("相关提问返回答案", hit["code"] == 0 and bool(hit["data"]["answer"]))
    check("带回了引用片段", len(hit["data"]["hits"]) > 0, f"{len(hit['data']['hits'])} 条")
    check("答案里有来源标注", "[" in hit["data"]["answer"])
    print(f"        → {hit['data']['answer'].strip()[:90]}...")

    miss = client.post(f"{BASE}/api/chat", json={"message": "推荐几家好吃的火锅店"}).json()
    check("无关提问被拒绝作答", "无法回答" in miss["data"]["answer"])
    check("拒绝时无引用", len(miss["data"]["hits"]) == 0)

    print("\n── 流式 SSE ──")
    events: dict[str, int] = {}
    text_parts: list[str] = []
    with client.stream(
        "POST", f"{BASE}/api/chat/stream", json={"message": "网关超时时间是多少？"}
    ) as resp:
        check(
            "Content-Type 是 event-stream",
            "text/event-stream" in resp.headers.get("content-type", ""),
        )
        current = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
                events[current] = events.get(current, 0) + 1
            elif line.startswith("data:") and current == "message":
                text_parts.append(line.split(":", 1)[1].lstrip())

    check("收到 retrieval 事件", events.get("retrieval", 0) == 1)
    check("收到多个 message 事件", events.get("message", 0) > 1, f"{events.get('message', 0)} 个")
    check("收到 done 事件", events.get("done", 0) == 1)
    check("没有 error 事件", events.get("error", 0) == 0)
    check("流式正文非空", len("".join(text_parts).strip()) > 10)
    print(f"        → {''.join(text_parts).strip()[:90]}...")

    print("\n── 删除与清理 ──")
    check("删除成功", client.delete(f"{BASE}/api/documents/{doc_id}").json()["code"] == 0)
    check("列表已为空", client.get(f"{BASE}/api/documents").json()["data"] == [])

    print(f"\n{'=' * 60}")
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项：" + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
