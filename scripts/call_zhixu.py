"""Standard-library example client; calls the model only with an explicit --message."""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description="调用知序外部问答接口（可能产生模型费用）")
    parser.add_argument("--message", required=True)
    parser.add_argument("--client-id", default="example-app")
    parser.add_argument("--session-id")
    parser.add_argument("--mode", choices=["RAG", "CHAT"], default="RAG")
    args = parser.parse_args()
    base = os.environ.get("ZHIXU_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    key = os.environ.get("ZHIXU_API_KEY", "")
    if not key:
        raise SystemExit("请在调用方环境设置 ZHIXU_API_KEY；不要把密钥写进代码")
    address = urlsplit(base)
    if address.scheme not in {"http", "https"} or not address.netloc:
        raise SystemExit("ZHIXU_BASE_URL 必须是有效的 http/https 服务地址")
    if address.scheme == "http" and address.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise SystemExit("跨机器接入请使用 HTTPS 或在受控隧道内通过本机地址访问")
    body = {"clientId": args.client_id, "message": args.message, "mode": args.mode}
    if args.session_id:
        body["sessionId"] = args.session_id
    request = Request(
        base + "/api/integration/v1/ask",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    # Must exceed the default 180-second server budget. Increase together with it.
    try:
        with urlopen(request, timeout=210) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            error = json.loads(exc.read())
            message = (
                f"code={error.get('code')}, message={error.get('message')}, "
                f"requestId={error.get('requestId')}"
            )
        except ValueError:
            message = "非知序 JSON 错误，请检查反向代理与服务地址"
        raise SystemExit(f"HTTP {exc.code}: {message}") from exc
    except (URLError, TimeoutError) as exc:
        raise SystemExit("连接失败或客户端等待超时；请检查服务和网络，不自动重试问答") from exc
    if result.get("serviceId") != "zhixu.qa.v1" or result.get("code") != 0:
        raise SystemExit("服务标识或业务结果不符合 zhixu.qa.v1，请检查调用地址")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
