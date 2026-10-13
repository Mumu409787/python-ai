"""Generate a local integration key without printing it or replacing an existing key."""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from dotenv import dotenv_values


def configure(env_path: Path) -> bool:
    if not env_path.is_file():
        raise ValueError("未找到 .env；请先按 README 配置本项目运行环境")
    existing = dotenv_values(env_path).get("INTEGRATION_API_KEY")
    if existing:
        if len(existing) < 32 or existing != existing.strip():
            raise ValueError("已有 INTEGRATION_API_KEY 不合格；请手动修正，脚本不会覆盖它")
        return False
    content = env_path.read_bytes().decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in content else "\n"
    entry = "INTEGRATION_API_KEY=" + secrets.token_urlsafe(32)
    pattern = re.compile(r"^[ \t]*(?:export[ \t]+)?INTEGRATION_API_KEY[ \t]*=.*$", re.M)
    if pattern.search(content):
        content = pattern.sub(entry, content)
    else:
        content = content.rstrip("\r\n") + newline + newline + entry + newline
    env_path.write_bytes(content.encode("utf-8"))
    return True


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        changed = configure(env_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("Integration key configured." if changed else "Existing integration key preserved.")
    print(f"Key stored only in: {env_path}")
    print("Restart the Zhixu service to load configuration. The key was not printed.")


if __name__ == "__main__":
    main()
