"""备份本项目默认 Compose PostgreSQL；二进制直接写文件，不经过 PowerShell 文本管道。"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from python_ai.config import PROJECT_ROOT, get_settings


def main():
    settings = get_settings()
    if (
        settings.postgres_host not in {"localhost", "127.0.0.1"}
        or settings.postgres_port != 5432
        or settings.postgres_db != "python_ai"
        or settings.postgres_user != "postgres"
    ):
        raise SystemExit("配置不是本项目默认 Compose 数据库，请使用对应数据库的专用备份方式。")
    executable = shutil.which("docker") or str(
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"
    )
    if not Path(executable).is_file():
        raise SystemExit("未找到 Docker CLI，请确认 Docker Desktop 已安装并启动。")
    identity = subprocess.run(
        [executable, "inspect", "pyai-postgres", "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        check=True,
        text=True,
    )
    labels = json.loads(identity.stdout)
    if labels.get("com.docker.compose.project") != "python-ai":
        raise SystemExit("容器不属于 python-ai Compose 工程，已停止备份。")
    directory = PROJECT_ROOT / ".demo-run/backups"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"python_ai-{time.time_ns()}.dump"
    try:
        with target.open("xb") as output:
            subprocess.run(
                [
                    executable,
                    "exec",
                    "pyai-postgres",
                    "pg_dump",
                    "-U",
                    "postgres",
                    "-d",
                    "python_ai",
                    "-Fc",
                ],
                stdout=output,
                check=True,
            )
    except Exception:
        target.unlink(missing_ok=True)
        raise
    print(f"PostgreSQL 备份完成：{target} ({target.stat().st_size} bytes)")
    print("仅备份 PostgreSQL，Qdrant 和 Redis 不在该文件中。")


if __name__ == "__main__":
    main()
