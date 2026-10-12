"""四阶段环境盘点：只读，不调用收费模型，也不创建/迁移数据库表。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "checkpoint_and_tracing": [
        "langgraph",
        "langgraph-checkpoint",
        "langgraph-checkpoint-postgres",
        "psycopg",
        "psycopg-binary",
        "psycopg-pool",
        "langfuse",
    ],
    "migration": ["alembic", "sqlalchemy", "asyncpg"],
    "ocr": ["pypdf", "pillow", "pypdfium2", "pytesseract"],
}


def versions(names: list[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {
            "ok": result.returncode == 0,
            "exitCode": result.returncode,
            "output": result.stdout.strip()[:2000],
            "error": result.stderr.strip()[:1000] if result.returncode else None,
        }
    except OSError as exc:
        return {"ok": False, "errorType": type(exc).__name__}
    except subprocess.TimeoutExpired:
        return {"ok": False, "errorType": "TimeoutExpired"}


async def services(settings) -> dict:
    import httpx
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from python_ai.clients import close_clients, redis_client

    result = {}
    engine = create_async_engine(settings.postgres_dsn, connect_args={"timeout": 5})
    try:
        async with engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            "SELECT current_database() AS database, "
                            "has_database_privilege(current_user, current_database(), 'CREATE') "
                            "AS can_create_schema, "
                            "has_schema_privilege(current_user, 'public', 'CREATE') "
                            "AS can_create_tables"
                        )
                    )
                )
                .mappings()
                .one()
            )
            result["postgres"] = {"ok": True, **dict(row)}
    except Exception as exc:
        result["postgres"] = {"ok": False, "errorType": type(exc).__name__}
    finally:
        await engine.dispose()
    try:
        async with asyncio.timeout(5):
            result["redis"] = {"ok": bool(await redis_client().ping())}
    except Exception as exc:
        result["redis"] = {"ok": False, "errorType": type(exc).__name__}
    finally:
        await close_clients()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                settings.qdrant_url.rstrip("/") + "/collections/" + settings.qdrant_collection
            )
            result["qdrant"] = {
                "ok": response.is_success,
                "collection": settings.qdrant_collection,
                "httpStatus": response.status_code,
            }
    except Exception as exc:
        result["qdrant"] = {"ok": False, "errorType": type(exc).__name__}
    return result


async def inspect(check_services: bool) -> dict:
    from python_ai.config import get_settings

    settings = get_settings()
    result = {
        "python": platform.python_version(),
        "pythonExecutable": sys.executable,
        "packages": {stage: versions(names) for stage, names in PACKAGES.items()},
        "langfuse": {
            "publicKeyPresent": bool(settings.langfuse_public_key.strip()),
            "secretKeyPresent": bool(settings.langfuse_secret_key.strip()),
            "host": settings.langfuse_host,
            "cloudAuthenticationVerified": False,
        },
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "freeDiskGiB": round(shutil.disk_usage(ROOT).free / 1024**3, 1),
    }
    engine = settings.tesseract_path or shutil.which("tesseract")
    conventional = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if not engine and conventional.is_file():
        engine = str(conventional)
    language_args = [engine, "--list-langs"] if engine else []
    if engine and settings.ocr_tessdata_dir.is_dir():
        language_args += ["--tessdata-dir", str(settings.ocr_tessdata_dir)]
    result["ocrEngine"] = {
        "executable": engine,
        "languages": command_output(language_args) if engine else None,
        "projectTessdata": str(settings.ocr_tessdata_dir),
    }
    training_python = ROOT / "training" / ".venv" / "Scripts" / "python.exe"
    result["training"] = {
        "pythonExists": training_python.is_file(),
        "runtime": command_output(
            [
                str(training_python),
                "-c",
                "import json,sys,sklearn,numpy,joblib; "
                "print(json.dumps({'python':sys.version.split()[0], "
                "'sklearn':sklearn.__version__, 'numpy':numpy.__version__, "
                "'joblib':joblib.__version__}))",
            ]
        )
        if training_python.is_file()
        else None,
    }
    if check_services:
        result["services"] = await services(settings)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", action="store_true", help="只读检查本项目三个本地服务")
    parser.add_argument("--output", type=Path, help="保存不含密钥的 JSON 报告")
    args = parser.parse_args()
    report = asyncio.run(inspect(args.services))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
