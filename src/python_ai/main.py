"""FastAPI 应用入口。对应 Java 的 `AiRagApplication` + `GlobalExceptionHandler`。

启动顺序：检查迁移版本 → 初始化会话检查点 → 校验向量集合 → 对外服务。
任何一步失败都直接让进程起不来（fail-fast）—— 带着错误配置跑起来，
问题只会在第一个请求时才暴露，更难查。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import checkpoint_memory
from .api import chat, documents, health, integration
from .clients import close_clients
from .config import PROJECT_ROOT, get_settings
from .db.session import dispose_db, init_db
from .errors import BizError, ResultCode
from .observability import close_tracing
from .schemas import Result
from .vectorstore import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    maintenance = None
    try:
        await init_db()
        if get_settings().memory_backend == "langgraph":
            await checkpoint_memory.initialize()

            async def sweep():
                while True:
                    await asyncio.sleep(60)
                    try:
                        await checkpoint_memory.cleanup()
                    except Exception:
                        logger.exception("会话过期清理失败")

            maintenance = asyncio.create_task(sweep())
        logger.info("PostgreSQL 就绪（业务表迁移版本已验证）")
        dimension = await ensure_collection()
        logger.info("Qdrant 就绪（集合维度 %d，已与向量模型实测值校验）", dimension)
        yield
    finally:
        if maintenance:
            maintenance.cancel()
            await asyncio.gather(maintenance, return_exceptions=True)
        try:
            for finish in (checkpoint_memory.close, close_tracing):
                try:
                    await finish()
                except Exception:
                    logger.exception("释放会话/追踪资源失败")
            await dispose_db()
        finally:
            await close_clients()
        logger.info("资源已释放")


app = FastAPI(
    title="python-ai RAG",
    description="Python + AI 复刻项目：RAG 检索问答 + Rerank + 评测。接口与 java-ai-rag 对齐。",
    version="0.1.0",
    lifespan=lifespan,
    response_model_exclude_none=True,
    # 关掉 FastAPI 自带的文档页 —— 它从 jsdelivr CDN 拉 JS/CSS，
    # 国内直连不通（实测状态码 000），页面会白屏。改用下面自建的本地版。
    docs_url=None,
    redoc_url=None,
)


def _custom_openapi() -> dict:
    """改写 OpenAPI schema，让 Swagger UI 能正确渲染文件上传框。

    **背景**：FastAPI 默认输出 OpenAPI 3.1，Pydantic v2 把上传字段描述成
    `{"type": "string", "contentMediaType": "application/octet-stream"}`。
    但 Swagger UI 只认 OpenAPI 3.0 的 `{"format": "binary"}` —— 认不出来就把上传框
    渲染成一个**普通文本框**。用户在里面填什么都发不出真文件，服务端收到空文件直接
    返回 422。这个坑很隐蔽：接口本身没问题，是文档页渲染错了。

    修法：把 openapi 版本号降到 3.0.3，并把所有 `contentMediaType:
    application/octet-stream` 改写成 `format: binary`。
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"

    for component in (schema.get("components", {}).get("schemas") or {}).values():
        for prop in (component.get("properties") or {}).values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[method-assign]

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(integration.router)

# ── 接口文档页：静态资源全部本地化，不依赖任何 CDN ──

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def demo_home() -> FileResponse:
    """本地演示工作台，与 API 同源，浏览本身不调用模型。"""
    return FileResponse(STATIC_DIR / "demo" / "index.html")


@app.get("/demo/sample", include_in_schema=False)
async def demo_sample() -> FileResponse:
    return FileResponse(
        PROJECT_ROOT / "data" / "samples" / "star-link-spec.md",
        filename="star-link-spec.md",
        media_type="text/markdown; charset=utf-8",
    )


@app.get("/demo/ocr-sample", include_in_schema=False)
async def demo_ocr_sample(kind: Literal["mixed", "scan", "image"] = "mixed") -> FileResponse:
    filename = {
        "mixed": "crane-mixed.pdf",
        "scan": "crane-scanned.pdf",
        "image": "crane-scan.png",
    }[kind]
    return FileResponse(PROJECT_ROOT / "data" / "samples" / "ocr" / filename, filename=filename)


@app.get("/compare", include_in_schema=False)
async def compare_home() -> FileResponse:
    """两个独立项目的交互入口；打开页面不会自动发起模型请求。"""
    return FileResponse(STATIC_DIR / "demo" / "compare.html")


@app.get("/demo/guide", response_class=HTMLResponse, include_in_schema=False)
async def demo_guide(stages: bool = False) -> str:
    """以可阅读、可打印的页面展示同一份验收指南。"""
    return _guide_html("升级功能验收.md" if stages else "功能验收.md")


@app.get("/demo/operations", response_class=HTMLResponse, include_in_schema=False)
async def operations_guide(
    document: Literal["run", "troubleshoot", "verify", "stages", "compare", "records"] = "run",
) -> str:
    return _guide_html(
        {
            "run": "运行手册.md",
            "troubleshoot": "故障排查.md",
            "verify": "功能验收.md",
            "stages": "升级功能验收.md",
            "compare": "版本对照.md",
            "records": "验证记录.md",
        }[document]
    )


@app.get("/demo/comparison-guide", response_class=HTMLResponse, include_in_schema=False)
@app.get("/static/demo/comparison-guide.md", response_class=HTMLResponse, include_in_schema=False)
async def comparison_guide() -> str:
    """保留旧链接，但正文仅存于运维目录。"""
    return _guide_html("版本对照.md")


def _guide_html(filename: str) -> str:
    guide = (PROJECT_ROOT / "运维" / filename).read_text(encoding="utf-8")
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>知序 · 验收指南</title><style>"
        "body{max-width:950px;margin:40px auto;padding:0 24px;background:#f6f7f1;color:#172e2b;"
        'font:15px/1.9 "Segoe UI","Microsoft YaHei",sans-serif}'
        "a{color:#236c50}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;"
        "background:white;border:1px solid #e4e7de;padding:30px;border-radius:14px}"
        "@media print{body{background:white;margin:0}pre{border:0;padding:0}}"
        '</style><nav><a href="/">← 返回工作台</a> · '
        '<a href="/demo/operations">运行手册</a> · '
        '<a href="/demo/operations?document=troubleshoot">故障排查</a> · '
        '<a href="/demo/guide">功能验收</a> · '
        '<a href="/demo/guide?stages=true">升级功能验收</a></nav><pre>'
        + escape(guide)
        + "</pre></html>"
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui() -> HTMLResponse:
    """Swagger UI 交互式文档页（离线可用）。

    资源文件是从国内镜像（bootcdn）一次性下载到 `src/python_ai/static/` 的，
    换网络环境也不会白屏。升级方式见 README「踩过的坑」。
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} · 接口文档",
        swagger_js_url="/static/swagger-ui-bundle.min.js",
        swagger_css_url="/static/swagger-ui.min.css",
        swagger_favicon_url="",  # 默认的 favicon 也在外网，去掉避免拖慢首屏
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── 全局异常处理：业务错误统一 HTTP 200 + code != 0，与 Java 侧同构 ──


@app.exception_handler(BizError)
async def handle_biz_error(_: Request, exc: BizError) -> JSONResponse:
    logger.log(
        logging.WARNING if exc.retryable else logging.INFO,
        "业务异常：%s",
        exc,
    )
    return JSONResponse(
        status_code=200,
        content=Result.fail(int(exc.code), exc.message).model_dump(by_alias=True),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}" for err in exc.errors()
    )
    return JSONResponse(
        status_code=200,
        content=Result.fail(int(ResultCode.PARAM_INVALID), details or "参数不合法").model_dump(
            by_alias=True
        ),
    )


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    # 不把堆栈回给调用方，日志里留全量
    logger.exception("未预期异常：%s", exc)
    return JSONResponse(
        status_code=500,
        content=Result.fail(int(ResultCode.INTERNAL_ERROR), "系统内部错误").model_dump(
            by_alias=True
        ),
    )


def main() -> None:
    """`uv run python-ai` 直接起服务。"""
    uvicorn.run("python_ai.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
