"""文档解析：把上传的字节流变成纯文本。

对应 Java 的 `DocumentParserFactory`。按扩展名分派：
- `.pdf` → pypdf 抽文本（纯 Python 实现，无编译依赖，对 3.14 更稳）
- 文本类扩展名 → **显式按 UTF-8 解码**（Java 侧踩过这个坑：Windows 默认编码是 GBK，
  不指定会乱码）
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from .errors import BizError, ResultCode

TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".text",
        ".log",
        ".md",
        ".markdown",
        ".java",
        ".py",
        ".xml",
        ".yml",
        ".yaml",
        ".json",
        ".properties",
        ".sql",
        ".csv",
    }
)


def source_type_of(filename: str) -> str:
    """取小写扩展名，用于落库和展示。"""
    return Path(filename).suffix.lower().lstrip(".")


def parse_document(filename: str, data: bytes) -> str:
    """按扩展名解析，返回纯文本。不支持的类型抛 `BizError`。"""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(data)

    if suffix in TEXT_SUFFIXES:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BizError(
                ResultCode.PARAM_INVALID,
                "文件不是 UTF-8 编码，请转成 UTF-8 后再上传",
            ) from exc

    raise BizError(
        ResultCode.FILE_TYPE_UNSUPPORTED,
        f"不支持的文件类型：{suffix or '(无扩展名)'}",
    )


def _parse_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise BizError(ResultCode.PARAM_INVALID, f"PDF 解析失败：{exc}") from exc

    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    if not text:
        raise BizError(ResultCode.FILE_EMPTY, "PDF 里没有可提取的文本（可能是扫描件）")
    return text
