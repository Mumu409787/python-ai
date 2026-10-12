"""文档解析：把上传的字节流变成纯文本。

对应 Java 的 `DocumentParserFactory`。按扩展名分派：
- `.pdf` → pypdf 优先抽取原生文本；扫描页由 PDFium 渲染后本地 OCR
- 文字图片 → Pillow 校验与方向修正，Tesseract 识别
- 文本类扩展名 → **显式按 UTF-8 解码**（Java 侧踩过这个坑：Windows 默认编码是 GBK，
  不指定会乱码）
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pypdf import PdfReader

from .config import get_settings
from .errors import BizError, ResultCode
from .ocr import check_pixels, recognize

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


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


@dataclass
class ParsedPage:
    text: str
    source_page: int | None = None


def parse_document(filename: str, data: bytes) -> str:
    return "\n\n".join(page.text for page in parse_pages(filename, data))


def parse_pages(filename: str, data: bytes) -> list[ParsedPage]:
    suffix = Path(filename).suffix.lower()
    deadline = time.monotonic() + get_settings().ocr_total_timeout_seconds
    if suffix == ".pdf":
        return _parse_pdf_pages(data, deadline)
    if suffix in IMAGE_SUFFIXES:
        try:
            with Image.open(io.BytesIO(data)) as source:
                check_pixels(*source.size)
                if getattr(source, "n_frames", 1) != 1:
                    raise BizError(ResultCode.PARAM_INVALID, "请上传单帧图片，多帧图片请逐帧导出")
                image = ImageOps.exif_transpose(source).convert("RGB")
                try:
                    text = recognize(image, deadline=deadline)
                finally:
                    image.close()
        except BizError:
            raise
        except Exception as exc:
            raise BizError(ResultCode.PARAM_INVALID, "图片损坏或格式无法解析") from exc
        if not text:
            raise BizError(ResultCode.FILE_EMPTY, "图片未识别到有效文字，请提供清晰正向的文字图片")
        return [ParsedPage(text, 1)]
    if suffix in TEXT_SUFFIXES:
        try:
            return [ParsedPage(data.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise BizError(
                ResultCode.PARAM_INVALID, "文件不是 UTF-8 编码，请转成 UTF-8 后再上传"
            ) from exc
    raise BizError(ResultCode.FILE_TYPE_UNSUPPORTED, f"不支持的文件类型：{suffix or '(无扩展名)'}")


def _parse_pdf_pages(data, deadline):
    renderer = None
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise BizError(ResultCode.PARAM_INVALID, "暂不支持加密 PDF，请解密后上传")
        if len(reader.pages) > get_settings().ocr_max_pages:
            raise BizError(ResultCode.PARAM_INVALID, "PDF 页数超限，请拆分后上传")
        for index, page in enumerate(reader.pages):
            native = (page.extract_text() or "").strip()
            # 原生文字优先；有图片且原生文本很少时补 OCR，覆盖带短标题的扫描页。
            needs_ocr = not native or (len(native) < 40 and bool(page.images))
            if needs_ocr:
                if renderer is None:
                    renderer = pdfium.PdfDocument(data)
                rendered_page = renderer[index]
                bitmap = None
                try:
                    scale = get_settings().ocr_pdf_dpi / 72
                    width, height = rendered_page.get_size()
                    check_pixels(math.ceil(width * scale), math.ceil(height * scale))
                    bitmap = rendered_page.render(scale=scale)
                    image = bitmap.to_pil()
                    try:
                        text = recognize(image, deadline=deadline)
                    finally:
                        image.close()
                finally:
                    if bitmap is not None:
                        bitmap.close()
                    rendered_page.close()
                if native and native not in text:
                    text = native + "\n" + text
            else:
                text = native
            if text.strip():
                pages.append(ParsedPage(text.strip(), index + 1))
    except BizError:
        raise
    except Exception as exc:
        raise BizError(ResultCode.PARAM_INVALID, "PDF 解析失败，请检查文件是否完整") from exc
    finally:
        if renderer is not None:
            renderer.close()
    if not pages:
        raise BizError(ResultCode.FILE_EMPTY, "PDF 未提取或识别到有效文字")
    return pages
