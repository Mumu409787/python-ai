"""本地 OCR：单页/总时间预算、像素限制、EXIF 方向修正，不调用云模型。"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .config import PROJECT_ROOT, get_settings
from .errors import BizError, ResultCode


def executable():
    configured = get_settings().tesseract_path
    value = configured or shutil.which("tesseract")
    if not value:
        default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        value = str(default) if default.is_file() else None
    return value


def recognize(image, *, deadline: float) -> str:
    s = get_settings()
    if not s.ocr_enabled:
        raise BizError(ResultCode.PARAM_INVALID, "此文件需要 OCR，但 OCR_ENABLED 已关闭")
    command = executable()
    if not command:
        raise BizError(
            ResultCode.PARAM_INVALID, "未找到 Tesseract，请安装引擎或配置 TESSERACT_PATH"
        )
    remaining = min(s.ocr_timeout_seconds, deadline - time.monotonic())
    if remaining <= 0:
        raise BizError(ResultCode.PARAM_INVALID, "OCR 总处理时间已超限，请拆分文件后重试")
    # Use an argument list: pytesseract's Windows config splitting can preserve literal
    # quotes around tessdata paths and cause false "language not found" failures.
    temporary = PROJECT_ROOT / ".demo-run" / "ocr-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="page-", dir=temporary) as folder:
            image_path = Path(folder) / "input.png"
            image.save(image_path)
            args = [command, str(image_path), "stdout", "-l", s.ocr_language, "--psm", "3"]
            if s.ocr_tessdata_dir.is_dir():
                args += ["--tessdata-dir", str(s.ocr_tessdata_dir)]
            result = subprocess.run(
                args,
                capture_output=True,
                timeout=remaining,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        if result.returncode:
            raise BizError(
                ResultCode.PARAM_INVALID, "OCR 识别失败，请检查 OCR_LANGUAGE、引擎及对应语言包"
            )
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired as exc:
        raise BizError(ResultCode.PARAM_INVALID, "OCR 识别超时，请减小图片或拆分 PDF") from exc
    except BizError:
        raise
    except OSError as exc:
        raise BizError(ResultCode.PARAM_INVALID, "OCR 引擎无法运行，请检查安装路径和权限") from exc


def check_pixels(width, height):
    if width * height > get_settings().ocr_max_pixels:
        raise BizError(ResultCode.PARAM_INVALID, "图片像素超限，请缩小图片或降低 OCR_PDF_DPI")
