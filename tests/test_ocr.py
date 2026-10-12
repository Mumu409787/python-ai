import io
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pypdf import PdfWriter

from python_ai import ocr, parsing
from python_ai.config import PROJECT_ROOT, get_settings
from python_ai.errors import BizError


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_limits_failure_and_no_text(monkeypatch):
    monkeypatch.setattr(get_settings(), "ocr_max_pixels", 100)
    with pytest.raises(BizError, match="像素超限"):
        parsing.parse_pages("a.png", png())
    monkeypatch.setattr(get_settings(), "ocr_max_pixels", 20_000_000)
    with pytest.raises(BizError, match="图片损坏"):
        parsing.parse_pages("a.jpg", b"broken")
    monkeypatch.setattr(parsing, "recognize", lambda *a, **kw: "")
    with pytest.raises(BizError, match="未识别到有效文字"):
        parsing.parse_pages("a.png", png())
    frames = [Image.new("RGB", (32, 32), color) for color in ("white", "black")]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100)
    with pytest.raises(BizError, match="单帧图片"):
        parsing.parse_pages("animated.webp", buffer.getvalue())
    for frame in frames:
        frame.close()


def test_pdf_page_limit_and_encryption(monkeypatch):
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_blank_page(100, 100)
    buffer = io.BytesIO()
    writer.write(buffer)
    monkeypatch.setattr(get_settings(), "ocr_max_pages", 1)
    with pytest.raises(BizError, match="页数超限"):
        parsing.parse_pages("a.pdf", buffer.getvalue())
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(BizError, match="加密 PDF"):
        parsing.parse_pages("a.pdf", buffer.getvalue())


def test_native_pdf_does_not_need_ocr(monkeypatch):
    call = Mock(side_effect=AssertionError("native text must not invoke OCR"))
    monkeypatch.setattr(parsing, "recognize", call)
    from pypdf import PdfReader

    writer = PdfWriter()
    writer.add_page(PdfReader(PROJECT_ROOT / "data/samples/ocr/crane-mixed.pdf").pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)
    assert "29012" in parsing.parse_pages("native.pdf", buffer.getvalue())[0].text
    call.assert_not_called()


def test_ocr_timeout_missing_engine_and_argument_quoting(monkeypatch, tmp_path):
    image = Image.new("RGB", (32, 32), "white")
    monkeypatch.setattr(ocr, "executable", lambda: None)
    with pytest.raises(BizError, match="未找到"):
        ocr.recognize(image, deadline=time.monotonic() + 10)
    monkeypatch.setattr(ocr, "executable", lambda: "engine.exe")
    data = tmp_path / "language data"
    data.mkdir()
    monkeypatch.setattr(get_settings(), "ocr_tessdata_dir", data)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="你好 18743".encode()))
    monkeypatch.setattr(ocr.subprocess, "run", run)
    assert "18743" in ocr.recognize(image, deadline=time.monotonic() + 10)
    args = run.call_args.args[0]
    assert args[args.index("--tessdata-dir") + 1] == str(data)
    run.side_effect = subprocess.TimeoutExpired("ocr", 1)
    with pytest.raises(BizError, match="超时"):
        ocr.recognize(image, deadline=time.monotonic() + 10)
    with pytest.raises(BizError, match="总处理时间"):
        ocr.recognize(image, deadline=time.monotonic() - 1)
    image.close()
