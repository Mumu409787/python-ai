"""生成 OCR 验收样例；PowerShell 使用项目 Python 运行，依赖 Windows 微软雅黑字体。"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data/samples/ocr"
    target.mkdir(parents=True, exist_ok=True)
    font = "C:/Windows/Fonts/msyh.ttc"
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 24), fill="#24634c")
    lines = [
        ("OCR 验收 - 苍鹭测试服务", 110, 64),
        ("苍鹭图像服务端口是 18743。", 280, 52),
        ("请求失败退避时间是 9 秒。", 390, 52),
        ("OCR acceptance code: CRANE-18743", 540, 42),
        ("仅用于测试，请勿当作真实生产配置。", 720, 34),
    ]
    for text, y, size in lines:
        draw.text((100, y), text, font=ImageFont.truetype(font, size), fill="#183c32")
    image.save(target / "crane-scan.png")
    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=200)
    (target / "crane-scanned.pdf").write_bytes(buffer.getvalue())
    writer = PdfWriter()
    page = writer.add_blank_page(width=576, height=324)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 20 Tf 36 260 Td (Native text - OCR acceptance) Tj "
        b"0 -60 Td /F1 16 Tf (Native reference port: 29012.) Tj "
        b"0 -40 Td (Page two contains a Chinese scanned image.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_page(PdfReader(io.BytesIO(buffer.getvalue())).pages[0])
    with (target / "crane-mixed.pdf").open("wb") as file:
        writer.write(file)
    image.close()
    print(target)


if __name__ == "__main__":
    main()
