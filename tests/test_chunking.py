"""切分与内容指纹。不碰网络、不需要 API Key。"""

from __future__ import annotations

from python_ai.chunking import SEPARATORS, estimate_chunk_count, sha256_hex, split_text


def test_short_text_stays_one_piece() -> None:
    assert split_text("很短的一句话。", chunk_size=800, chunk_overlap=150) == ["很短的一句话。"]


def test_long_text_is_split() -> None:
    # 造一段没有换行的长中文 —— 靠标点分隔符切，不应该被硬切得支离破碎
    text = "".join(f"这是第{i}句话，用来测试切分行为。" for i in range(120))
    pieces = split_text(text, chunk_size=200, chunk_overlap=40)

    assert len(pieces) > 1
    # 除最后一片外，每片都应该接近 chunk_size（说明是按语义切的，不是碎片）
    assert all(len(p) <= 200 for p in pieces[:-1])


def test_overlap_carries_context() -> None:
    text = "".join(f"句子{i}。" for i in range(200))
    pieces = split_text(text, chunk_size=100, chunk_overlap=50)

    assert len(pieces) >= 2
    # 相邻两片之间应当有重叠内容（overlap 生效）
    assert pieces[0][-20:] in pieces[1] or pieces[1][:20] in pieces[0]


def test_no_content_is_lost() -> None:
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    pieces = split_text(text, chunk_size=12, chunk_overlap=2)
    joined = "".join(pieces)

    for fragment in ("第一段", "第二段", "第三段"):
        assert fragment in joined


def test_blank_text_yields_nothing() -> None:
    assert split_text("   \n\n  \t ") == []


def test_separators_cover_chinese_punctuation() -> None:
    """中文没有空格可切，每个句末标点都得在分隔表里。"""
    for punct in ("。", "！", "？", "；"):
        assert punct in SEPARATORS


def test_sha256_is_stable_and_hex() -> None:
    assert sha256_hex(b"abc") == sha256_hex(b"abc")
    assert sha256_hex(b"abc") != sha256_hex(b"abd")
    assert len(sha256_hex(b"x")) == 64
    assert all(c in "0123456789abcdef" for c in sha256_hex(b"x"))


def test_estimate_chunk_count() -> None:
    assert estimate_chunk_count("") == 0
    assert estimate_chunk_count("a" * 800) == 1
    assert estimate_chunk_count("a" * 801) == 2
