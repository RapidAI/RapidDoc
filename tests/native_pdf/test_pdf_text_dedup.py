"""验证来源感知去重不会损坏连字、独立字符及唯一隐藏 OCR 文本。"""

from __future__ import annotations

import copy
import math
from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfgen.canvas import Canvas

from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox, Char
from rapid_doc._vendor.docvortex_core.document.pdf.text.dedup import deduplicate_chars


def _char(
    text: str, x: float = 0, y: float = 0, *, obj: int | None = 0, mode: int | None = 0, width: float = 10, angle: float = 0
) -> Char:
    """构造有真实来源语义的字符，可模拟不同对象和渲染模式。"""
    return {
        "char": text,
        "char_idx": 0,
        "source_indices": (0,),
        "bbox": Bbox([x, y, x + width, y + 12]),
        "origin": (x, y + 10),
        "font": {"name": "test", "flags": 0, "size": 12, "weight": 400},
        "rotation": -angle,
        "writing_angle": angle,
        "text_object_id": obj,
        "text_render_mode": mode,
    }


def _line(text: str, x: float = 0, y: float = 0, *, obj: int = 0, mode: int = 0) -> list[Char]:
    """生成连续字形，书写推进与源索引分别由各层保持。"""
    return [_char(c, x + i * 10, y, obj=obj, mode=mode) for i, c in enumerate(text)]


def _indexed(chars: list[Char]) -> list[Char]:
    """为完整字符流分配连续源索引，不让测试辅助数据伪造同一字形。"""
    return [dict(c, char_idx=i, source_indices=(i,)) for i, c in enumerate(chars)]


def _text(chars: list[Char]) -> str:
    """提取结果中的完整字符序列。"""
    return "".join(c["char"] for c in chars)


@pytest.mark.parametrize("text", ["ff", "ffi", "ffl", "fi", "fl", "a\u0301", "人人", "𝜃𝜃", "AB"])
def test_same_glyph_mapping_is_preserved(text: str) -> None:
    """同字形的多码值默认全部保留，不通过重复字母或汉字白名单删字。"""
    chars = _indexed([_char(c) for c in text])
    assert _text(deduplicate_chars(chars)) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("⼒力力", "力"),
        ("年年", "年"),
        ("⻓长", "长"),
        ("⻔门", "门"),
        ("⺠民", "民"),
        ("⾏行行", "行"),
        ("⻓門", "⻓門"),
        ("⼒力力", "力"),
    ],
)
def test_only_equivalent_han_mapping_is_collapsed(text: str, expected: str) -> None:
    """不同编码须全部等价，合并后仍可追溯所有原始字符。"""
    chars = _indexed([_char(c) for c in text])
    before = copy.deepcopy(chars)
    result = deduplicate_chars(chars)
    assert _text(result) == expected
    assert sorted({i for c in result for i in c["source_indices"]}) == list(range(len(text)))
    assert _text(chars) == _text(before)
    assert all(c["source_indices"] == b["source_indices"] for c, b in zip(chars, before))


@pytest.mark.parametrize("text", ["人人", "年年", "ff", "llll", "==", "∑∑", "⻓长"])
def test_distinct_origins_are_not_mapping_duplicates(text: str) -> None:
    """紧排字符即使框近重合，独立原点仍须保留。"""
    chars = _indexed([_char(c, i * 0.3, width=1) for i, c in enumerate(text)])
    assert _text(deduplicate_chars(chars)) == text


def test_missing_provenance_keeps_identical_characters() -> None:
    """缺失来源、原点或有效几何时不能恢复旧的猜测删除逻辑。"""
    chars = _indexed([_char("f", obj=None), _char("f", obj=None), _char("年", obj=1), _char("年", obj=1)])
    for c in chars[2:]:
        c["origin"] = None
    assert _text(deduplicate_chars(chars)) == "ff年年"


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("offset", [(0, 0), (2.04, 2.04), (1.5, 0), (0, 1.5)])
def test_repeated_paint_layers_keep_one_complete_sequence(interleaved: bool, layers: int, offset: tuple[float, float]) -> None:
    """整段与逐字交错的两层、三层绘制均保留完整的一份文字。"""
    lines = [_line("ABCABC", n * offset[0], n * offset[1], obj=n) for n in range(layers)]
    chars = _indexed([c for group in (zip(*lines) if interleaved else lines) for c in group])
    result = deduplicate_chars(chars)
    assert _text(result) == "ABCABC"
    assert sorted({i for c in result for i in c["source_indices"]}) == list(range(len(chars)))
    assert all(c["text_object_id"] == 0 for c in result)
    assert _text(deduplicate_chars(result)) == "ABCABC"


def test_repeated_layers_with_ligatures_preserve_each_piece() -> None:
    """同一连字的三段码值必须共同参与重复层匹配。"""
    first = [_char(c) for c in "ffi"] + [_char("c", 10), _char("e", 20)]
    second = [dict(c, text_object_id=1) for c in first]
    result = deduplicate_chars(_indexed(first + second))
    assert _text(result) == "ffice"
    assert result[0]["source_indices"] == (0, 5)
    assert result[1]["source_indices"] == (1, 6)


@pytest.mark.parametrize("text", ["A", "AA", "AB"])
def test_offset_requires_continuous_varied_text(text: str) -> None:
    """孤立同字、单调重复和不足三个字形的偏移不构成阴影证据。"""
    chars = _indexed(_line(text) + _line(text, 2, 2, obj=1))
    assert _text(deduplicate_chars(chars)) == text * 2


def test_non_contiguous_offset_candidates_are_preserved() -> None:
    """分散的相同字符不能拼成一条虚假的阴影序列。"""
    a = [_char(c, i * 80) for i, c in enumerate("ABC")]
    b = [_char(c, i * 80 + 2, 2, obj=1) for i, c in enumerate("ABC")]
    assert _text(deduplicate_chars(_indexed(a + b))) == "ABCABC"


def test_unique_hidden_text_is_preserved() -> None:
    """扫描页唯一隐藏 OCR 层以及不同位置的同文不能删除。"""
    chars = _indexed(_line("OCR only", mode=3) + _line("Visible", y=100, obj=1))
    assert _text(deduplicate_chars(chars)) == "OCR onlyVisible"


@pytest.mark.parametrize(
    "visible,hidden,removed",
    [
        ("建筑物理与设备", "莲筑物理与设备", True),
        ("九门提督带你精读设备", "力门提督带你精读设备", True),
        ("GB 50736-2012", "GB50736-2012", True),
        ("第1条规范", "第2条规范", False),
        ("some text", "some texts", False),
        ("some text", "same text", False),
        ("规范", "现范", False),
        ("规范文件", "规范文件补充", False),
    ],
)
def test_hidden_requires_visible_content_and_conservative_match(visible: str, hidden: str, removed: bool) -> None:
    """隐藏副本允许明确的汉字 OCR 替换，不允许数字、字母或增删内容差异。"""
    # 空格不占字形，模拟隐藏 OCR 和可见文字的空格编码差异。
    visible_chars = _line(visible.replace(" ", ""))
    hidden_chars = _line(hidden.replace(" ", ""), obj=1, mode=3)
    for c in hidden_chars:
        c["font"] = {**c["font"], "name": "OCR font"}
    result = deduplicate_chars(_indexed(visible_chars + hidden_chars))
    assert _text(result) == visible.replace(" ", "") + ("" if removed else hidden.replace(" ", ""))


def test_hidden_partial_overlap_and_adjacent_cells_are_preserved() -> None:
    """覆盖不足以及旁边单元格的同文不能抑制隐藏内容。"""
    for x in (7, 100):
        chars = _indexed(_line("ABC") + _line("ABC", x=x, obj=1, mode=3))
        assert _text(deduplicate_chars(chars)) == "ABCABC"


def test_rotated_hidden_duplicate_and_page_regions() -> None:
    """文字轴投影支持旋转副本，同时保留另一个位置上的正文。"""
    chars = _indexed(_line("ABCD") + _line("ABCD", obj=1, mode=3) + _line("ABCD", y=100, obj=2))
    for c in chars:
        x0, y0, x1, y1 = c["bbox"]
        c["bbox"] = Bbox([200 - y1, x0, 200 - y0, x1])
        x, y = c["origin"]
        c["origin"] = (200 - y, x)
        c["writing_angle"] = math.pi / 2
        c["rotation"] = 0
    assert _text(deduplicate_chars(chars)) == "ABCDABCD"


def _mapping_pdf(mapping: str) -> bytes:
    """构造真实 ToUnicode 一对多映射，验证 PDFium 提取与两种公开字符入口。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=(200, 100))
    canvas.drawString(20, 50, "X")
    canvas.save()
    reader = PdfReader(BytesIO(output.getvalue()))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    font = writer.pages[0]["/Resources"]["/Font"]["/F1"].get_object()
    cmap = DecodedStreamObject()
    target = mapping.encode("utf-16-be").hex()
    cmap.set_data(
        f"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /Test def /CMapType 2 def 1 begincodespacerange <00><FF> endcodespacerange 1 beginbfchar <58><{target}> endbfchar endcmap CMapName currentdict /CMap defineresource pop end end".encode()
    )
    font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize(
    "mapping,expected", [("ffi", "ffi"), ("ﬃ", "ffi"), ("ff", "ff"), ("⼒力力", "力"), ("⻓长", "长"), ("𝜃", "𝜃")]
)
def test_real_pdfium_mapping_and_geometry_interfaces(mapping: str, expected: str) -> None:
    """基础和扩展接口输出相同语义，连字每段与代理对来源均被保留。"""
    with PDFDocument(_mapping_pdf(mapping)) as document:
        base = document.get_page_chars(0)
        extended = document.get_page_chars_with_geometry(0)
    assert _text(base) == expected
    assert _text(extended.chars) == expected
    assert [(c["char_idx"], c["source_indices"], list(c["bbox"])) for c in base] == [
        (c["char_idx"], c["source_indices"], list(c["bbox"])) for c in extended.chars
    ]
    assert all(type(c["text_object_id"]) is int for c in base)
    assert all(c["char_idx"] in extended.origins for c in extended.chars)


def test_recorded_badcase_characters() -> None:
    """重放五个真实样例的最小字符片段，无需外接硬盘或完整业务文档。"""
    import json
    from pathlib import Path

    # 固定 UTF-8，避免 Windows 默认编码将单个汉字解码成多个字符。
    fixtures = json.loads((Path(__file__).parents[1] / "fixtures/pdf_char_dedup.json").read_text(encoding="utf-8"))
    for case in fixtures:
        chars = [{**c, "bbox": Bbox(c["bbox"]), "source_indices": tuple(c["source_indices"])} for c in case["chars"]]
        assert _text(deduplicate_chars(chars)) == case["expected"], case["name"]


def test_pdfium_shear_angle_does_not_change_writing_direction() -> None:
    """真实 PDF 人工斜体剪切不应被当成书写基线的旋转。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=(300, 150))
    canvas.transform(1, 0, 1 / 3, 1, 0, 0)
    canvas.drawString(20, 80, "SHEAR BASELINE")
    canvas.save()
    with PDFDocument(output.getvalue()) as document:
        chars = document.get_page_chars(0)
    assert any(abs(c["rotation"]) > 0.1 for c in chars)
    assert all(abs(c["writing_angle"]) < 0.001 for c in chars if c["char"].strip())


def test_removed_hidden_paragraph_leaves_no_orphaned_lines() -> None:
    """已删除副本之间的空白被清除，原有正文段落分隔仍保留。"""
    chars = _indexed(
        _line("ABC") + [_char("\r"), _char("\n")] + _line("ABC", obj=1, mode=3) + [_char("\r", obj=None), _char("\n", obj=None)]
    )
    assert _text(deduplicate_chars(chars)) == "ABC\r\n"


def test_transparent_text_cannot_suppress_unique_hidden_ocr() -> None:
    """填充模式但透明度为零的文字不能被当作可见原文。"""
    visible = _line("ABCD")
    for c in visible:
        c["text_is_visible"] = False
    chars = _indexed(visible + _line("ABCD", obj=1, mode=3))
    assert _text(deduplicate_chars(chars)) == "ABCDABCD"


def test_pdfium_reports_transparent_objects_without_losing_ocr() -> None:
    """从真实 PDF 读取对象透明度，并保留其旁边的唯一隐藏文本。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=(300, 150))
    canvas.setFillAlpha(0)
    canvas.drawString(20, 80, "TRANSPARENT")
    canvas.setFillAlpha(1)
    text = canvas.beginText(20, 40)
    text.setTextRenderMode(3)
    text.textOut("OCR ONLY")
    canvas.drawText(text)
    canvas.save()
    with PDFDocument(output.getvalue()) as document:
        chars = document.get_page_chars(0)
    assert "OCR ONLY" in _text(chars)
    assert all(not c.get("text_is_visible") for c in chars)


def test_shadow_evidence_cannot_skip_different_interior_characters() -> None:
    """三个相同字母之间有不同正文时，不能跳过差异并删除一部分原文。"""
    chars = _indexed(_line("AxBxC") + _line("AyByC", 1, 1, obj=1))
    for c in chars:
        b = c["bbox"]
        c["bbox"] = Bbox([b[0] * 0.4, b[1], b[2] * 0.4, b[3]])
        x, y = c["origin"]
        c["origin"] = (x * 0.4, y)
    assert _text(deduplicate_chars(chars)) == "AxBxCAyByC"


def test_three_layers_of_two_glyphs_are_still_insufficient_evidence() -> None:
    """三层叠字仍只有两个独立字形，不能靠副本数量凑足连续证据。"""
    layers = [_line("AB", i * 1.5, obj=i) for i in range(3)]
    chars = _indexed([c for group in zip(*layers) for c in group])
    assert len(deduplicate_chars(chars)) == 6


def test_invalid_and_extreme_geometry_is_conservatively_preserved() -> None:
    """非有限方向、零面积及极大隐藏框不会误删正文或枚举无限空网格。"""
    cases = [_char("A", obj=1, mode=3), _char("A", obj=1, mode=3), _char("A", obj=1, mode=3)]
    cases[0]["writing_angle"] = float("nan")
    cases[1]["bbox"] = Bbox([0, 0, 0, 0])
    cases[2]["bbox"] = Bbox([-1e12, -1e12, 1e12, 1e12])
    for other in cases:
        assert _text(deduplicate_chars(_indexed([_char("A"), other]))) == "AA"
