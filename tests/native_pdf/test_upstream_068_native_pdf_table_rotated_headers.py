"""验证单元格旋转书写的文字顺序、上下标及原生路径一致性。"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from rapid_doc._vendor.docvortex_core.analyzers.pdf import PDFTablePage, prepare_table_page, recover_table_region
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import text as table_text
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput, NativeTableRectangle, NativeTableRule
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.engine import recover_native_pdf_table
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument, PDFPageTextGeometry
from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox


def _real_table() -> tuple[PDFTablePage, dict]:
    """读取带原件指纹的真实几何及人工逐格真值，不依赖用户下载目录。"""
    path = Path(__file__).parents[1] / "fixtures/native_pdf_table_rotated_headers.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    geometry = PDFPageTextGeometry(
        chars=[{**char, "bbox": Bbox(char["bbox"])} for char in evidence["chars"]],
        tight_bboxes={int(key): tuple(value) for key, value in evidence["tight_bboxes"].items()},
        origins={int(key): tuple(value) for key, value in evidence["origins"].items()},
    )
    page = PDFTablePage(
        tuple(evidence["page_size"]),
        geometry,
        tuple(NativeTableRule(**rule) for rule in evidence["drawing_lines"]),
        tuple(NativeTableRectangle(**rect) for rect in evidence["rectangles"]),
    )
    return page, evidence


def test_datasheet_rotated_headers_match_all_source_cells() -> None:
    """真实四行二十三列表格须逐格正确，普通表头不得出现伪上下标。"""
    page, evidence = _real_table()
    result = recover_table_region(page, tuple(evidence["bbox"]))
    assert result is not None
    assert result.source == "vector_grid"
    soup = BeautifulSoup(result.html, "html.parser")
    rows = [[cell.get_text() for cell in row.find_all("td")] for row in soup.find_all("tr")]
    assert rows == evidence["expected_rows"]
    assert soup.find_all(["sub", "sup"]) == []


def _input(page: PDFTablePage, bbox: tuple[float, float, float, float], angle: int = 0) -> NativeTableInput:
    """把公开页证据转换为内部输入，便于核查字符归属与双路径输出。"""
    return NativeTableInput(bbox, page.page_size, angle, tuple(page.geometry.chars), page.drawing_lines, page.rectangles)


def test_datasheet_glyph_ownership_and_adapter_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实字形归属唯一，快速物化和适配路径须得到同一顺序与 HTML。"""
    page, evidence = _real_table()
    source = _input(page, tuple(evidence["bbox"]))
    fast_pending = table_text._select_pending_glyphs(source)
    assert fast_pending == table_text._select_pending_glyphs_adapter(source)
    fast = recover_native_pdf_table(source)
    assert fast is not None
    indices = [index for cell in fast.cells for index in cell.source_char_indices]
    assert len(indices) == len(set(indices)) == len(fast.text.glyphs)
    assert set(indices) == {glyph.source_index for glyph in fast.text.glyphs}
    fast_html = recover_table_region(page, tuple(evidence["bbox"])).html
    monkeypatch.setattr(table_text, "_select_pending_glyphs", table_text._select_pending_glyphs_adapter)
    assert recover_native_pdf_table(source) == fast
    assert recover_table_region(page, tuple(evidence["bbox"])).html == fast_html


def _point_from_upright(point: tuple[float, float], size: tuple[float, float], angle: int) -> tuple[float, float]:
    """把合成正向基线逆变换到页面，独立于待测表格正向化实现。"""
    x, y = point
    width, height = size
    return {0: (x, y), 90: (width - y, x), 180: (width - x, height - y), 270: (y, height - x)}[angle]


def _synthetic_table(
    *,
    table_angle: int = 0,
    cell_angle: int = 90,
    font_size: float = 10,
    column_width: float = 150,
    offset: tuple[float, float] = (40, 50),
    label: str = "Core",
    shuffle: bool = False,
    merged: bool = False,
    script_text: str = "A2",
) -> tuple[PDFTablePage, NativeTableInput, list[list[str]]]:
    """构造物理网格、独立字形方向及真实上下标，避免从候选输出生成期望。"""
    width, height = column_width * 3, 300.0
    page_size = (max(width, height) + 200, max(width, height) + 200)
    table_bbox = (offset[0], offset[1], offset[0] + width, offset[1] + height)
    chars = []

    def add_text(row: int, col: int, lines: list[str], angle: int, *, scripts: bool = False) -> None:
        """逐字生成墨迹框、排版框和原点，词间空白仅由几何表示。"""
        cell_size = (column_width * (2 if merged and row == 0 and col == 1 else 1), 150.0)
        for line_index, content in enumerate(lines):
            x, baseline = 12.0, 35.0 + line_index * 18
            for position, char in enumerate(content):
                if char == " ":
                    x += 0.35 * font_size
                    continue
                is_script = scripts and position == 1
                size = font_size * (0.6 if is_script else 1)
                y = baseline + (-0.4 * font_size if script_text == "A2" else 0.2 * font_size) if is_script else baseline
                advance = size * 0.5
                ink = (x + 0.2, y - 0.8 * size, x + advance - 0.2, y)
                loose = (x, y - 0.85 * size, x + advance, y + 0.15 * size)
                origin = (x, y)
                boxes = []
                for box in (ink, loose):
                    local = _rotate_bbox_from_upright(box, cell_size, angle)
                    absolute = (
                        local[0] + offset[0] + col * column_width,
                        local[1] + offset[1] + row * 150,
                        local[2] + offset[0] + col * column_width,
                        local[3] + offset[1] + row * 150,
                    )
                    boxes.append(_rotate_bbox_from_upright(absolute, page_size, table_angle))
                ox, oy = _point_from_upright(origin, cell_size, angle)
                origin = _point_from_upright(
                    (ox + offset[0] + col * column_width, oy + offset[1] + row * 150),
                    page_size,
                    table_angle,
                )
                chars.append(
                    {
                        "char": char,
                        "char_idx": len(chars),
                        "bbox": Bbox(list(boxes[0])),
                        "tight_bbox": boxes[0],
                        "loose_bbox": boxes[1],
                        "origin": origin,
                        "writing_angle": math.radians((angle + table_angle) % 360),
                        "font": {"name": "IndependentTestFont", "size": size},
                    }
                )
                x += advance

    add_text(0, 0, ["Part No."], 0)
    add_text(0, 1, [label], cell_angle)
    if not merged:
        add_text(0, 2, ["Operating", "voltage (V)"], cell_angle)
    add_text(1, 0, ["BAT32G135B"], 0)
    add_text(1, 1, ["M0+"], cell_angle)
    add_text(1, 2, [script_text], cell_angle, scripts=True)
    if shuffle:
        random.Random(2610).shuffle(chars)
        for index, char in enumerate(chars):
            char["char_idx"] = index
    rules = []
    for axis, coordinates in (("horizontal", [0, 150, 300]), ("vertical", [0, column_width, 2 * column_width, width])):
        for value in coordinates:
            box = (
                (0, value, width, value + 0.1)
                if axis == "horizontal"
                else (
                    value,
                    150 if merged and value == 2 * column_width else 0,
                    value + 0.1,
                    height,
                )
            )
            absolute = (box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1])
            box = _rotate_bbox_from_upright(absolute, page_size, table_angle)
            orientation = axis if table_angle in {0, 180} else {"horizontal": "vertical", "vertical": "horizontal"}[axis]
            rules.append(NativeTableRule(box, 0.1, orientation))
    geometry = PDFPageTextGeometry(
        chars=chars,
        tight_bboxes={char["char_idx"]: char["tight_bbox"] for char in chars},
        origins={char["char_idx"]: char["origin"] for char in chars},
    )
    page = PDFTablePage(page_size, geometry, tuple(rules))
    source = _input(page, _rotate_bbox_from_upright(table_bbox, page_size, table_angle), table_angle)
    expected = [["Part No.", label] + ([] if merged else ["Operating voltage (V)"]), ["BAT32G135B", "M0+", script_text]]
    return page, source, expected


@pytest.mark.parametrize("table_angle", [0, 90, 180, 270])
@pytest.mark.parametrize("cell_angle", [0, 90, 180, 270])
@pytest.mark.parametrize("script_text", ["A2", "H2O"])
def test_cell_direction_composes_with_whole_table_rotation(table_angle: int, cell_angle: int, script_text: str) -> None:
    """旋转单元格和整表叠加后，文字、词界和真实上下标均保持同一语义。"""
    page, source, expected = _synthetic_table(table_angle=table_angle, cell_angle=cell_angle, script_text=script_text)
    result = recover_table_region(page, source.table_bbox, angle=table_angle)
    assert result is not None
    soup = BeautifulSoup(result.html, "html.parser")
    assert [[cell.get_text() for cell in row.find_all("td")] for row in soup.find_all("tr")] == expected
    assert [tag.get_text() for tag in soup.find_all("sup")] == (["2"] if script_text == "A2" else [])
    assert [tag.get_text() for tag in soup.find_all("sub")] == (["2"] if script_text == "H2O" else [])


@pytest.mark.parametrize(
    "options",
    [
        {"font_size": 8},
        {"font_size": 12},
        {"column_width": 180},
        {"offset": (67, 83)},
        {"label": "GPIO"},
        {"label": "Main frequency"},
        {"shuffle": True},
        {"merged": True},
    ],
)
def test_rotated_header_generalizes_independent_layout_changes(options: dict) -> None:
    """独立改变字号、栏宽、位置、原文和写入顺序，并检查合并单元格归属。"""
    page, source, expected = _synthetic_table(**options)
    before = repr(page)
    result = recover_table_region(page, source.table_bbox)
    assert result is not None
    soup = BeautifulSoup(result.html, "html.parser")
    assert [[cell.get_text() for cell in row.find_all("td")] for row in soup.find_all("tr")] == expected
    assert repr(page) == before
    if options.get("merged"):
        assert soup.find_all("tr")[0].find_all("td")[1]["colspan"] == "2"


@pytest.mark.parametrize("angle", [math.radians(37), math.radians(180), float("nan")])
def test_ambiguous_direction_does_not_accept_native_candidate(angle: float) -> None:
    """一个单元格中的非标准或真正混合方向必须拒绝，不用多数方向强行拼接。"""
    _page, source, _expected = _synthetic_table(cell_angle=0)
    chars = [dict(char) for char in source.chars]
    header = [char for char in chars if char["char"] in {"C", "o", "r", "e"} and char["bbox"][1] < 100]
    for char in header[:2]:
        char["writing_angle"] = angle
        # 不同基线及墨迹方向阻止孤立字形纠偏规则掩盖真实混合方向。
        char["origin"] = (char["origin"][0], char["bbox"][1] - 12)
    assert header
    assert recover_native_pdf_table(replace(source, chars=tuple(chars))) is None


@pytest.mark.parametrize("table_angle", [0, 90, 180, 270])
@pytest.mark.parametrize("direction", [None, 0.0, math.pi / 2, math.pi, -math.pi / 2, 0.5, float("nan"), "bad"])
def test_pending_direction_fast_adapter_parity(table_angle: int, direction: object) -> None:
    """方向缺失、标准角、无效角以及定制输入均保持快速物化与适配行为一致。"""
    char = {
        "char": "A",
        "char_idx": 2,
        "bbox": Bbox([10, 10, 20, 20]),
        "loose_bbox": (9, 9, 21, 21),
        "origin": (10, 20),
        "writing_angle": direction,
    }
    source = NativeTableInput((0, 0, 50, 50), (50, 50), table_angle, (char,))
    assert table_text._select_pending_glyphs(source) == table_text._select_pending_glyphs_adapter(source)


@pytest.mark.parametrize("origin", [("10", "20"), ("bad", 20), (float("nan"), 20), (10,), None])
def test_pending_origin_adapter_preserves_validation(origin: object) -> None:
    """非普通数值原点转交适配校验，不能让快速路径悄悄丢失可用基线。"""
    char = {"char": "A", "bbox": (10, 10, 20, 20), "origin": origin, "writing_angle": math.pi / 2}
    source = NativeTableInput((0, 0, 50, 50), (50, 50), 0, (char,))
    assert table_text._select_pending_glyphs(source) == table_text._select_pending_glyphs_adapter(source)


@pytest.mark.parametrize("angle", [90, 180, 270])
@pytest.mark.parametrize("script_text", ["A2", "H2O"])
def test_pdf_single_character_objects_preserve_rotated_scripts(angle: int, script_text: str) -> None:
    """从真实 PDF 提取独立字符对象，验证没有 advance 方向时仍能按字体基线恢复。"""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    payload = BytesIO()
    pdf = canvas.Canvas(payload, pagesize=(380, 380))
    for x in (20, 130, 240, 350):
        pdf.line(x, 30, x, 330)
    for y in (30, 130, 330):
        pdf.line(20, y, 350, y)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, 230, "Part No.")
    pdf.drawString(30, 70, "Chip-1")
    for x, y, value in ((180, 230, "Core"), (290, 230, "GPIO"), (180, 80, "M0+"), (290, 80, script_text)):
        pdf.saveState()
        pdf.translate(x, y)
        pdf.rotate(-angle)
        advance = -10.0
        for index, char in enumerate(value):
            script = value == script_text and index == 1
            size = 7 if script else 12
            pdf.setFont("Helvetica", size)
            pdf.drawString(advance, (5 if script_text == "A2" else -2.5) if script else 0, char)
            advance += stringWidth(char, "Helvetica", size)
        pdf.restoreState()
    pdf.save()
    with PDFDocument(payload.getvalue()) as document:
        result = recover_table_region(prepare_table_page(document[0]), (20, 50, 350, 350))
    assert result is not None
    soup = BeautifulSoup(result.html, "html.parser")
    assert [[cell.get_text() for cell in row.find_all("td")] for row in soup.find_all("tr")] == [
        ["Part No.", "Core", "GPIO"],
        ["Chip-1", "M0+", script_text],
    ]
    tag = "sup" if script_text == "A2" else "sub"
    assert [item.get_text() for item in soup.find_all(tag)] == ["2"]


def test_full_catalog_rotated_row_headers_keep_reading_order() -> None:
    """完整重放真实长文档，逐页锁定纵向行标题及合并单元格内的正常阅读顺序。"""
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel

    source = Path(__file__).parents[1] / "fixtures/upstream_068/deep_structure_catalog.pdf"
    with PDFDocument(source.read_bytes()) as document:
        pages = PdfModel().predict(document)
    assert len(pages) == 328
    for page in (36, 53, 54, *range(185, 213)):
        anchor = "Welcome and Introductions" if page in {36, 54} else "Years of Professional Experience in Education"
        tables = [
            block
            for block in pages[page - 1]
            if block["type"] == "table" and anchor in BeautifulSoup(block["content"], "html.parser").get_text(" ")
        ]
        assert len(tables) == 1
        soup = BeautifulSoup(tables[0]["content"], "html.parser")
        texts = [cell.get_text() for cell in soup.find_all("td")]
        expected = ("Day 1", "Day 2") if page in {36, 54} else ("Current Position",)
        for label in expected:
            assert texts.count(label) == 1, (page, label, texts)
