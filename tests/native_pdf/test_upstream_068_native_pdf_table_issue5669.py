"""用少线层级参数表的原件证据验证分区恢复和保守拒绝。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import (
    NativeTableInput,
    NativeTableRectangle,
    NativeTableRule,
    recover_native_pdf_table,
)
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.engine import diagnose_native_pdf_table
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableResult
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.hierarchical import _subset_text, find_nested_table_evidence
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
from test_native_pdf_table import _char_items, _local_to_page_bbox


def issue5669_input(region: str = "hybrid") -> NativeTableInput:
    """加载原页字符和完整绘图证据，表框来自真实产品入口。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/native_pdf_table_issue5669.json").read_text(encoding="utf-8"))
    assert fixture["source_sha256"] == "dfc5383cf68f63b98d155cad55fa8007bcf80c86b007a975a6f378b8091366e1"
    return NativeTableInput(
        table_bbox=tuple(fixture["regions"][region]),
        page_size=tuple(fixture["page_size"]),
        angle=0,
        chars=tuple(fixture["chars"]),
        drawing_lines=tuple(NativeTableRule(**item) for item in fixture["drawing_lines"]),
        rectangles=tuple(NativeTableRectangle(**item) for item in fixture["rectangles"]),
    )


def assert_hierarchical_cells(result: NativeTableResult | None) -> None:
    """按原页逐格核对四条参数及四组组件等级，不用候选输出生成期望。"""
    assert result is not None
    assert (result.rows, result.cols) == (12, 3)
    expected = [
        (0, 0, 1, 1, "测试按钮可用"),
        (0, 1, 1, 2, "可配置"),
        (1, 0, 1, 1, "测试按钮版本"),
        (1, 1, 1, 2, "带锁测试按钮,点动测试按钮"),
        (2, 0, 1, 1, "机械开关位置指示器"),
        (2, 1, 1, 2, "是"),
        (3, 0, 1, 1, "颜色编码"),
        (3, 1, 1, 2, "透明"),
        (4, 0, 8, 1, "UL94阻燃等级组件"),
    ]
    for row, name, rating in [
        (4, "继电器测试按钮", "HB"),
        (6, "继电器状态显示器", "HB"),
        (8, "继电器基板", "V-0"),
        (10, "继电器盖", "V-2"),
    ]:
        expected.extend(
            [(row, 1, 1, 1, "组件"), (row, 2, 1, 1, name), (row + 1, 1, 1, 1, "UL94阻燃等级"), (row + 1, 2, 1, 1, rating)]
        )
    actual = [(c.row, c.col, c.rowspan, c.colspan, "".join(c.content.split())) for c in result.cells]
    assert sorted(actual) == sorted(expected)
    indices = [index for cell in result.cells for index in cell.source_char_indices]
    assert len(indices) == len(set(indices)) == len(result.text.glyphs)


@pytest.mark.parametrize("top", [546.3866195068359, 537.0])
def test_hierarchical_table_keeps_first_row_and_nested_columns(top: float) -> None:
    """模型框裁掉上边线或包含上边线时，都保留独立首行和三列层级关系。"""
    original = issue5669_input()
    table_input = replace(original, table_bbox=(original.table_bbox[0], top, *original.table_bbox[2:]))
    before = deepcopy(table_input)
    assert_hierarchical_cells(recover_native_pdf_table(table_input))
    assert table_input == before


def test_large_flash_region_does_not_publish_merged_nested_columns() -> None:
    """Flash 的整页技术区不能由局部子表证据冒充完整正确网格。"""
    table_input = issue5669_input("flash")
    assert recover_native_pdf_table(table_input) is None
    assert diagnose_native_pdf_table(table_input)["adopted"] is None


def _synthetic_nested_table(*, offset: float = 0.0, value_start: float = 210.0, glyph_height: float = 8.0) -> NativeTableInput:
    """用不同文字和可独立变化的列宽、字号构造相同物理层级，不复制原件字句。"""
    entries = []
    for row in range(4):
        entries.extend(
            [
                (f"Key {row}", (6.0, row * 20 + 4, 48.0, row * 20 + 4 + glyph_height)),
                (f"Value {row}", (110.0, row * 20 + 4, 160.0, row * 20 + 4 + glyph_height)),
            ]
        )
    entries.append(("Parent", (6.0, 84.0, 48.0, 84.0 + glyph_height)))
    for row in range(8):
        entries.extend(
            [
                (f"Item {row}", (110.0, 84 + row * 20, 150.0, 84 + row * 20 + glyph_height)),
                (f"Data {row}", (value_start, 84 + row * 20, value_start + 40, 84 + row * 20 + glyph_height)),
            ]
        )
    rules = [NativeTableRule((0.0, y - 0.1, 300.0, y + 0.1), 0.2, "horizontal") for y in (20, 40, 60, 80, 240)]
    rules.extend(NativeTableRule((100.0, y - 0.1, 300.0, y + 0.1), 0.2, "horizontal") for y in range(100, 240, 20))
    rules.append(NativeTableRule((99.9, 81.0, 100.1, 239.0), 0.2, "vertical"))
    chars = tuple(
        dict(char, bbox=[x + offset for x in char["bbox"]], font={"name": "UnrelatedSyntheticFont", "size": glyph_height})
        for char in _char_items(entries)
    )
    return NativeTableInput(
        (offset, offset, offset + 300, offset + 240),
        (offset + 310, offset + 250),
        0,
        chars,
        tuple(replace(rule, bbox=tuple(x + offset for x in rule.bbox)) for rule in rules),
    )


@pytest.mark.parametrize("offset,value_start,glyph_height", [(0, 210, 8), (30, 210, 8), (0, 240, 8), (0, 210, 11)])
def test_nested_recovery_generalizes_independent_geometry(offset: float, value_start: float, glyph_height: float) -> None:
    """独立移动位置、改变列宽或字号后，层级结构与成员归属保持一致。"""
    result = recover_native_pdf_table(
        _synthetic_nested_table(offset=offset, value_start=value_start, glyph_height=glyph_height)
    )
    assert result is not None
    assert (result.rows, result.cols) == (12, 3)
    assert [(c.row, c.col, c.rowspan, c.content) for c in result.cells if c.rowspan > 1] == [(4, 0, 8, "Parent")]
    for row in range(8):
        assert [c.content for c in result.cells if c.row == row + 4 and c.col > 0] == [f"Item {row}", f"Data {row}"]


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_nested_recovery_uses_upright_coordinates(angle: int) -> None:
    """整表旋转只改变坐标，不改变分区及子表的合并语义。"""
    original = _synthetic_nested_table()
    width, height = (240.0, 300.0) if angle in (90, 270) else (300.0, 240.0)
    table_input = replace(
        original,
        table_bbox=(0.0, 0.0, width, height),
        page_size=(width, height),
        angle=angle,
        chars=tuple(dict(c, bbox=_local_to_page_bbox(c["bbox"], width, height, angle)) for c in original.chars),
        drawing_lines=tuple(
            replace(rule, bbox=_local_to_page_bbox(rule.bbox, width, height, angle)) for rule in original.drawing_lines
        ),
    )
    result = recover_native_pdf_table(table_input)
    assert result is not None
    assert (result.rows, result.cols) == (12, 3)
    assert any(cell.rowspan == 8 and cell.content == "Parent" for cell in result.cells)


@pytest.mark.parametrize("missing_row", [0, 3, 7])
def test_ambiguous_nested_blank_cell_rejects_all_fallback_candidates(missing_row: int) -> None:
    """局部空值使子表无法独立验证时，不允许旧的高分两列候选重新夺回结果。"""
    original = _synthetic_nested_table()
    table_input = replace(
        original,
        chars=tuple(
            c
            for c in original.chars
            if not (c["bbox"][0] >= 210 and 84 + missing_row * 20 <= c["bbox"][1] <= 92 + missing_row * 20)
        ),
    )
    assert recover_native_pdf_table(table_input) is None
    assert diagnose_native_pdf_table(table_input)["first_rejection_gate"] == "nested_cell_undercount"


def test_single_value_multiline_region_is_not_assumed_to_have_child_columns() -> None:
    """带父级标签的普通多行单值区没有重复子列，不触发嵌套表拒绝规则。"""
    original = _synthetic_nested_table()
    table_input = replace(original, chars=tuple(c for c in original.chars if c["bbox"][0] < 210))
    text = build_native_table_text(table_input)
    assert text is not None
    assert find_nested_table_evidence(table_input, text) == ()


def test_missing_hierarchy_boundary_does_not_invent_rowspan() -> None:
    """去除子表上下界后不凭重复文案猜测父级跨行，恢复只能使用其他已验证证据。"""
    original = _synthetic_nested_table()
    table_input = replace(
        original,
        drawing_lines=tuple(
            rule
            for rule in original.drawing_lines
            if not (rule.orientation == "horizontal" and abs((rule.bbox[1] + rule.bbox[3]) / 2 - 80) < 1)
        ),
    )
    text = build_native_table_text(table_input)
    assert text is not None
    assert find_nested_table_evidence(table_input, text) == ()


@pytest.mark.parametrize("writing_angle", [90, 180, 270, None])
def test_hierarchy_does_not_reinterpret_cross_direction_rows(writing_angle: int | None) -> None:
    """同一物理分隔下出现旋转或未定向字符时，不用横向文本行证明嵌套子表。"""
    source = _synthetic_nested_table()
    text = build_native_table_text(source)
    assert text is not None
    assert find_nested_table_evidence(source, text)
    mixed = replace(text, glyphs=(replace(text.glyphs[0], writing_angle=writing_angle), *text.glyphs[1:]))
    assert find_nested_table_evidence(source, mixed) == ()


def test_child_region_translation_preserves_all_character_geometry() -> None:
    """裁取右下子表时，落格框、排版框和基线共同平移，保持输入及来源索引不变。"""
    source = _synthetic_nested_table()
    text = build_native_table_text(source)
    assert text is not None
    text = replace(
        text,
        glyphs=tuple(
            replace(
                glyph,
                advance_bbox=(glyph.bbox[0] - 1, glyph.bbox[1] - 1, glyph.bbox[2] + 1, glyph.bbox[3] + 1),
                origin=(glyph.bbox[0], glyph.bbox[3]),
            )
            for glyph in text.glyphs
        ),
    )
    before = deepcopy(text)
    child = _subset_text(text, (100, 80, 300, 240))
    assert child is not None
    first = child.glyphs[0]
    assert first.text == "I"
    assert first.bbox[:2] == (10, 4)
    assert first.advance_bbox is not None and first.advance_bbox[:2] == (9, 3)
    assert first.origin == (10, 12)
    assert first.source_index in {glyph.source_index for glyph in text.glyphs}
    assert text == before
