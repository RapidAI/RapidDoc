"""验证乱码表格拒绝、正常字符保护及真实区域的准入准出边界。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.pdf import PDFTablePage, recover_table_region
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import engine, text as table_text
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import (
    NativeTableCandidate,
    NativeTableInput,
    NativeTableRectangle,
    NativeTableRule,
)
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text_quality import (
    measure_table_input_quality,
    measure_table_text_quality,
)
from rapid_doc._vendor.docvortex_core.document.pdf import PDFPageTextGeometry


def _real_input(*, clean: bool = False) -> NativeTableInput:
    """加载第三页最小区域证据；干净对照仅替换乱码控制字符，保持原始几何。"""
    data = json.loads((Path(__file__).parents[1] / "fixtures/native_pdf_table_text_quality.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == "c2b7ef637fefefbe9a85b5a33681137a41121983a93419d27bb9707bdf3bf4c0"
    assert data["baseline"]["confidence"] == 1.0
    if clean:
        for char in data["chars"]:
            char["char"] = "".join("5" if ord(c) < 32 and c not in "\t\r\n" else c for c in char["char"])
    return NativeTableInput(
        table_bbox=tuple(data["bbox"]),
        page_size=tuple(data["page_size"]),
        angle=0,
        chars=tuple(data["chars"]),
        drawing_lines=tuple(NativeTableRule(**item) for item in data["drawing_lines"]),
        rectangles=tuple(NativeTableRectangle(**item) for item in data["rectangles"]),
    )


@pytest.mark.parametrize(
    ("value", "total", "bad", "rejected"),
    [
        ("", 0, 0, False),
        ("\t\r\n \u00a0\u3000", 0, 0, False),
        ("\x02中文Aﬁ\U00020000\ue000\U000f0000\U001001b0", 10, 0, False),
        ("\ud840\udc00", 1, 0, False),
        ("\ufffd", 1, 1, False),
        ("A" * 39 + "\x08\x18", 41, 2, False),
        ("A" * 38 + "\x08\x18", 40, 2, True),
        ("A" * 37 + "\ufffd\ud800", 39, 2, True),
        ("\x00\x1c\x7f\x85\ufffd\udfff\ud800", 7, 7, True),
    ],
)
def test_text_quality_thresholds_and_unicode(value: str, total: int, bad: int, rejected: bool) -> None:
    """阈值按有效字符计算，正常空白、既有规范化、PUA 与非 BMP 字符保持可用。"""
    for stage in ("input", "output"):
        quality = measure_table_text_quality([value], stage=stage)
        assert (quality.text_chars, quality.abnormal_chars, quality.rejected) == (total, bad, rejected)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_region_members_exclude_generated_and_outside_chars(angle: int) -> None:
    """同一中心点归属用于四种旋转，表外乱码与自动生成字符不能污染准入。"""
    source = NativeTableInput(
        (20, 20, 80, 80),
        (100, 100),
        angle,
        (
            {"char": "AB", "bbox": (30, 30, 40, 40)},
            {"char": "\x08\x18", "bbox": (0, 0, 10, 10)},
            {"char": "\x08\x18", "bbox": (75, 30, 100, 40)},
            {"char": "\x08\x18", "bbox": (30, 30, 40, 40), "is_generated": True},
            {"char": "\r\n", "bbox": (30, 30, 40, 40)},
            {"char": "\x08\x18", "bbox": (30, 30, 30, 40)},
        ),
    )
    original = deepcopy(source)
    quality = measure_table_input_quality(source)
    assert (quality.text_chars, quality.abnormal_chars, quality.rejected) == (2, 0, False)
    assert source == original


@pytest.mark.parametrize("adapter", [False, True])
def test_real_garbled_table_rejects_before_candidate_generation(monkeypatch: pytest.MonkeyPatch, adapter: bool) -> None:
    """真实错映射表必须在结构恢复前拒绝，公开接口返回 None 且不改写输入。"""
    if adapter:
        monkeypatch.setattr(table_text, "_select_pending_glyphs", table_text._select_pending_glyphs_adapter)
    source = _real_input()
    original = deepcopy(source)

    def unexpected_candidate(*args: object, **kwargs: object) -> None:
        """准入失败后不得再尝试恢复结构候选。"""
        pytest.fail("bad text reached candidate generation")

    monkeypatch.setattr(engine, "build_vector_candidates", unexpected_candidate)
    assert engine.recover_native_pdf_table(source) is None
    diagnostic = engine.diagnose_native_pdf_table(source)
    assert diagnostic["first_rejection_gate"] == "text_quality"
    assert diagnostic["generated_candidates"] == []
    assert diagnostic["text_quality"][0]["stage"] == "input"
    assert diagnostic["text_quality"][0]["abnormal_ratio"] > 0.3
    page = PDFTablePage(
        source.page_size, PDFPageTextGeometry(list(source.chars), {}, {}), source.drawing_lines, source.rectangles
    )
    assert recover_table_region(page, source.table_bbox) is None
    assert source == original


def test_clean_table_keeps_native_html_and_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """同几何干净文本继续以原置信度接受，原生快路径与适配路径结果一致。"""
    source = _real_input(clean=True)
    result = engine.recover_native_pdf_table(source)
    assert result is not None and result.confidence == 1.0
    assert result.source == "sparse_hybrid" and (result.rows, result.cols) == (4, 7)
    monkeypatch.setattr(table_text, "_select_pending_glyphs", table_text._select_pending_glyphs_adapter)
    assert engine.recover_native_pdf_table(source) == result


def test_output_quality_rejects_corrupted_high_confidence_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入合格但单元格重建损坏时，准出门必须否决结构高分且诊断不丢失。"""
    select = engine._select_candidate

    def corrupt_cells(candidates: list[NativeTableCandidate]) -> NativeTableCandidate | None:
        """模拟重建产生控制字符，保留真实网格及全部结构评分。"""
        candidate = select(candidates)
        if candidate is None:
            return None
        return replace(candidate, cells=tuple(replace(cell, content="\x08\ufffd") for cell in candidate.cells))

    monkeypatch.setattr(engine, "_select_candidate", corrupt_cells)
    source = _real_input(clean=True)
    assert engine.recover_native_pdf_table(source) is None
    diagnostic = engine.diagnose_native_pdf_table(source)
    assert diagnostic["first_rejection_gate"] == "text_quality"
    assert [item["rejected"] for item in diagnostic["text_quality"]] == [False, True]
    assert diagnostic["text_quality"][1]["stage"] == "output"
    assert diagnostic["adopted"] is None
