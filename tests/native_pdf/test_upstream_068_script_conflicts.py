"""验证 PDF 上下标证据的来源裁决、局部弃权及严格协议边界。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from rapid_doc._vendor.docvortex_core.analyzers.pdf import (
    PDF_NATIVE_SCRIPT_MARKUP_KEY,
    PDFTextEvidence,
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextScriptLine,
    PDFTextScriptRange,
    PDFTextStyleLine,
    PDFTextStyleRange,
    apply_text_evidence,
)
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.materialize import materialize_pdf_inline_spans
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.types import PDFScriptStyle, _PDF_STYLE_INTERVALS_KEY, _RawStyleInterval
from rapid_doc.backend.office.office_magic_model import parse_text_block_spans
from rapid_doc.backend.office.mkcontent.inline_renderer import merge_para_with_text
from rapid_doc._vendor.docvortex_core.schema import ModelJson, TextSpan, parse_inline_spans


def _script_line(text: str, start: int, end: int, style: PDFScriptStyle, source_index: int) -> PDFTextScriptLine:
    """构造各自只含一种脚本方向、在页面上相互分离的合法源行。"""
    top = 10.0 + source_index * 20
    return PDFTextScriptLine(
        bbox=(10.0, top, 90.0, top + 10),
        text=text,
        script_ranges=(PDFTextScriptRange(start, end, style, (30.0, top, 50.0, top + 5), 3, False),),
        source_index=source_index,
        angle=0,
    )


def _apply(
    content: str, lines: list[PDFTextScriptLine], *, native: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """通过共享公开入口物化证据，并返回可与最终文本逐区间核对的诊断。"""
    blocks: list[dict[str, Any]] = [{"type": "text", "bbox": [0.0, 0.0, 1.0, 1.0], "content": content}]
    if native:
        blocks[0][PDF_NATIVE_SCRIPT_MARKUP_KEY] = True
    diagnostics: list[dict[str, Any]] = []
    apply_text_evidence(blocks, PDFTextEvidence((100.0, 100.0), scripts=tuple(lines)), diagnostics=diagnostics)
    return blocks[0]["content"], diagnostics


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("content", "specs", "expected"),
    [
        (
            "new123text",
            [("old123text", 3, 6, "superscript"), ("new123text", 3, 6, "subscript")],
            [("new", []), ("123", ["subscript"]), ("text", [])],
        ),
        ("x2", [("x2", 1, 2, "superscript"), ("x2", 1, 2, "subscript")], [("x2", [])]),
        (
            "a123z",
            [("a12", 1, 3, "superscript"), ("23z", 0, 2, "subscript")],
            [("a", []), ("1", ["superscript"]), ("2", []), ("3", ["subscript"]), ("z", [])],
        ),
        (
            "new123tail",
            [("old123lost", 3, 6, "superscript"), ("new12", 3, 5, "subscript")],
            [("new", []), ("12", ["subscript"]), ("3", ["superscript"]), ("tail", [])],
        ),
        (
            "x12z",
            [("x1", 1, 2, "superscript"), ("2z", 0, 1, "subscript")],
            [("x", []), ("1", ["superscript"]), ("2", ["subscript"]), ("z", [])],
        ),
        (
            "x2",
            [("x2", 1, 2, "superscript"), ("x2", 1, 2, "superscript")],
            [("x", []), ("2", ["superscript"])],
        ),
    ],
    ids=["exact-over-fragment", "ambiguous-source", "partial-overlap", "partial-priority", "adjacent", "same-direction"],
)
def test_source_script_conflicts_are_local_and_order_independent(
    content: str,
    specs: list[tuple[str, int, int, PDFScriptStyle]],
    expected: list[tuple[str, list[str]]],
    reverse: bool,
) -> None:
    """源字符各自合法时，目标区间按证据强弱选择单方向，同级冲突仅在交集弃权。"""
    lines = [_script_line(*spec, source_index=index) for index, spec in enumerate(specs)]
    spans, diagnostics = _apply(content, list(reversed(lines)) if reverse else lines)
    assert [(span.content, span.styles) for span in parse_inline_spans(spans)] == expected
    expected_roles = [roles for text, roles in expected for _ in text]
    for item in diagnostics:
        assert content[item["raw_start"] : item["raw_end"]] == item["text"]
        assert all(item["role"] in roles for roles in expected_roles[item["raw_start"] : item["raw_end"]])
    if content == "x2" and expected == [("x2", [])]:
        assert diagnostics == []
    if content == "new123text":
        assert [(item["source_index"], item["text"]) for item in diagnostics] == [(1, "123")]
    if content == "a123z":
        assert [(item["range_start"], item["range_end"], item["text"]) for item in diagnostics] == [(1, 2, "1"), (1, 2, "3")]


def test_duplicate_projection_paths_do_not_outvote_equal_conflict() -> None:
    """同一上标证据重复进入两条路径或多次输入，都不能压过同级下标证据。"""
    upper = _script_line("x2", 1, 2, "superscript", 0)
    lower = _script_line("x2", 1, 2, "subscript", 1)
    spans, diagnostics = _apply("x2", [upper, deepcopy(upper), lower, upper])
    assert spans == [{"type": "text", "content": "x2"}]
    assert diagnostics == []
    _spans, diagnostics = _apply("x2", [upper, deepcopy(upper)])
    assert len(diagnostics) == 1


@pytest.mark.parametrize(
    ("tag", "native_style", "mapped_style"), [("sup", "superscript", "subscript"), ("sub", "subscript", "superscript")]
)
def test_owned_script_markup_overrides_opposite_projection(
    tag: str, native_style: PDFScriptStyle, mapped_style: PDFScriptStyle
) -> None:
    """有所有权的合法标签保持方向，被覆盖的相反方向不能进入物化诊断。"""
    content = f"x<{tag}>123</{tag}>"
    spans, diagnostics = _apply(content, [_script_line("x123", 1, 4, mapped_style, 0)], native=True)
    assert spans == [{"type": "text", "content": "x"}, {"type": "text", "content": "123", "styles": [native_style]}]
    assert diagnostics == []


@pytest.mark.parametrize("native", [False, True])
def test_unowned_or_malformed_markup_is_not_authoritative(native: bool) -> None:
    """用户原文标签和畸形 detector 标签仍是文字，不会被当成最高优先级证据。"""
    content = "x<sup>123" if native else "x<sup>123</sup>"
    spans, diagnostics = _apply(content, [_script_line("x123", 1, 4, "subscript", 0)], native=native)
    assert "".join(span["content"] for span in spans) == content
    assert next(span for span in spans if span["content"] == "123")["styles"] == ["subscript"]
    assert [item["role"] for item in diagnostics] == ["subscript"]


def test_formula_gap_match_beats_fragment_without_changing_formula_or_links() -> None:
    """跨公式的完整行匹配胜出后，粗体、链接、公式和两种渲染结果保持可用。"""
    content = r"new\(x+1\)123text"
    lines = [_script_line("old123lost", 3, 6, "superscript", 0), _script_line("newx+1123text", 6, 9, "subscript", 1)]
    bbox = (10.0, 10.0, 90.0, 50.0)
    evidence = PDFTextEvidence(
        (100.0, 100.0),
        styles=(PDFTextStyleLine(bbox, "new123text", (PDFTextStyleRange(3, 6, ("bold",)),), 0),),
        links=(PDFTextLinkLine(bbox, "new123text", (PDFTextLinkRange(3, 6, "https://example.test/123"),), 0),),
        scripts=tuple(lines),
    )
    blocks = [{"type": "text", "bbox": [0.0, 0.0, 1.0, 1.0], "content": content}]
    diagnostics: list[dict[str, Any]] = []
    apply_text_evidence(blocks, evidence, diagnostics=diagnostics)
    spans = blocks[0]["content"]
    assert spans[1] == {"type": "equation_inline", "content": "x+1"}
    assert spans[2]["url"] == "https://example.test/123"
    assert spans[2]["content"] == [{"type": "text", "content": "123", "styles": ["bold", "subscript"]}]
    assert [(item["text"], item["range_start"], item["range_end"]) for item in diagnostics] == [("123", 6, 9)]
    # RapidDoc exposes its own mixed Markdown/HTML inline renderer.
    paragraph = {"type": "text", "lines": [{"spans": parse_text_block_spans(spans)}]}
    rendered = merge_para_with_text(paragraph)
    assert "<sub>" in rendered and "123" in rendered
    assert "https://example.test/123" in rendered
    assert "<sup>" not in rendered



def test_dehyphenated_line_match_beats_fragment() -> None:
    """断词恢复的完整源行优先于同文本片段回退，诊断中不包含已删除连字符。"""
    lines = [
        _script_line("oldinterlost", 3, 8, "superscript", 0),
        _script_line("inter-", 0, 6, "subscript", 1),
        _script_line("national", 0, 8, "subscript", 2),
    ]
    spans, diagnostics = _apply("international", lines)
    assert spans == [{"type": "text", "content": "international", "styles": ["subscript"]}]
    assert [(item["text"], item["range_end"]) for item in diagnostics] == [("inter", 5), ("national", 8)]


def test_final_guard_preserves_other_styles_and_strict_schema() -> None:
    """没有匹配来源的双重脚本只丢弃上下标，公开 Schema 仍拒绝非法输入。"""
    blocks = [
        {
            "type": "text",
            "content": "a123z",
            _PDF_STYLE_INTERVALS_KEY: [
                _RawStyleInterval(1, 4, ("bold", "superscript")),
                _RawStyleInterval(2, 3, ("underline", "subscript")),
            ],
        }
    ]
    materialize_pdf_inline_spans(blocks)
    assert blocks[0]["content"][2] == {"type": "text", "content": "2", "styles": ["bold", "underline"]}
    assert blocks[0]["content"][1]["styles"] == ["bold", "superscript"]
    assert blocks[0]["content"][3]["styles"] == ["bold", "superscript"]
    with pytest.raises(ValidationError, match="both superscript and subscript"):
        TextSpan(type="text", content="2", styles=["superscript", "subscript"])


def test_ligature_raw_character_conflict_is_not_reported_as_materialized() -> None:
    """展开连字的不同投影字符映回同一原字符时，最终弃权与诊断保持一致。"""
    spans, diagnostics = _apply("ﬂ", [_script_line("f", 0, 1, "superscript", 0), _script_line("l", 0, 1, "subscript", 1)])
    assert spans == [{"type": "text", "content": "ﬂ"}]
    assert diagnostics == []
