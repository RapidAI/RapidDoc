"""非标准 OOXML 包（非惯例 part 路径、ISO Strict、缺 sldSz、AlternateContent 公式）的转换回归测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native import DocxModel, PptxModel
from rapid_doc._vendor.docvortex_core.schema import BlockType
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("fixture", sorted([
    *FIXTURES.joinpath("docx").glob("*.docx"),
    *FIXTURES.joinpath("pptx").glob("*.pptx"),
]))
def test_nonstandard_ooxml_detection_does_not_need_magika(fixture, monkeypatch):
    monkeypatch.setattr("rapid_doc.utils.guess_suffix_or_lang._get_magika", lambda: pytest.fail("Structural detection used Magika"))
    assert guess_suffix_by_bytes(fixture.read_bytes()) == fixture.suffix[1:]
    assert guess_suffix_by_path(fixture) == fixture.suffix[1:]


def _block_text(block: dict) -> str:
    """把 block 的 content（字符串或任意深度嵌套的内联 span）压成纯文本。"""
    content = block.get("content")

    def render(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(filter(None, (render(part) for part in value)))
        if isinstance(value, dict):
            return render(value.get("content"))
        return ""

    return render(content)


def _flatten_pages(pages: list[list[dict]]) -> list[dict]:
    """展开分页 block 列表并过滤空内容。"""
    return [block for page in pages for block in page]


def _convert(model, fixture: Path) -> str:
    """转换 fixture 并返回全部文本内容，供子串断言。"""
    pages = model().predict(BytesIO(fixture.read_bytes()))
    blocks = _flatten_pages(pages)
    return "\n".join(filter(None, (_block_text(block) for block in blocks)))


ROBUST_CASES = [
    (
        "handmade-altpath.docx",
        DocxModel,
        ["Relocated heading", "Body found via rels"],
    ),
    (
        "handmade-strict.docx",
        DocxModel,
        ["Strict heading", "toggled bold"],
    ),
    (
        "handmade-altpath.pptx",
        PptxModel,
        ["Relocated deck title"],
    ),
    (
        "handmade-strict.pptx",
        PptxModel,
        ["Strict slide title", "Strict body text"],
    ),
    (
        "handmade-order.pptx",
        PptxModel,
        ["Kicker before the title", "Title placed second", "Body after the title"],
    ),
    (
        "handmade-inherit.pptx",
        PptxModel,
        ["Inherited Title Slide", "Roman three bold via master", "Plain closing line"],
    ),
    (
        "handmade-links.pptx",
        PptxModel,
        ["Jump to the second slide", "External link", "Second slide content"],
    ),
    (
        "handmade-math.pptx",
        PptxModel,
        ["Quadratic formula", "Solve with"],
    ),
]


@pytest.mark.parametrize(
    ("filename", "model", "expected_fragments"),
    ROBUST_CASES,
    ids=[case[0] for case in ROBUST_CASES],
)
def test_robust_ooxml_fixture_converts(model, filename: str, expected_fragments: list[str]) -> None:
    """非标准包应完整转换并保留关键内容，而不是在入口或解析阶段失败。"""
    text = _convert(model, FIXTURES / ("docx" if filename.endswith(".docx") else "pptx") / filename)
    for fragment in expected_fragments:
        assert fragment in text, f"{fragment!r} missing from {filename} output:\n{text}"


def test_handmade_math_pptx_outputs_alternate_content_formula() -> None:
    """AlternateContent Choice 分支中的 OMML 公式应转为 LaTeX 输出。"""
    pages = PptxModel().predict(BytesIO((FIXTURES / "pptx" / "handmade-math.pptx").read_bytes()))
    equation_text = "\n".join(
        _block_text(block)
        for block in _flatten_pages(pages)
        if block.get("type") in {BlockType.EQUATION, BlockType.TEXT}
    )
    assert "frac" in equation_text
    assert "2a" in equation_text
