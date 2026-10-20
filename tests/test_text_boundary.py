import pytest

from rapid_doc.utils.text_boundary import merge_text_line_contents, uses_unspaced_script
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text


@pytest.mark.parametrize(("text", "expected"), [
    ("中文段落", True), ("かなの文章", True), ("한국어 문장", False),
    ("English words", False), ("中文 text with more Latin letters", False),
    ("123 ，。", False), ("", False),
])
def test_script_threshold(text, expected):
    assert uses_unspaced_script(text) is expected


@pytest.mark.parametrize(("lines", "expected"), [
    (["中文", "段落"], "中文段落"),
    (["中文 English", "words"], "中文 English words"),
    (["한국어", "문장"], "한국어 문장"),
    (["inter-", "national"], "international"),
    (["self-", "contained"], "self-contained"),
    (["https://example.org/a", "/b", "?q=c"], "https://example.org/a/b?q=c"),
    (["https://a.test", "https://b.test"], "https://a.test https://b.test"),
    (["123–", "456"], "123–456"),
])
def test_pdf_text_joins_actual_line_boundaries(lines, expected):
    block = {"type": "text", "lines": [
        {"spans": [{"type": "text", "content": line}]} for line in lines
    ]}
    assert merge_para_with_text(block) == expected
    assert [line["spans"][0]["content"] for line in block["lines"]] == lines


def test_font_span_boundary_does_not_split_a_word_or_remove_a_hard_hyphen():
    block = {"type": "text", "lines": [{"spans": [
        {"type": "text", "content": "foo-", "bbox": [0, 0, 20, 10]},
        {"type": "text", "content": "bar", "bbox": [20, 0, 30, 10]},
        {"type": "text", "content": "next", "bbox": [34, 0, 50, 10]},
    ]}]}
    assert merge_para_with_text(block) == "foo-bar next"


def test_formula_is_not_rewritten_as_an_english_word():
    block = {"type": "text", "lines": [
        {"spans": [{"type": "inline_equation", "content": "x-y-"}]},
        {"spans": [{"type": "text", "content": "next"}]},
    ]}
    assert merge_para_with_text(block) == "$x-y-$ next"


def test_surrogates_are_removed_in_the_shared_line_rules():
    assert merge_text_line_contents(["a\ud800", "b"]) == "a b"
    assert merge_para_with_text({"type": "text", "lines": [
        {"spans": [{"type": "text", "content": "a\ud800b"}]},
    ]}) == "ab"
