import pytest

from rapid_doc.backend.office.office_magic_model import MagicModel, parse_text_block_spans
from rapid_doc.backend.office.mkcontent.inline_renderer import merge_para_with_text
from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
from rapid_doc.utils.enum_class import MakeMode


def test_structured_styles_link_formula_and_code_survive_rendering():
    spans = parse_text_block_spans([
        {"type": "text", "content": "Bold ", "styles": ["bold"]},
        {"type": "hyperlink", "url": "https://example.org", "content": [
            {"type": "text", "content": "value ", "styles": ["italic"]},
            {"type": "equation_inline", "content": "x-y"},
        ]},
        {"type": "text", "content": " "},
        {"type": "code_inline", "content": "a`b"},
    ])
    result = merge_para_with_text({"type": "text", "lines": [{"spans": spans}]})
    assert "Bold" in result and ("**Bold" in result or "<strong>Bold" in result)
    assert "https://example.org" in result and "$x-y$" in result
    assert "<code>a`b</code>" in result


def test_old_tag_string_protocol_is_rejected():
    with pytest.raises(TypeError, match="structured spans"):
        parse_text_block_spans("<text>old</text>")


def test_unsafe_link_is_flattened_without_losing_its_styled_text():
    spans = parse_text_block_spans([
        {"type": "hyperlink", "url": "javascript:alert(1)", "content": [
            {"type": "text", "content": "safe label", "styles": ["bold"]},
        ]},
    ])
    assert spans == [{"type": "text", "content": "safe label", "style": ["bold"]}]


def test_new_title_and_image_fields_are_not_dropped():
    model = MagicModel([
        {"type": "doc_title", "content": [{"type": "text", "content": "Document"}], "level": 1},
        {"type": "paragraph_title", "content": [{"type": "text", "content": "Section"}], "level": 2, "anchor": "section"},
        {"type": "image", "image_base64": "data:image/png;base64,AA=="},
    ])
    assert [block["level"] for block in model.get_title_blocks()] == [1, 2]
    assert model.get_title_blocks()[1]["anchor"] == "section"
    assert len(model.get_image_blocks()) == 1


def test_native_heading_numbering_keeps_heading_level_without_extra_ancestor():
    blocks = [
        {"type": "doc_title", "level": 1, "content": [{"type": "text", "content": "Report"}]},
        *[{"type": "paragraph_title", "level": level, "is_numbered_style": True,
           "content": [{"type": "text", "content": str(index)}]}
          for index, level in enumerate([2, 3, 2])],
    ]
    titles = result_to_middle_json([blocks], None)["pdf_info"][0]["para_blocks"]
    assert "section_number" not in titles[0]
    assert [item["section_number"] for item in titles[1:]] == ["1", "1.1", "2"]
    assert [item["level"] for item in titles] == [1, 2, 3, 2]
    assert not any("_numbering_depth" in item for item in titles)


def test_exact_native_list_label_survives_markdown():
    block = {"type": "list", "attribute": "ordered", "content": [
        {"type": "text", "list_label": "IV.", "content": [{"type": "text", "content": "Entry"}]},
    ]}
    middle = result_to_middle_json([[block]], None)
    assert middle["pdf_info"][0]["para_blocks"][0]["blocks"][0]["prefix"] == "IV."
    assert "IV." in union_make(middle["pdf_info"], MakeMode.MM_MD, "images")


def test_v2_keeps_formula_inside_hyperlink_and_inline_code():
    from rapid_doc.backend.office.mkcontent.output_builders import merge_para_with_text_v2
    from copy import deepcopy
    block = {"type": "text", "lines": [{"spans": parse_text_block_spans([
        {"type": "hyperlink", "url": "https://example.org", "content": [
            {"type": "equation_inline", "content": "x-y"},
        ]},
        {"type": "code_inline", "content": "a`b"},
    ])}]}
    original = deepcopy(block)
    output = merge_para_with_text_v2(block)
    assert output[0]["children"] == [{"type": "equation_inline", "content": "x-y"}]
    assert output[1] == {"type": "code_inline", "content": "a`b"}
    assert block == original


def test_simple_inline_code_uses_safe_markdown_delimiters():
    block = {"type": "text", "lines": [{"spans": [{"type": "code_inline", "content": "a`b"}]}]}
    assert merge_para_with_text(block) == "``a`b``"
