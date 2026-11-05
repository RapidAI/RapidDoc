"""Exercise the consecutive-page UNET regression through the installed public API."""

import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    from rapid_doc_rust import RapidDoc

    engine = RapidDoc(
        layout_config={"model_type": "docvortex_rule"},
        parse_method="txt",
        formula_enable=False,
        ocr_config={"seal_enable": False},
        checkbox_config={},
    )
    result = engine(
        root / "tests/fixtures/upstream_068/deep_structure_catalog.pdf",
        start_page_id=288,
        end_page_id=289,
    )
    assert len(result.middle_json["pdf_info"]) == 2
    assert {row["page_idx"] for row in result.content_list_json} == {0, 1}
    assert "CHEMISTRY" in result.markdown and "PHYSICS" in result.markdown
    assert "Round 3 Panelist Agreement Data" in result.markdown
    assert any(row["type"] == "table" for row in result.content_list_json)
    print("Installed public API: consecutive-page UNET regression passed")


if __name__ == "__main__":
    main()
