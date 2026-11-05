"""Installed candidate merge and complete native PDF host regression."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import argparse
import copy
import gzip
import json
from pathlib import Path

import rapid_doc_rust as sdk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Legacy analyze_native_pdf whole-document diagnostic; separate from the RapidDoc range test",
    )
    parser.add_argument("--catalog-ranges", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with gzip.open(
        root / "tests/fixtures/upstream_068/caption_grid_merge.json.gz", "rt", encoding="utf-8"
    ) as file:
        cases = json.load(file)["cases"]
    for case in cases:
        request = copy.deepcopy(case["request"])
        before = copy.deepcopy(request)
        # Exercise the public SDK codec as well as the shared Rust implementation.
        for candidate in request["candidates"]:
            for annotation in candidate["annotations"]:
                annotation["line_bboxes"] = dict(annotation["line_bboxes"])
        public_before = copy.deepcopy(request)
        actual = sdk.merge_native_table_candidates(**request)
        assert request == public_before, case["label"]
        for candidate in actual["candidates"]:
            for annotation in candidate["annotations"]:
                annotation["line_bboxes"] = list(map(list, annotation["line_bboxes"].items()))
        assert actual == case["expected"], case["label"]
        assert before == case["request"]
    print(f"Installed candidate merge: {len(cases)} reference/ownership cases passed", flush=True)
    report = dict(candidate_cases=len(cases), sdk=sdk.__file__)
    if args.catalog:
        from bs4 import BeautifulSoup

        source = root / "tests/fixtures/upstream_068/deep_structure_catalog.pdf"
        result = sdk.analyze_native_pdf(source)
        pages = result["pages"]
        assert len(pages) == result["source_page_count"] == 328
        checked = []
        for page in (36, 53, 54, *range(185, 213)):
            anchor = (
                "Welcome and Introductions"
                if page in {36, 54}
                else "Years of Professional Experience in Education"
            )
            tables = [
                b
                for b in pages[page - 1]
                if b["type"] == "table"
                and anchor in BeautifulSoup(b["content"], "html.parser").get_text(" ")
            ]
            assert len(tables) == 1, page
            texts = [
                c.get_text()
                for c in BeautifulSoup(tables[0]["content"], "html.parser").find_all("td")
            ]
            for label in ("Day 1", "Day 2") if page in {36, 54} else ("Current Position",):
                assert texts.count(label) == 1, (page, label, texts)
            checked.append(page)
        print(
            f"Installed native host: 328 pages, {len(checked)} upstream rotated-header assertions passed",
            flush=True,
        )
        report.update(pages=328, checked_pages=checked, success=True)
    if args.catalog_ranges:
        from bs4 import BeautifulSoup

        source = root / "tests/fixtures/upstream_068/deep_structure_catalog.pdf"
        engine = sdk.RapidDoc(
            layout_config={"model_type": "docvortex_rule"},
            parse_method="txt",
            formula_enable=False,
            ocr_config={"seal_enable": False},
            checkbox_config={},
        )
        checked = []
        for start, end, targets in (
            (34, 37, (36,)),
            (51, 55, (53, 54)),
            (183, 213, tuple(range(185, 213))),
        ):
            result = engine(source, start_page_id=start, end_page_id=end - 1)
            assert len(result.middle_json["pdf_info"]) == end - start
            for page in targets:
                anchor = (
                    "Welcome and Introductions"
                    if page in {36, 54}
                    else "Years of Professional Experience in Education"
                )
                tables = [
                    b
                    for b in result.content_list_json
                    if b["type"] == "table"
                    and b["page_idx"] == page - 1 - start
                    and anchor in BeautifulSoup(b["table_body"], "html.parser").get_text(" ")
                ]
                assert len(tables) == 1, (page, len(tables))
                texts = [
                    c.get_text()
                    for c in BeautifulSoup(tables[0]["table_body"], "html.parser").find_all("td")
                ]
                for label in ("Day 1", "Day 2") if page in {36, 54} else ("Current Position",):
                    assert texts.count(label) == 1, (page, label, texts)
                checked.append(page)
            print(f"Installed RapidDoc: source pages {start + 1}-{end}, headers passed", flush=True)
        report.update(
            catalog_source_pages=328,
            parsed_pages=37,
            checked_pages=checked,
            entry="RapidDoc with explicit page ranges and neighboring context",
            success=True,
        )
    if args.report:
        args.report.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode())


if __name__ == "__main__":
    main()
