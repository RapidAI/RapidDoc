"""整页布局→裁图→表格路由，含独立旧裁图/布局参考和页面几何合同。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, sys, math
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    recognize_page_tables_png,
    recognize_table_auto_png,
    recognize_pdf_page_tables,
    render_pdf_page,
)
from rapid_doc_rust._native import _PageImage

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("page-tables")
cls = load("table-classification")
wired = load("wired-pipeline")
wireless = load("table-match")
image = _PageImage(Path(fixture["crop_input"]).read_bytes())
for case in fixture["crop_cases"]:
    plan = json.loads(image.plan_json(json.dumps(case["regions"]), case["grid"]))
    assert len(plan["tables"]) == len(case["tables"])
    for actual, expected in zip(plan["tables"], case["tables"]):
        assert actual == {k: v for k, v in expected.items() if k != "rgb_sha256"}
        assert (
            hashlib.sha256(image.crop_rgb(actual["bounds"])).hexdigest() == expected["rgb_sha256"]
        )
        decoded = _PageImage(image.crop_png(actual["bounds"]))
        assert (decoded.width, decoded.height) == (actual["width"], actual["height"])
        assert decoded.crop_rgb([0, 0, decoded.width, decoded.height]) == image.crop_rgb(
            actual["bounds"]
        )
with ThreadPoolExecutor(max_workers=3) as pool:
    expected = image.crop_rgb([1, 2, 30, 35])
    assert all(v == expected for v in pool.map(lambda _: image.crop_rgb([1, 2, 30, 35]), range(9)))
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
layout = OnnxSession(runtime, root / fixture["model"], fixture["sha256"])
classifier = OnnxSession(runtime, root / cls["model"], cls["sha256"])
secondary = OnnxSession(runtime, root / cls["qmodel"], cls["qsha256"])
unet = OnnxSession(runtime, root / wired["unet_model"], wired["unet_sha256"])
detector = OnnxSession(runtime, root / wired["det_model"], wired["det_sha256"])
recognizer = OnnxSession(runtime, root / wired["rec_model"], wired["rec_sha256"])
case = next(v for v in wireless["png_cases"] if v["variant"] == "slanet-plus")
slanet = OnnxSession(runtime, root / case["model"], case["sha256"])
table_options = dict(
    classifier=classifier,
    secondary_classifier=secondary,
    wired_model=unet,
    wireless_model=slanet,
    detector=detector,
    recognizer=recognizer,
)


def compare_layout(actual, expected):
    assert len(actual) == len(expected), (actual, expected)
    for a, b in zip(actual, expected):
        assert all(a[k] == b[k] for k in ["label", "cls_id", "order"])
        assert math.isclose(a["score"], b["score"], abs_tol=1e-5)
        assert all(
            math.isclose(x, y, abs_tol=1e-4) for x, y in zip(a["coordinate"], b["coordinate"])
        )


def compare_geometry(result, bounds):
    local = result["recognition"]
    page = result["page_geometry"]
    x, y = bounds[:2]
    assert page["crop_to_page"] == [1, 0, 0, 1, x, y] and page["coordinate_space"] == "page_pixels"
    assert len(page["ocr_lines"]) == len(local["ocr_lines"])
    for p, l in zip(page["ocr_lines"], local["ocr_lines"]):
        assert p["text"] == l["text"] and p["confidence"] == l["confidence"]
        assert p["points"] == [[a + x, b + y] for a, b in l["points"]]
    if local["route"] == "wired":
        polygons = local["cells"]["polygons"]
        assert len(page["matched_cells"]) == len(local["matches"]["cells"])
        for projected, source in zip(page["matched_cells"], local["matches"]["cells"]):
            for p, l in zip(projected, source):
                assert p == dict(l, points=[[a + x, b + y] for a, b in l["points"]])
    else:
        polygons = []
        for b in local["structure"]["cell_bboxes"]:
            polygons.append(
                [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                if len(b) == 4
                else list(map(list, zip(b[::2], b[1::2])))
            )
        assert page["matched_cells"] is None
    assert page["cell_polygons"] == [[[a + x, b + y] for a, b in p] for p in polygons]


saved = {}
total = 0
for case in fixture["cases"]:
    result = recognize_page_tables_png(case["input"], layout_model=layout, **table_options)
    saved[Path(case["input"]).stem] = result
    assert (result["width"], result["height"]) == (case["width"], case["height"])
    compare_layout(result["layout"], case["layout"])
    assert len(result["tables"]) == len(case["tables"])
    assert result["crop_pixels"] == sum(c["width"] * c["height"] for c in case["tables"])
    for actual, expected in zip(result["tables"], case["tables"]):
        total += 1
        assert actual["crop"] == {
            k: v for k, v in expected.items() if k not in ("rgb_sha256", "input")
        }
        direct = recognize_table_auto_png(expected["input"], **table_options)
        assert actual["recognition"] == direct, (
            case["input"],
            actual["recognition"]["html"],
            direct["html"],
        )
        compare_geometry(actual, expected["bounds"])
assert total > 0
empty = dict(classifier=None, wired_model=None, wireless_model=None, detector=None, recognizer=None)
blank = next(v for v in fixture["cases"] if Path(v["input"]).stem == "blank")
assert recognize_page_tables_png(blank["input"], layout_model=layout, **empty)["tables"] == []
nonempty = next(v for v in fixture["cases"] if v["tables"])
for limits in [dict(max_tables=0), dict(max_total_crop_pixels=1)]:
    try:
        recognize_page_tables_png(nonempty["input"], layout_model=layout, **empty, **limits)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("page budget did not fail before table inference")
for limits in [
    dict(crop_grid=0),
    dict(cls_conf_threshold=float("nan")),
    dict(wireless_variant="unknown"),
]:
    try:
        recognize_page_tables_png(blank["input"], layout_model=None, **empty, **limits)
    except RapidDocError as e:
        assert e.code == "invalid_input"
    except ValueError:
        pass
    else:
        raise AssertionError("invalid options ignored on blank page")
rendered = render_pdf_page(fixture["pdf_input"], page_index=1, dpi=72)
pixels = _PageImage(rendered.png)
assert (
    hashlib.sha256(pixels.crop_rgb([0, 0, pixels.width, pixels.height])).hexdigest()
    == fixture["cases"][0]["rgb_sha256"]
)
pdf = recognize_pdf_page_tables(
    fixture["pdf_input"], page_index=1, dpi=72, layout_model=layout, **table_options
)
assert pdf["page_index"] == 1 and pdf["dpi"] == 72
assert {k: v for k, v in pdf.items() if k not in ("page_index", "dpi")} == saved["single"]
pdf = recognize_pdf_page_tables(
    fixture["pdf_input"], page_index=0, dpi=144, layout_model=layout, **empty
)
assert (pdf["width"], pdf["height"]) == (1200, 1800) and not pdf["tables"]
print(
    f"Installed wheel: {len(fixture['crop_cases'])} legacy RGB crop contracts, 3 real page layouts/{total} tables, local→page geometry, 2 PDF page/DPI pipelines, budgets and crop concurrency passed"
)
