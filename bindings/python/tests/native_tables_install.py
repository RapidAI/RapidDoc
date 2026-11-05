"""已安装 wheel：原生表格优先、物理轨道/合并格/源文字，以及真实模型回退。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import hashlib, json, sys
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    parse_pdf_text,
    render_pdf_page,
    recognize_pdf_page_hybrid,
    recognize_table_auto_png,
    capabilities,
)
from rapid_doc_rust._native import _NativeTablePage, _PageImage

root = Path(sys.argv[1]).resolve()
# 此套合同专测原生表格实验策略，明确开启；模型默认策略由 compatibility_install 单独验证。
_hybrid = recognize_pdf_page_hybrid


def recognize_pdf_page_hybrid(*args, **options):
    options.setdefault("prefer_native_tables", True)
    return _hybrid(*args, **options)


load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("native-tables")


class FixedLayout:
    def __init__(self, regions):
        self.regions = regions
        self.calls = 0

    def layout_png(self, *args, **kwargs):
        self.calls += 1
        return self.regions


class SentinelClassifier:
    def __init__(self):
        self.calls = 0

    def classify_table_png(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("model-fallback-sentinel")


sentinel = SentinelClassifier()
saved = {}
fact_cache = {}


def close(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) < 0.003 for a, b in zip(actual, expected)), (actual, expected)


for case in fixture["cases"]:
    index = case["page_index"]
    if index not in fact_cache:
        fact_cache[index] = parse_pdf_text(
            fixture["pdf_input"], start_page=index, end_page=index + 1, allow_partial=True
        ).pages[0]
    facts = fact_cache[index]
    frame = _NativeTablePage(json.dumps(facts), case["width"], case["height"])
    proposals = [dict(layout_index=i, bounds=r["coordinate"]) for i, r in enumerate(case["layout"])]
    plan = json.loads(frame.plan_json(json.dumps(proposals)))
    assert [s["reason"] for s in plan["selections"]] == case.get(
        "reasons", [case["reason"]] * len(proposals)
    ), (case["name"], plan)
    rendered = render_pdf_page(fixture["pdf_input"], page_index=index, dpi=case["dpi"])
    image = _PageImage(rendered.png)
    assert (
        hashlib.sha256(image.crop_rgb([0, 0, image.width, image.height])).hexdigest()
        == case["rgb_sha256"]
    )
    layout = FixedLayout(case["layout"])
    if case["reason"] != "closed_native_grid":
        before = sentinel.calls
        try:
            recognize_pdf_page_hybrid(
                fixture["pdf_input"],
                experimental=True,
                layout_model=layout,
                detector=object(),
                recognizer=object(),
                classifier=sentinel,
                page_index=index,
                dpi=case["dpi"],
            )
        except RuntimeError as e:
            assert str(e) == "model-fallback-sentinel"
        else:
            raise AssertionError("failed native candidate did not run model classifier")
        assert sentinel.calls == before + 1
        continue
    # 纯原生表格无需检测/识别/分类或结构会话；即使传入会失败的分类器也不调用它。
    before = sentinel.calls
    result = recognize_pdf_page_hybrid(
        fixture["pdf_input"],
        experimental=True,
        layout_model=layout,
        classifier=sentinel,
        page_index=index,
        dpi=case["dpi"],
    )
    saved[index, case["dpi"]] = result
    assert sentinel.calls == before and layout.calls == 1 and not result["native_routes"]
    assert result["native_work"] == 0 and result["native_table_work"] == plan["work"]
    assert len(result["blocks"]) == len(case["tables"])
    source = fixture["facts"]["pages"][index]["chars"]
    for block, expected in zip(result["blocks"], case["tables"]):
        native = block["recognition"]
        table = native["table"]
        geometry = block["page_geometry"]
        assert block["table_selection"] == dict(method="native", reason="closed_native_grid")
        assert native["route"] == "native" and native["coordinate_space"] == "visual_points"
        assert geometry["coordinate_space"] == "page_pixels" and block["lines"] == []
        assert (table["rows"], table["columns"]) == (2, 2)
        assert len(table["cells"]) == 3
        close(native["x_tracks"], expected["x_tracks"])
        close(native["y_tracks"], expected["y_tracks"])
        close(table["bounds"], expected["bounds"])
        used = []
        sx = case["width"] / 600
        sy = case["height"] / 500
        close(geometry["point_to_page"], [sx, 0, 0, sy, 0, 0])
        for cell, golden, polygon in zip(
            table["cells"], expected["cells"], geometry["cell_polygons"]
        ):
            assert {k: cell[k] for k in golden} == golden, (cell, golden)
            assert cell["text"] == "".join(source[i]["char"] for i in cell["character_indices"])
            used.extend(cell["character_indices"])
            x0 = expected["x_tracks"][cell["column"]]
            x1 = expected["x_tracks"][cell["column"] + cell["column_span"]]
            y0 = expected["y_tracks"][cell["row"]]
            y1 = expected["y_tracks"][cell["row"] + cell["row_span"]]
            for p, q in zip(
                polygon,
                [[x0 * sx, y0 * sy], [x1 * sx, y0 * sy], [x1 * sx, y1 * sy], [x0 * sx, y1 * sy]],
            ):
                close(p, q)
        assert len(used) == len(set(used))
        assert block["text"] == "\n".join(c["text"] for c in table["cells"])
        assert '<td colspan="2">' in block["html"]
        if index == 0:
            assert "Quarter &lt;&amp;&gt; 123" in block["html"] and "<&>" not in block["html"]

first = fixture["cases"][0]
multiple = next(c for c in fixture["cases"] if c["name"] == "multiple" and c["dpi"] == 72)
for case, work in [
    (first, saved[0, 72]["native_table_work"]),
    (multiple, saved[6, 72]["native_table_work"]),
]:
    try:
        recognize_pdf_page_hybrid(
            fixture["pdf_input"],
            experimental=True,
            layout_model=FixedLayout(case["layout"]),
            page_index=case["page_index"],
            dpi=72,
            max_native_table_work=work - 1,
        )
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("native table work budget reset or swallowed")
for options in [
    dict(native_table_margin_pt=float("nan")),
    dict(native_table_margin_pt=33),
    dict(max_native_table_work=-1),
    dict(prefer_native_tables="yes"),
]:
    try:
        recognize_pdf_page_hybrid(b"bad PDF", experimental=True, layout_model=None, **options)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid native table configuration accepted")
for options in [dict(prefer_native_tables=False), dict(native_table_margin_pt=0)]:
    try:
        recognize_pdf_page_hybrid(
            fixture["pdf_input"],
            experimental=True,
            layout_model=FixedLayout(first["layout"]),
            detector=object(),
            recognizer=object(),
            classifier=sentinel,
            dpi=72,
            **options,
        )
    except RuntimeError as e:
        assert str(e) == "model-fallback-sentinel"
    else:
        raise AssertionError("explicit model path or strict margin did not run classifier")

runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
wired = load("wired-pipeline")
cls = load("table-classification")
wireless = load("table-match")
layout_info = load("page-tables")
model = lambda path, sha: OnnxSession(runtime, root / path, sha)
detector = model(wired["det_model"], wired["det_sha256"])
recognizer = model(wired["rec_model"], wired["rec_sha256"])
classifier = model(cls["model"], cls["sha256"])
secondary = model(cls["qmodel"], cls["qsha256"])
unet = model(wired["unet_model"], wired["unet_sha256"])
slanet_info = next(c for c in wireless["png_cases"] if c["variant"] == "slanet-plus")
slanet = model(slanet_info["model"], slanet_info["sha256"])
options = dict(
    detector=detector,
    recognizer=recognizer,
    classifier=classifier,
    secondary_classifier=secondary,
    wired_model=unet,
    wireless_model=slanet,
)
scan = next(c for c in fixture["cases"] if c["name"] == "scan")
result = recognize_pdf_page_hybrid(
    fixture["pdf_input"],
    experimental=True,
    layout_model=FixedLayout(scan["layout"]),
    page_index=1,
    dpi=72,
    **options,
)
assert result["blocks"][0]["table_selection"] == dict(method="model", reason="raster_overlap")
assert result["blocks"][0]["recognition"] == recognize_table_auto_png(
    fixture["scan_crop"], **options
)
mixed = next(c for c in fixture["cases"] if c["name"] == "native-and-scan")
combined = recognize_pdf_page_hybrid(
    fixture["pdf_input"],
    experimental=True,
    layout_model=FixedLayout(mixed["layout"]),
    page_index=7,
    dpi=72,
    **options,
)
assert [b["table_selection"]["method"] for b in combined["blocks"]] == ["native", "model"]
assert combined["blocks"][0]["text"] == "T3 header\nL3\nR3"
assert combined["blocks"][1]["recognition"] == recognize_table_auto_png(
    fixture["mixed_scan_crop"], **options
)
layout = model(layout_info["model"], layout_info["sha256"])
real = recognize_pdf_page_hybrid(
    fixture["pdf_input"], experimental=True, layout_model=layout, dpi=72, **options
)
assert any(
    b["kind"] == "table" and b["recognition"]["route"] == "native" for b in real["blocks"]
), real["native_table_routes"]
assert capabilities()["pdf_native_table_selection"]
print(
    "Installed wheel: 10 page/DPI native-selection contracts; merged/unequal tracks and source text, 4 cases/6 native tables without inference models, explicit fallback/error propagation, budgets, same-page native/model coexistence, real raster model equivalence and real layout→native table passed"
)
