"""真实 wheel：旧 raw layout_dets→页面/段落→Markdown/content_list 的完整字段差分。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib, json, sys
from rapid_doc_rust import (
    assemble_model_page,
    assemble_model_pages,
    render_model_pages,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    data = json.loads((root / f"build/migration/{name}.json").read_text())
    for path, sha in data["sources"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
    return data


def equal(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            equal(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


pages = load("legacy-page")
chains = load("legacy-page-chain")
outputs = load("legacy-output")
numbers = load("legacy-page-numbers")
assert len(numbers["cases"]) == 18
assert len(pages["cases"]) == 286 and len(chains["cases"]) == 382 and len(outputs["cases"]) == 726
for c in pages["cases"] + numbers["cases"]:
    before = copy.deepcopy(c["request"])
    actual = assemble_model_page(**c["request"])
    equal(actual, c["expected"], c["name"])
    equal(c["request"], before, "ownership")
assembled = {}
for c in chains["cases"]:
    before = copy.deepcopy(c["requests"])
    actual = assemble_model_pages(c["requests"])
    equal(actual, c["expected"], c["name"])
    equal(c["requests"], before, "ownership")
    assembled[c["name"]] = actual
for c in outputs["cases"]:
    r = c["request"]
    before = copy.deepcopy(r)
    equal(render_model_pages(**r), c["expected"], c["name"])
    equal(r, before, "ownership")
    if c["name"].startswith("model-chain/"):
        name = c["name"][len("model-chain/") :].rsplit("/", 1)[0]
        equal(
            render_model_pages(
                assembled[name], make_mode=r["make_mode"], image_path_prefix=r["image_path_prefix"]
            ),
            c["expected"],
            "full-model-chain/" + name,
        )
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = pages["cases"][::40]
    for result, c in zip(pool.map(lambda c: assemble_model_page(**c["request"]), samples), samples):
        equal(result, c["expected"])


def fails(function, code, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(("missing error", code))


base = dict(category_id=1, poly=[0, 0, 100, 0, 100, 30, 0, 30], score=0.95, original_order=0)
request = dict(model=dict(layout_dets=[base]), page_width=600, page_height=800)
for change, code in [
    (dict(original_order=None), "unsupported"),
    (dict(poly=[0, 0]), "invalid_input"),
    (dict(original_order="0"), "invalid_input"),
    (dict(vl_ocr=True), "unsupported"),
    (dict(html='<img src="data:image/png;base64,eA==">'), "unsupported"),
]:
    r = copy.deepcopy(request)
    r["model"]["layout_dets"][0].update(change)
    fails(assemble_model_page, code, **r)
for change in [
    dict(scale=0),
    dict(page_width=0),
    dict(page_index=-1),
    dict(formula_enabled="true"),
]:
    fails(assemble_model_page, "invalid_input", **(request | change))
r = copy.deepcopy(request)
r["model"]["layout_dets"] *= 1001
fails(assemble_model_page, "resource_limit", **r)
r = copy.deepcopy(request)
r["model"]["metadata"] = "x" * 2_000_001
fails(assemble_model_page, "resource_limit", **r)
r = copy.deepcopy(request)
r["model"]["metadata"] = "\ud800"
fails(assemble_model_page, "invalid_input", **r)
fails(
    assemble_model_pages,
    "invalid_input",
    [request | dict(page_index=0), request | dict(page_index=0)],
)
fails(assemble_model_pages, "resource_limit", [request] * 1001)
for invalid in [
    None,
    [{}],
    [
        dict(
            page_idx=0,
            page_size=[600, 800],
            para_blocks=[dict(type="image", blocks=[dict(type="image_body")])],
        )
    ],
]:
    fails(render_model_pages, "invalid_input", invalid)
algorithm = [
    dict(
        page_idx=0,
        page_size=[600, 800],
        para_blocks=[dict(type="text", original_label="algorithm", lines=[])],
    )
]
assert render_model_pages(algorithm) == ""
fails(render_model_pages, "unsupported", [], make_mode="content_list_v2")
assert capabilities()["legacy_model_page_assembly"] is True
assert not capabilities()["compatibility"]["python_parser"]
print(
    "Installed wheel: 286+18 old page conversions, 382 page→paragraph chains, 726 old Markdown/content_list references; strict types, ownership/concurrency and explicit errors/budgets passed; nonempty Magika chains covered by magika_output_install.py"
)
