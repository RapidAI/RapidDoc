"""Installed contiguous original final-page flow through member/caption/local-list rebuilding."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b), a, b)
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def invoke(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    try:
        if kind == "page":
            out = sdk.prepare_native_final_page_blocks(
                q["page"],
                page_index=q.get("page_index", 0),
                body_profile=q.get("document_body_profile"),
                canonical_body_profile=q.get("canonical_body_profile"),
                title_profile=q.get("document_title_profile"),
                refine_blocks=True,
            )
        elif kind == "document":
            out = sdk.prepare_native_final_document_blocks(q["pages"], refine_blocks=True)
        else:
            out = json.loads(
                sdk.legacy_pipeline_json(
                    "refine_native_page_blocks", json.dumps(q, allow_nan=False)
                )
            )
    except sdk.RapidDocError as e:
        assert "error" in c, (c["label"], str(e))
        assert e.code in ["invalid_input", "resource_limit"]
        return
    assert "error" not in c, c["label"]
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


pages = load("native-final-refined-blocks-paths")["cases"]
chains = load("native-final-refined-blocks-chain-paths")["cases"]
for c in pages:
    invoke(c, "page")
for c in chains:
    invoke(c, "document")
count = 0
for name in [
    "native-page-block-rules",
    "native-caption-block-rules-canonical",
    "native-page-block-contracts",
]:
    for c in load(name)["cases"]:
        invoke(c, "rules")
        count += 1
for c in pages[::29]:
    if "expected" not in c:
        continue
    q = c["request"]
    expected = {k: v for k, v in c["expected"].items() if k != "refined_text_blocks"}
    same(
        sdk.prepare_native_final_page_blocks(
            q["page"],
            page_index=q.get("page_index", 0),
            body_profile=q.get("document_body_profile"),
            canonical_body_profile=q.get("canonical_body_profile"),
            title_profile=q.get("document_title_profile"),
        ),
        expected,
        "default-checkpoint",
    )
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: invoke(c, "document"), chains[-3:]))
live = load("native-reference-runtime")
expected = next(c for c in chains if c["label"] == "live-reference-document")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
raw = []
for i in range(3):
    g = sdk.read_pdf_native_text_geometry(data, page_index=i, pdfium_library=library)
    v = sdk.read_pdf_vector_geometry(data, page_index=i, pdfium_library=library)
    forms = sdk.read_pdf_form_geometry(data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    source = dict(
        page_size=g["page_size"],
        page_index=i,
        lines=ls,
        chars=g["chars"],
        drawing_lines=v["drawing_lines"],
        path_infos=v["path_infos"],
        form_infos=forms["form_infos"],
        form_bboxes=forms["form_bboxes"],
        image_bboxes=[],
        signature_bboxes=[],
    )
    geometry = dict(chars=g["chars"])
    for key, value in [("tight_bboxes", "bbox"), ("origins", "origin"), ("loose_bboxes", "bbox")]:
        geometry[key] = [
            dict(char_idx=int(j), **{value: b})
            for j, b in sorted(g[key].items(), key=lambda e: int(e[0]))
        ]
    raw.append(dict(source=source, geometry=geometry))
same(raw, expected["request"]["pages"], "live-source")
same(
    sdk.prepare_native_final_document_blocks(raw, refine_blocks=True),
    expected["expected"],
    "live-refined-blocks",
)
formula = load("native-formula-runtime")
data = (root / formula["pdf"]).read_bytes()
page_cases = {c["label"]: c for c in pages}
for i, c in enumerate(formula["cases"]):
    g = sdk.read_pdf_native_text_geometry(data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    v = sdk.read_pdf_vector_geometry(data, page_index=i, pdfium_library=library)
    old = page_cases["formula/" + c["label"]]
    p = copy.deepcopy(old["request"]["page"])
    same(ls, p["remaining_lines"], "live-formula-lines")
    same(v["drawing_lines"], p["drawing_lines"], "live-formula-rules")
    p["remaining_lines"] = ls
    same(
        sdk.prepare_native_final_page_blocks(
            p, body_profile=old["request"].get("document_body_profile"), refine_blocks=True
        ),
        old["expected"],
        "live-formula-refined",
    )
for mode in ["vector", "letters", "second-page"]:
    assert any(
        b["type"] == "list"
        for b in page_cases["bullet-path/" + mode]["expected"]["refined_text_blocks"]
    ), mode
for value in [None, 1, "yes"]:
    try:
        sdk.prepare_native_final_document_blocks([], refine_blocks=value)
    except ValueError:
        pass
    else:
        raise AssertionError("non-bool refine flag")
for bs in [[None], [dict(type="text", bbox=[1.0, 2.0, 3.0, 4.0])]]:
    try:
        sdk.legacy_pipeline_json(
            "refine_native_page_blocks",
            json.dumps(dict(blocks=bs, page_size=[400.0, 600.0], kind="sequence")),
        )
    except sdk.RapidDocError as e:
        assert e.code in ["invalid_input", "resource_limit"]
    else:
        raise AssertionError("invalid internal block metadata")
assert sdk.capabilities()["native_page_block_refinement"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
assert not any(n.startswith("rapid_doc._vendor") for n in sys.modules)
print(
    f"{len(pages)} contiguous member/caption/list pages, {len(chains)} raw-source chains, {count} rules/contracts; strict types/ownership/concurrency/flag/default/error gates, vector and letter lists, 5 fresh PDFium pages; no Python parser import passed"
)
