"""Installed pure Rust fixed-font runtime: old classification, real CJK geometry and pixels."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def read(name):
    data = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in data["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return data


def rgba(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    at = 8
    compressed = b""
    while at < len(png):
        n = struct.unpack_from(">I", png, at)[0]
        kind = png[at + 4 : at + 8]
        data = png[at + 8 : at + 8 + n]
        at += 12 + n
        if kind == b"IHDR":
            width, height, bits, color, *rest = struct.unpack(">IIBBBBB", data)
            assert bits == 8 and color in [2, 6] and rest == [0, 0, 0]
        if kind == b"IDAT":
            compressed += data
    bpp = 4 if color == 6 else 3
    stride = width * bpp
    data = zlib.decompress(compressed)
    out = bytearray()
    prev = bytearray(stride)
    for row in range(height):
        at = row * (stride + 1)
        f = data[at]
        cur = bytearray(data[at + 1 : at + 1 + stride])
        for x in range(stride):
            a = cur[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 0:
                v = 0
            elif f == 1:
                v = a
            elif f == 2:
                v = b
            elif f == 3:
                v = (a + b) // 2
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                v = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else:
                raise AssertionError(f)
            cur[x] = (cur[x] + v) & 255
        if bpp == 4:
            out.extend(cur)
        else:
            for x in range(0, stride, 3):
                out.extend(cur[x : x + 3])
                out.append(255)
        prev = cur
    return bytes(out)


with ThreadPoolExecutor(max_workers=4) as pool:
    infos = list(pool.map(lambda _: sdk.pdfium_font_runtime_info(), range(16)))
assert all(i == infos[0] for i in infos)
assert infos[0]["font_policy"] == "droid-cjk-v1" and not infos[0]["released"]
assert infos[0]["font_sha256"] == "8a4dea0899424438af25a6f1f6eb61e5d111d85367eece0a5d30170861ae6b2e"
facts = read("native-fixed-font-runtime")


def classify(c):
    same(
        sdk.classify_pdf_font_name(bytes(c["face"]), charset=c["charset"]),
        c["expected"],
        "font classification",
    )


assert len(facts["cases"]) == 4771
for c in facts["cases"]:
    classify(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(classify, facts["cases"][::41]))
pdf = read("pdf-fixed-font-runtime")["cases"]
assert len(pdf) == 8


def geometry(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    same(
        sdk.read_pdf_character_primitives(data, page_index=c["page_index"]),
        c["expected"],
        c["label"],
    )
    return data


for c in pdf:
    data = geometry(c)
    for r in c["renders"]:
        for mode in ["RGBA", "RGB"]:
            page = sdk.render_pdf_page(
                data, page_index=c["page_index"], dpi=r["dpi"], color_mode=mode
            )
            assert (page.width, page.height) == (r["width"], r["height"])
            assert hashlib.sha256(rgba(page.png)).hexdigest() == r["rgba_sha256"], (
                c["label"],
                mode,
                r["dpi"],
            )
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(geometry, pdf))
cid = next(c for c in read("pdf-numeric-candidates")["cases"] if c["label"] == "metric_pair")
data = (root / cid["pdf"]).read_bytes()
assert hashlib.sha256(data).hexdigest() == cid["pdf_sha256"]
g = sdk.read_pdf_native_text_geometry(data, page_index=cid["page_index"])
v = sdk.read_pdf_vector_geometry(data, page_index=cid["page_index"])
ls = sdk.build_native_visual_lines(
    g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
)
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()
s = dict(
    page_size=g["page_size"],
    lines=[
        {k: l.get(k, False if k == "compact_formula_cluster" else None) for k in fields} for l in ls
    ],
    chars=g["chars"],
    drawing_lines=v["drawing_lines"],
    path_infos=v["path_infos"],
    form_bboxes=[],
    image_bboxes=[],
    signature_bboxes=[],
    page_index=cid["page_index"],
)
same(s, cid["request"]["source"], "previously failing CID source")


def decoded(v):
    v = copy.deepcopy(v)
    for c in v["candidates"]:
        for a in c["annotations"]:
            a["line_bboxes"] = {int(k): b for k, b in a["line_bboxes"]}
    return v


same(
    sdk.detect_native_supplemental_tables(s, kind=cid["kind"]),
    decoded(cid["expected"]["individual"]),
    "CID candidate",
)
same(
    sdk.detect_native_structural_table_candidates(s),
    decoded(cid["expected"]["page"]),
    "CID combined page",
)
assert sdk.pdfium_font_runtime_info()["bundled_requests"] > 0
try:
    sdk.read_pdf_character_primitives(b"broken PDF")
except sdk.RapidDocError as e:
    assert e.code == "corrupt_document"
else:
    raise AssertionError("corrupt PDF accepted")
geometry(pdf[0])
try:
    sdk.classify_pdf_font_name(b"x" * 4097)
except ValueError:
    pass
else:
    raise AssertionError("font name limit")
assert (
    sdk.capabilities()["native_fixed_cjk_font_runtime"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
for name in ["MIGRATION-NOTICES.txt", "PYTHON-CODECS-LICENSE.txt"]:
    assert (Path(sdk.__file__).parent / "licenses" / name).is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "4771 old font classifications; 8 live four-language/two-orientation CID pages, 32 RGB/RGBA pixel comparisons; former CID geometry/candidate failure fixed; concurrent init/reads, ownership, limits/error recovery and pinned identity passed"
)
