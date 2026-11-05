"""独立wheel：完整字符去重、实际PDF读取及实时PDF原语串接旧表格引擎。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    deduplicate_native_characters as dedup,
    read_pdf_native_text_geometry as read,
    read_pdf_vector_geometry,
    evaluate_native_table_primitives,
    recover_native_table_primitives,
    diagnose_native_table_primitives,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
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


f = load("character-dedup")
extra = load("pdf-native-dedup-extra")
ties = load("character-dedup-ties")
cases = f["cases"] + extra["raw_cases"] + ties["cases"]


def run(c):
    rows = copy.deepcopy(c["chars"])
    before = copy.deepcopy(rows)
    same(dedup(rows), c["expected"], c["label"])
    same(rows, before, "ownership")


for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::50]))
assert len(cases) == 840
pdf_cases = load("pdf-native-geometry")["cases"] + extra["cases"]


def run_pdf(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    same(
        read(data, page_index=c["page_index"], include_geometry=c["extended"]),
        c["expected"],
        c["pdf"],
    )


for c in pdf_cases:
    run_pdf(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run_pdf, pdf_cases[::9]))
assert len(pdf_cases) == 64
# 字符与绘图均实时从Rust读取，不再从冻结JSON注入实际PDF的chars。
e = load("native-table-engine")
for r in e["real"]:
    c = e["cases"][r["case_index"]]
    ti = copy.deepcopy(c["request"]["table_input"])
    geometry = read(root / r["pdf"], page_index=r["page_index"])
    vectors = read_pdf_vector_geometry(root / r["pdf"], page_index=r["page_index"])
    assert len(geometry["chars"]) == len(ti["chars"])
    same(
        [{k: new[k] for k in old} for new, old in zip(geometry["chars"], ti["chars"])],
        ti["chars"],
        "actual-char-contract",
    )
    ti.update(
        chars=geometry["chars"],
        drawing_lines=vectors["drawing_lines"],
        rectangles=vectors["path_infos"],
        page_size=geometry["page_size"],
    )
    same(evaluate_native_table_primitives(ti), c["expected"]["evaluation"], "live-PDF-evaluate")
    same(recover_native_table_primitives(ti), c["expected"]["result"], "live-PDF-result")
    same(diagnose_native_table_primitives(ti), c["expected"]["diagnostics"], "live-PDF-diagnose")
stream = io.BytesIO((root / pdf_cases[0]["pdf"]).read_bytes())
same(read(stream, include_geometry=False), pdf_cases[0]["expected"])
assert not stream.closed
base = next(c["chars"] for c in cases if c["chars"])
for rows in (
    [base[0]] * 5001,
    [base[0], base[0]],
    [dict(base[0], source_indices=[])],
    [dict(base[0], writing_angle=float("nan"))],
):
    try:
        dedup(rows)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError("invalid character input accepted")
for kwargs in [
    dict(page_index=-1),
    dict(page_index=True),
    dict(page_index=99999),
    dict(include_geometry=1),
    dict(password="a\0b"),
]:
    try:
        read(root / pdf_cases[0]["pdf"], **kwargs)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(kwargs)
try:
    read(b"invalid")
except RapidDocError as exc:
    assert exc.code == "corrupt_document"
else:
    raise AssertionError("invalid PDF accepted")
for password in (None, "wrong"):
    try:
        read(root / "build/migration/encrypted-native.pdf", password=password)
    except RapidDocError as exc:
        assert exc.code == "encrypted"
    else:
        raise AssertionError("encrypted PDF accepted")
read(root / "build/migration/encrypted-native.pdf", password="test-password")
run(cases[0])
run_pdf(pdf_cases[0])
assert (
    capabilities()["docvortex_character_dedup"]
    and capabilities()["docvortex_pdf_native_text_geometry"]
)
assert capabilities()["legacy_native_table_recovery"] is False
import rapid_doc_rust as sdk

for name in ("UNICODE-DATA.txt", "EQUIVALENT-IDEOGRAPH-17.txt", "MIGRATION-NOTICES.txt"):
    reference = (
        root / "licenses" / name
        if name == "MIGRATION-NOTICES.txt"
        else root / "rust/rapiddoc-pdf/licenses" / name
    )
    assert (Path(sdk.__file__).parent / "licenses" / name).read_bytes() == reference.read_bytes()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "Passed 840 full dedup cases, 64 actual PDF geometry cases, 16 live-PDF engine eval/recovery/diagnostic chains, concurrency, ownership, errors and licenses"
)
