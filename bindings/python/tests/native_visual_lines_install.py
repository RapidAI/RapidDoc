"""Installed SDK: complete old visual lines, live PDF characters, ownership and budgets."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    build_native_visual_lines as build,
    read_pdf_character_primitives as raw,
    read_pdf_native_text_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
cases = []
for name in ["native-visual-lines", "native-visual-lines-extra"]:
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    cases.extend(f["cases"])


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


def run(c):
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    same(build(**request), c["expected"], c["label"])
    same(request, before, "ownership")


assert len(cases) == 317
for c in cases:
    run(c)
actual = [c for c in cases if "pdf" in c]
assert len(actual) == 57
for c in actual:
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    dedup = c.get("reader") == "deduplicated"
    geometry = (read if dedup else raw)(data, page_index=c["page_index"])
    chars = geometry["chars" if dedup else "restored_chars"]
    same(chars, c["request"]["chars"], "live characters")
    same(
        build(chars, page_size=geometry["page_size"], page_rotation=geometry["rotation"]),
        c["expected"],
        "live " + c["label"],
    )
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::19]))
c = cases[6]["request"]["chars"][0]
for request in [
    dict(chars=[c] * 4097, page_size=[240.0, 180.0]),
    dict(chars=[dict(c, bbox=[0.0, 0.0, 1e10, 1.0])], page_size=[240.0, 180.0]),
]:
    try:
        build(**request)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("limit not enforced")
# Distinct isolated runs fit individual input limits but exceed the combined grouping+pair budget.
crowded = [
    dict(
        c,
        char="x",
        bbox=[10.0, i * 20.0, 16.0, i * 20.0 + 10.0],
        char_idx=i,
        font=dict(c["font"], name=f"Font{i}"),
    )
    for i in range(3000)
]
try:
    build(crowded, page_size=[240.0, 60010.0])
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("combined work budget was reset between stages")
for options in [
    dict(page_rotation=True),
    dict(page_rotation=1.5),
    dict(supported_angles=[]),
    dict(page_size=[0.0, 1.0]),
]:
    request = copy.deepcopy(cases[6]["request"])
    request.update(options)
    try:
        build(**request)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(options)
run(cases[6])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_visual_lines"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "317 full old visual-line facts / 977 lines; 57 live PDF raw-restored/deduplicated -> visual lines, exact values/types/characters/source ids/typography, ownership/concurrency/limits passed; complete page-source collection/candidate detection still pending"
)
