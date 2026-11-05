"""独立wheel旧原生字形/视觉行/token与单元格来源片段；不依赖Python Unicode/NumPy。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    prepare_native_table_glyphs,
    build_native_table_text,
    build_native_table_cell_text,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return f


f = load("native-table-text")
cells = load("native-table-cell-text")


def unicode_digest(path, names, types):
    # rustfmt只改变布局；按原生成格式重建实际常量数据以核对冻结哈希，不改参考。
    source = path.read_text(encoding="utf-8")
    canonical = source.splitlines()[0] + "\n"
    for name, type_ in zip(names, types):
        body = source.split("pub const " + name, 1)[1].split("=", 1)[1].split(";", 1)[0]
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", body)
        assert not re.sub(r"[&\[\],\s]", "", re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)", "", body))
        canonical += (
            "pub const "
            + name
            + ": &["
            + type_
            + "] = &[\n"
            + "".join(f"({a},{b}),\n" for a, b in pairs)
            + "];\n"
        )
    return {
        hashlib.sha256(s.encode("utf-8")).hexdigest()
        for s in [canonical, canonical.replace("\n", "\r\n")]
    }


assert f["unicode_sha256"] in unicode_digest(
    root / "rust/rapiddoc-native/src/table_unicode.rs",
    ["PRINTABLE", "DECIMALS"],
    ["(u32,u32)", "(u32,u8)"],
)
assert cells["unicode_sha256"] in unicode_digest(
    root / "rust/rapiddoc-native/src/table_latin.rs", ["LATIN"], ["(u32,u32)"]
)


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    before = copy.deepcopy(c["request"])
    same(prepare_native_table_glyphs(**c["request"]), c["pending"], "pending")
    r = build_native_table_text(**c["request"])
    same(r, c["expected"], "text")
    same(c["request"], before, "ownership")
    return r


for c in f["cases"]:
    run(c)
extra = json.loads(
    (root / "build/migration/native-table-text-reversed.json").read_text(encoding="utf-8")
)
assert (
    hashlib.sha256(
        (
            root / "rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/text.py"
        ).read_bytes()
    ).hexdigest()
    == extra["source_sha256"]
)
for c in extra["cases"]:
    same(build_native_table_text(**c["request"]), c["expected"], "reversed table bbox")
for c in f["real"]:
    assert hashlib.sha256((root / c["pdf"]).read_bytes()).hexdigest() == c["pdf_sha256"]
for c in cells["cases"]:
    before = copy.deepcopy(c["request"])
    same(build_native_table_cell_text(**c["request"]), c["expected"], "cell")
    same(c["request"], before, "cell ownership")
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::20]))


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid/overbudget native text input accepted")


fails(
    lambda: build_native_table_text(
        [dict(char="x", bbox=[1, 1, 2, 2])] * 3001, table_bbox=[0, 0, 100, 100]
    )
)
fails(
    lambda: build_native_table_text(
        [dict(char="x", bbox=[1, 1, float("nan"), 2])], table_bbox=[0, 0, 100, 100]
    )
)
fails(
    lambda: build_native_table_text(
        [dict(char="x", bbox=[1, 1, 2, 2], char_idx=10**14)], table_bbox=[0, 0, 100, 100]
    )
)
fails(lambda: build_native_table_cell_text([], median_height=0))
assert (
    capabilities()["native_table_text_preparation"]
    and capabilities()["native_table_cell_text"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    "Installed wheel: 362 old glyph/row/token cases including 16 real PDF/angle inputs, 12 reversed-table advance cases, 1934 old cell text/source-parts cases; exact values/types, ownership/concurrency/budgets passed; no native candidate generation/PDF geometry extraction"
)
