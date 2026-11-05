"""独立wheel：基础几何分类与给定角色的单元格渲染；不将其当作整表自动语义。"""

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
    classify_native_script_roles as classify,
    native_numeric_script_indices as numeric,
    render_native_table_styled_cell as render,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


f = load("script-geometry")
styles = load("table-styled-cell")


def geom(c):
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    assert classify(**request) == c["expected"]["roles"], c["label"]
    assert numeric(**request) == c["expected"]["numeric_indices"], c["label"]
    assert request == before


for c in f["cases"]:
    geom(c)


def styled(c):
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    assert render(**request) == c["expected"], c["label"]
    assert request == before


for c in styles["cases"]:
    styled(c)
metadata = load("table-style-metadata")
for c in metadata["cases"]:
    styled(c)
assert len(metadata["cases"]) == 390
extra = load("script-numeric-extra")
for c in extra["cases"]:
    assert numeric(**c["request"]) == c["expected"], c["label"]
assert len(extra["cases"]) == 302
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(geom, f["cases"][::67]))
    list(pool.map(styled, styles["cases"][::43]))
assert len(f["cases"]) == 1536 and len(styles["cases"]) == 1288
# 同一几何仅改变来源位置时，保留真实CPython集合平局，不采用固定左右优先。
assert {
    c["expected"]["roles"][-2] for c in f["cases"] if c["label"].startswith("nearest-tie:")
} == {"body", "sup"}
for request in [
    dict(chars=[{}] * 1501),
    dict(chars=[[1]]),
    dict(chars=[], origins=[1]),
    dict(chars=[{}], roles=[]),
]:
    try:
        if "roles" in request:
            numeric(**request)
        else:
            classify(**request)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError("invalid script input accepted")
r = copy.deepcopy(styles["cases"][0]["request"])
r["cell"]["content"] = "&" * 900000
try:
    render(**r)
except RapidDocError as exc:
    assert exc.code == "resource_limit"
else:
    raise AssertionError("escaped fallback exceeded HTML budget")
try:
    render(**dict(styles["cases"][0]["request"], preserve_font_styles=1))
except ValueError:
    pass
else:
    raise AssertionError("non-bool styles flag accepted")
geom(f["cases"][0])
styled(styles["cases"][0])
assert (
    capabilities()["native_script_geometry"] is True
    and capabilities()["native_table_styled_cell"] is True
)
assert capabilities()["legacy_native_table_recovery"] is False
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").read_bytes() == (
    root / "licenses/MIGRATION-NOTICES.txt"
).read_bytes()
assert "rapid_doc" not in sys.modules
print(
    "Passed 1536 base script/numeric classifications including 72 set-order ties plus 302 explicit-role numeric cases and 1288 explicit-role cell HTML cases plus 390 font metadata conversions, ownership, concurrency and limits; this primitive suite does not cover whole-table semantic orchestration"
)
