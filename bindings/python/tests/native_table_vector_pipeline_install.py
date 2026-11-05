"""独立wheel完整矢量候选，实际旧函数/旧PDF原语对照，不运行旧解析器。"""

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
    generate_native_table_vector_candidates,
    inspect_native_table_vector_tracks,
    select_native_table_candidates,
    render_native_table_candidate,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    for r in f.get("real", []):
        assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]
    return f


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


f = load("native-table-vector-pipeline")
pdf = load("native-table-vector-pdf")
cases = f["cases"] + pdf["cases"] + load("native-table-vector-dense")["cases"]


def run(c):
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    result = generate_native_table_vector_candidates(**r)
    same(result, c["expected"])
    same(r, before, "ownership")
    for c, h in zip(result["candidates"], result["html"]):
        assert render_native_table_candidate(c) == h


for c in cases:
    run(c)
for c in f["track_cases"]:
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    same(inspect_native_table_vector_tracks(**r), c["expected"])
    same(r, before)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::15]))
good = next(
    c
    for c in cases
    if c["expected"]["candidates"] and c["expected"]["candidates"][0]["score"] == 1.0
)
result = generate_native_table_vector_candidates(**good["request"])
selected = select_native_table_candidates(result["candidates"])["selected"]
assert (
    selected is not None
    and render_native_table_candidate(result["candidates"][selected]) == result["html"][selected]
)


def fails(r):
    try:
        generate_native_table_vector_candidates(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid vector pipeline accepted")


r = copy.deepcopy(good["request"])
r["table_input"]["drawing_lines"] = [r["table_input"]["drawing_lines"][0]] * 5001
fails(r)
r = copy.deepcopy(good["request"])
r["table_input"]["drawing_lines"][0]["width"] = float("nan")
fails(r)
# 一次调用的raw/supported/rect假设共享预算，不能靠逐阶段重置绕过。
r = copy.deepcopy(good["request"])
base = next(
    c["request"]
    for c in cases
    if len(c["request"]["table_input"]["drawing_lines"]) == 6
    and len(c["request"]["text"]["glyphs"]) == 4
)
r = copy.deepcopy(base)
ls = r["table_input"]["drawing_lines"]
ls.extend([copy.deepcopy(ls[1])] * 4994)
g = copy.deepcopy(r["text"]["glyphs"][0])
r["text"]["glyphs"] = [g | dict(glyph_id=i, source_index=i) for i in range(3000)]
row = copy.deepcopy(r["text"]["rows"][0])
row["glyph_ids"] = list(range(3000))
row["tokens"] = [
    row["tokens"][0] | dict(glyph_ids=list(range(3000)), source_char_indices=list(range(3000)))
]
r["text"]["rows"] = [row]
fails(r)
assert (
    capabilities()["native_table_vector_generation"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(cases)} old full-vector candidates +{len(f['track_cases'])} track-stage cases, including {len(pdf['real'])} actual PDF/angle primitive inputs; exact values/types/diagnostics/HTML, ownership/concurrency/shared budgets and source arbitration passed; PDF extraction/full recovery not implemented"
)
