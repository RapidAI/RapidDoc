"""独立 wheel：旧跨页合并完整 JSON、864 输出及实际单页串联，不加载 Python 解析器。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    merge_cross_page_tables,
    assemble_model_pages,
    render_model_pages,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text())
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return f


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (a, b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


base = load("legacy-table-merge")
extended = load("legacy-table-merge-extended")
chain = load("legacy-table-merge-chain")
assert len(base["cases"]) == 99 and len(extended["cases"]) == 288 and len(chain["cases"]) == 48
edges = load("legacy-table-merge-edges")
assert len(edges["cases"]) == 25
for c in edges["cases"]:
    equal(merge_cross_page_tables(c["pages"]), c["expected"])
for c in extended["cases"]:
    before = copy.deepcopy(c["pages"])
    actual = merge_cross_page_tables(c["pages"])
    equal(c["pages"], before)
    equal(actual, c["expected"])
    for mode, expected in c["outputs"].items():
        equal(render_model_pages(actual, make_mode=mode, image_path_prefix="images"), expected)
for c in chain["cases"]:
    before = copy.deepcopy(c["requests"])
    options = dict(optimize_formula_numbers=c["optimize_formula_numbers"])
    equal(assemble_model_pages(c["requests"], **options), c["unmerged"])
    actual = assemble_model_pages(c["requests"], merge_tables=True, **options)
    equal(actual, c["expected"])
    equal(c["requests"], before)
    for mode, expected in c["outputs"].items():
        equal(render_model_pages(actual, make_mode=mode, image_path_prefix="images"), expected)
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = extended["cases"][::19]
    for actual, c in zip(pool.map(lambda c: merge_cross_page_tables(c["pages"]), samples), samples):
        equal(actual, c["expected"])
# 显式 API 不隐式读取 MINERU_TABLE_MERGE_ENABLE；正式路由尚未接入。
os.environ["MINERU_TABLE_MERGE_ENABLE"] = "false"
equal(merge_cross_page_tables(extended["cases"][0]["pages"]), extended["cases"][0]["expected"])
for html in [
    "<table><tr><td>unclosed",
    '<table><tr><td rowspan="0">bad</td></tr></table>',
    "<table><tr><td>&bogus;</td></tr></table>",
    "<table><tr><td><script>x</script></td></tr></table>",
    "<table><tr><td><table></table></td></tr></table>",
]:
    pages = copy.deepcopy(base["cases"][0]["pages"])
    pages[1]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["html"] = html
    before = copy.deepcopy(pages)
    try:
        merge_cross_page_tables(pages)
    except RapidDocError as e:
        assert e.code == "unsupported"
    else:
        raise AssertionError("unsupported HTML accepted")
    equal(pages, before)
for pages in [
    [dict(para_blocks=[None])],
    [dict(para_blocks="wrong")],
    [dict(para_blocks=[])] * 1001,
]:
    try:
        merge_cross_page_tables(pages)
    except RapidDocError:
        pass
    else:
        raise AssertionError("invalid/budgeted page accepted")
for value in [None, 1, "true"]:
    try:
        assemble_model_pages([], merge_tables=value)
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool merge option accepted")
assert merge_cross_page_tables([]) == []
assert capabilities()["legacy_cross_page_tables"] is True
assert capabilities()["compatibility"]["python_parser"] is False
print(
    "Installed wheel: 313 exact old cross-page table pages and 864 outputs; 48 actual model→paragraph→merge chains/144 outputs; spans, headers, sparse row carry, captions/footnotes, stable object identity, HTML entities/whitespace, ownership/concurrency and explicit errors/budgets passed"
)
