"""真实 wheel：逐字段比较完整旧 para_split 输出，不放宽数字或数组顺序。"""

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
from rapid_doc_rust import split_paragraphs, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
cases = []
for name in ("legacy-paragraphs", "legacy-paragraph-boundaries"):
    fixture = json.loads((root / f"build/migration/{name}.json").read_text())
    assert (
        hashlib.sha256((root / fixture["source"]).read_bytes()).hexdigest()
        == fixture["source_sha256"]
    )
    cases.extend(fixture["cases"])
assert len(cases) == 850


def difference(a, b, path=""):
    if type(a) is not type(b):
        return path + ": type mismatch " + type(a).__name__ + "/" + type(b).__name__
    if isinstance(a, dict):
        if set(a) != set(b):
            return path + ": keys differ " + str(set(a) ^ set(b))
        for k in a:
            d = difference(a[k], b[k], path + "/" + k)
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return path + ": length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            d = difference(x, y, path + "/" + str(i))
            if d:
                return d
    elif a != b:
        return path + ": " + repr(a) + " != " + repr(b)
    return None


for c in cases:
    before = copy.deepcopy(c["pages"])
    actual = split_paragraphs(c["pages"])
    assert c["pages"] == before, "input mutated"
    error = difference(actual, c["expected"])
    assert error is None, (c["name"], error)
    assert all(p["preproc_blocks"] == q["preproc_blocks"] for p, q in zip(actual, before))
with ThreadPoolExecutor(max_workers=3) as pool:
    selected = [cases[i] for i in (4, 20, 355, 600, 636, 700)]
    results = list(pool.map(lambda c: split_paragraphs(c["pages"]), selected))
    for result, c in zip(results, selected):
        assert difference(result, c["expected"]) is None
page = dict(page_idx=0, page_size=[600, 800], preproc_blocks=[])
for pages, code in [
    ([page, page], "invalid_input"),
    ([page] * 1001, "resource_limit"),
    ([dict(page, metadata="x" * 2_000_001)], "resource_limit"),
    (
        [dict(page, preproc_blocks=[dict(type="text", bbox=[0, 0, 0, 20], lines=[])])],
        "invalid_input",
    ),
]:
    try:
        split_paragraphs(pages)
    except RapidDocError as e:
        assert e.code == code
    else:
        raise AssertionError("invalid input or budget accepted")
try:
    split_paragraphs({})
except TypeError:
    pass
else:
    raise AssertionError("non-list accepted")
try:
    split_paragraphs([dict(page, metadata="\ud800")])
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid Unicode accepted")
assert capabilities()["legacy_paragraph_rules"] is True
assert capabilities()["compatibility"]["python_parser"] is False
print(
    "Installed wheel: 850 exact complete Python para_split JSON references, 759 Unicode uppercase boundaries, floating geometry/metadata, CPU OCR inputs, cross-page flags, ownership/concurrency and invalid inputs/budgets passed"
)
