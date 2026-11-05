"""安装包中的旧模型顺序整链差分，严格比较 Python 类型与完整 JSON。"""

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
from rapid_doc_rust import sort_model_blocks, split_paragraphs, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/legacy-block-sort.json").read_text())
assert (
    hashlib.sha256((root / fixture["source"]).read_bytes()).hexdigest() == fixture["source_sha256"]
)
cases = fixture["cases"]
assert len(cases) == 1206


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


for c in cases:
    before = copy.deepcopy(c["request"])
    equal(sort_model_blocks(**c["request"]), c["expected"], c["name"])
    equal(c["request"], before, "ownership")
with ThreadPoolExecutor(max_workers=3) as pool:
    sample = cases[::150]
    for result, c in zip(pool.map(lambda c: sort_model_blocks(**c["request"]), sample), sample):
        equal(result, c["expected"])

request = dict(
    blocks=[dict(type="text", bbox=[0, 0, 100, 20], lines=[], original_order=0)],
    page_width=600,
    page_height=800,
)
bad_cases = []
for change, code in [
    (dict(original_order=None), "unsupported"),
    (dict(original_order="0"), "invalid_input"),
    (dict(original_order=2**54), "invalid_input"),
    (dict(bbox=[0, 0, 0, 20]), "invalid_input"),
    (dict(real_lines=[{}]), "invalid_input"),
]:
    r = copy.deepcopy(request)
    r["blocks"][0].update(change)
    bad_cases.append((r, code))
r = copy.deepcopy(request)
r["blocks"] *= 1001
bad_cases.append((r, "resource_limit"))
r = copy.deepcopy(request)
r["blocks"][0]["metadata"] = "x" * 2_000_001
bad_cases.append((r, "resource_limit"))
r = copy.deepcopy(request)
r["blocks"][0]["lines"] = [dict(bbox=[0, 0, 100, 1], spans=[])]
r["blocks"].append(
    dict(type="image_body", bbox=[0, 0, 200, 1_000_000], lines=[], original_order=1, group_id=0)
)
bad_cases.append((r, "resource_limit"))
for r, code in bad_cases:
    try:
        sort_model_blocks(**r)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(("accepted invalid input", code))
try:
    sort_model_blocks({}, page_width=600, page_height=800)
except TypeError:
    pass
else:
    raise AssertionError("non-list accepted")
r = copy.deepcopy(request)
r["blocks"][0]["metadata"] = "\ud800"
try:
    sort_model_blocks(**r)
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid Unicode accepted")

# 顺序结果可交给上一阶段移植的旧段落入口；完整旧串联参考另行冻结。
paragraph_refs = json.loads((root / "build/migration/legacy-sort-paragraphs.json").read_text())
assert len(paragraph_refs["cases"]) == 308
for path, sha in paragraph_refs["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
for c in paragraph_refs["cases"]:
    blocks = sort_model_blocks(**c["request"])
    pages = [
        dict(
            page_idx=0,
            page_size=[c["request"]["page_width"], c["request"]["page_height"]],
            preproc_blocks=blocks,
        )
    ]
    equal(split_paragraphs(pages), c["expected"], c["name"])
assert capabilities()["legacy_model_block_order"] is True
assert capabilities()["compatibility"]["python_parser"] is False
print(
    f"Installed wheel: {len(cases)} exact legacy model-order JSON references, {len(paragraph_refs['cases'])} old sort→para_split references, stable ties/groups/virtual lines, ownership/concurrency and errors/budgets passed"
)
