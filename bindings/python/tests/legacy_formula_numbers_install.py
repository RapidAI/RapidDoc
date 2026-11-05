"""真实 wheel：旧公式编号关联与模型页面串联，严格 JSON 类型/字段/顺序比较。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib, inspect, json, sys
from rapid_doc_rust import (
    optimize_formula_numbers,
    assemble_model_pages,
    render_model_pages,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
data = json.loads((root / "build/migration/legacy-formula-numbers.json").read_text())
for path, sha in data["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
assert len(data["cases"]) == 718 and len(data["chains"]) == 382


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


for c in data["cases"]:
    before = copy.deepcopy(c["pages"])
    actual = optimize_formula_numbers(c["pages"])
    equal(actual, c["expected"], c["name"])
    equal(c["pages"], before, "ownership")
    equal(optimize_formula_numbers(actual), actual, "idempotent/" + c["name"])
for c in data["chains"]:
    before = copy.deepcopy(c["requests"])
    equal(
        assemble_model_pages(c["requests"], optimize_formula_numbers=True), c["expected"], c["name"]
    )
    equal(c["requests"], before, "ownership")
outputs = json.loads((root / "build/migration/legacy-formula-output.json").read_text())
assert len(outputs["cases"]) == 639
for path, sha in outputs["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
for c in outputs["cases"]:
    pages = assemble_model_pages(c["requests"], optimize_formula_numbers=True)
    equal(
        render_model_pages(pages, make_mode=c["make_mode"], image_path_prefix="images"),
        c["expected"],
        c["name"],
    )
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = data["cases"][::100]
    for actual, c in zip(
        pool.map(lambda c: optimize_formula_numbers(c["pages"]), samples), samples
    ):
        equal(actual, c["expected"])
for pages, code in [
    ([{}] * 1001, "resource_limit"),
    ([dict(metadata="x" * 2_000_001)], "resource_limit"),
    (
        [
            dict(
                preproc_blocks=[
                    dict(
                        type="formula_number",
                        bbox=[0, 0, 10, 10],
                        lines=[dict(spans=[dict(type="text", content=None)])],
                    )
                ]
            )
        ],
        "invalid_input",
    ),
    (
        [dict(preproc_blocks=[dict(type="formula_number", bbox=[0, 0, 10], lines=[])])],
        "invalid_input",
    ),
]:
    try:
        optimize_formula_numbers(pages)
    except RapidDocError as e:
        assert e.code == code
    else:
        raise AssertionError("invalid formula input accepted")
try:
    optimize_formula_numbers({})
except TypeError:
    pass
else:
    raise AssertionError("non-list accepted")
try:
    optimize_formula_numbers([dict(metadata="\ud800")])
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid Unicode accepted")
for flag in [1, None, "true"]:
    try:
        assemble_model_pages([], optimize_formula_numbers=flag)
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool formula flag accepted")
assert (
    inspect.signature(assemble_model_pages).parameters["optimize_formula_numbers"].default is False
)
assert capabilities()["legacy_formula_number_rules"] is True
assert capabilities()["compatibility"]["python_parser"] is False
print(
    "Installed wheel: 718 exact old formula-number JSON references, 382 raw-model→formula→paragraph chains, 639 complete old model→formula→output references; stable ties/neighbor precedence/existing tags/fullwidth punctuation/orphan handling, ownership/idempotency/concurrency and budgets/errors passed"
)
