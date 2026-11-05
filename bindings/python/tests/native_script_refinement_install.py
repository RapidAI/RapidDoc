"""Installed wheel: old token, region, line and fraction reference chains."""

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
    refine_native_script_tokens as refine,
    classify_native_script_runs as runs,
    classify_native_script_line as line,
    native_fraction_member_sources as fraction,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def exact(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def check(function, c):
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    exact(function(**request), c["expected"], c["label"])
    exact(request, before)


f = load("script-refinement")
g = load("script-fractions")
h = load("script-fraction-pipeline")
for key, fn, n in [("refine", refine, 1838), ("runs", runs, 2757), ("lines", line, 560)]:
    assert len(f[key]) == n
    for c in f[key]:
        check(fn, c)
assert len(g["cases"]) == 843 and sum(bool(c["expected"]) for c in g["cases"]) == 255
for c in g["cases"]:
    check(fraction, c)


def pipeline(c):
    request = copy.deepcopy(c["fraction"])
    before = copy.deepcopy(request)
    members = fraction(**request)
    exact(members, c["expected_members"], c["label"])
    exact(request, before)
    request = copy.deepcopy(c["line"])
    before = copy.deepcopy(request)
    exact(line(**request), c["before"], c["label"])
    exact(request, before)
    exact(line(**request, fraction_members=members), c["expected"], c["label"])
    exact(request, before)
    assert c["before"]["roles"] != c["expected"]["roles"]


assert len(h["cases"]) == 20
for c in h["cases"]:
    pipeline(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    for key, fn in [("refine", refine), ("runs", runs), ("lines", line)]:
        list(pool.map(lambda c: check(fn, c), f[key][::157]))
    list(pool.map(lambda c: check(fraction, c), g["cases"][::97]))
    list(pool.map(pipeline, h["cases"][::4]))


def rejects(fn, request):
    try:
        fn(**request)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(("invalid input accepted", fn.__name__))


for fn, request in [
    (refine, dict(chars=[{}] * 1501, roles=["body"] * 1501)),
    (refine, dict(chars=[{}], roles=[])),
    (refine, dict(chars=[], roles=[], formula_region=1)),
    (refine, dict(chars=[], roles=[], ordinary_native_roles="yes")),
    (runs, dict(chars=[{}], memberships=[])),
    (runs, dict(chars=[{}], tight_bboxes=[1])),
    (line, dict(line=dict(chars=[{}]), page_size=[-1, 300])),
    (line, dict(line=dict(chars=[{}]), page_size=[float("nan"), 300])),
    (line, dict(line=dict(chars=[{}]), page_size=[300, 300], fraction_members=[0] * 20001)),
    (fraction, dict(chars=[], drawing_lines=[[]] * 5001, page_size=[300, 300])),
    (
        fraction,
        dict(chars=[{}], drawing_lines=[[0, 0, 1, 1]], page_size=[300, 300], prepared_rules=[]),
    ),
]:
    rejects(fn, request)
for key, fn in [("refine", refine), ("runs", runs), ("lines", line)]:
    check(fn, f[key][0])
check(fraction, g["cases"][0])
pipeline(h["cases"][0])
caps = capabilities()
for key in (
    "native_script_token_refinement",
    "native_script_fraction_protection",
    "native_script_line_roles",
):
    assert caps[key] is True
assert caps["legacy_native_table_recovery"] is False and caps["legacy_table_execution"] is False
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").read_bytes() == (
    root / "licenses/MIGRATION-NOTICES.txt"
).read_bytes()
assert "rapid_doc" not in sys.modules
print(
    "Passed 1838 token refinements, 2757 full runs, 560 full lines, 843 fraction references (255 positive), 20 computed fraction-to-line chains, ownership, concurrency, limits and attribution; whole-table orchestration pending"
)
