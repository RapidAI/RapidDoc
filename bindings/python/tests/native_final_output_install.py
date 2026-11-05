"""Installed original output and final lane-order kernels, independently verified."""

# Historical output checkpoints are source-only developer diagnostics.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, json, sys
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b), a, b)
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


count = 0
for name in ["native-final-output", "native-final-order-edges"]:
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for c in f["cases"]:
        q = copy.deepcopy(c["request"])
        before = copy.deepcopy(q)
        try:
            out = json.loads(
                sdk.legacy_pipeline_json("native_final_output", json.dumps(q, allow_nan=False))
            )
        except sdk.RapidDocError as e:
            assert "error" in c, (c["label"], e)
            assert e.code in ["invalid_input", "resource_limit"]
        else:
            assert "error" not in c, c["label"]
            same(out, c["expected"], c["label"])
        same(q, before, "ownership")
        count += 1
assert sdk.capabilities()["compatibility"]["python_parser"] is False
assert not any(n.startswith("rapid_doc._vendor") for n in sys.modules)
print(
    f"{count} original tight/line/normalized output and reference/narrow-graphic order cases; strict types/ownership and no Python parser imports passed"
)
