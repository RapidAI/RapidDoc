"""Installed wheel versus captured upstream regression inputs and Python outputs."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy
import gzip
import json
import sys
from pathlib import Path

from rapid_doc_rust import (
    evaluate_native_table_primitives,
    recover_native_table_primitives,
    diagnose_native_table_primitives,
    render_native_table_html_with_scripts,
    trace_native_inline_evidence,
)


def same(a, b, path="result"):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for key in a:
            same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    else:
        # JSON integer vs float representation is not a geometry difference.
        assert a == b, (path, a, b)


root = Path(sys.argv[1]).resolve()
with gzip.open(
    root / "tests/fixtures/upstream_068/pdf_reference.json.gz", "rt", encoding="utf-8"
) as file:
    cases = json.load(file)["cases"]
for index, case in enumerate(cases):
    request = copy.deepcopy(case["request"])
    before = copy.deepcopy(request)
    if case["kind"] == "table":
        actual = dict(
            evaluation=evaluate_native_table_primitives(**request),
            result=recover_native_table_primitives(**request),
            diagnostics=diagnose_native_table_primitives(**request),
        )
    elif case["kind"] == "render":
        actual = render_native_table_html_with_scripts(**request)
    else:
        actual = trace_native_inline_evidence(**request)
    same(actual, case["expected"], f"case {index} ({case['kind']})")
    same(request, before, f"case {index} input ownership")
print(
    f"Installed upstream PDF reference parity: {len(cases)} cases, including styled table HTML and diagnostics"
)
