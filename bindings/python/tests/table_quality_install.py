"""对照旧 Python selector，验证规范 HTML 的计数/文字命中和选优结果。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, sys
from rapid_doc_rust import select_table_candidate, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-quality.json").read_text())
reasons = set()
for case in fixture["cases"]:
    result = select_table_candidate(
        case["ocr_texts"], wired_html=case["wired_html"], wireless_html=case["wireless_html"]
    )
    assert result["wired"] == case["wired"] and result["wireless"] == case["wireless"], (
        case["id"],
        result,
        case,
    )
    assert case[result["route"] + "_html"] == case["expected_html"], (
        case["id"],
        result,
        case["expected_route"],
    )
    if case["expected_route"] is not None:
        assert result["route"] == case["expected_route"]
    reasons.add(result["reason"])
assert reasons == {
    "wired_text_coverage",
    "wireless_scale",
    "wireless_cell_gap",
    "wireless_small_equal",
    "wireless_text_coverage",
    "wired_default",
}, reasons
for bad in [
    "<table><tr><td>x</tr></table>",
    "<table><tr><td>&nbsp;</td></tr></table>",
    "<script>run()</script>",
    '<table><tr><td colspan="0"></td></tr></table>',
]:
    try:
        select_table_candidate([], wired_html=bad, wireless_html="")
    except RapidDocError as e:
        assert e.code == "invalid_input"
    else:
        raise AssertionError("noncanonical candidate HTML accepted")
try:
    select_table_candidate(["x"] * 1001, wired_html="", wireless_html="")
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("excessive OCR accepted")
large = "<table><tr><td>" + "x" * 100000 + "</td></tr></table>"
try:
    select_table_candidate(["x"] * 300, wired_html=large, wireless_html=large)
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("text-search work bypassed budget")
case = fixture["cases"][0]


def run(_):
    return select_table_candidate(
        case["ocr_texts"], wired_html=case["wired_html"], wireless_html=case["wireless_html"]
    )


expected = run(0)
with ThreadPoolExecutor(max_workers=3) as pool:
    assert all(v == expected for v in pool.map(run, range(12)))
print(
    f"Installed wheel: {len(fixture['cases'])} legacy table-quality decisions/metrics, all 6 reasons, budgets/errors and concurrency passed"
)
