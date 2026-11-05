"""真实 wheel：实验默认关闭、来源可追溯和旧 Python 合同的实际偏差审查。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import copy, hashlib, inspect, json, sys
from rapid_doc_rust import (
    organize_page_text,
    recognize_pdf_page_hybrid,
    compatibility_report,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())


class Unreadable:
    def read(self, *args):
        raise AssertionError("experimental gate read the input")


for function, args, kwargs in [
    (organize_page_text, (None,), {}),
    (recognize_pdf_page_hybrid, (Unreadable(),), dict(layout_model=None)),
]:
    for value in (None, False, 1, "yes"):
        try:
            function(*args, experimental=value, **kwargs)
        except ValueError as e:
            assert "experimental=True" in str(e)
        else:
            raise AssertionError("experimental rules enabled implicitly")
    assert inspect.signature(function).parameters["experimental"].default is False
assert (
    inspect.signature(recognize_pdf_page_hybrid).parameters["prefer_native_tables"].default is False
)
audit = compatibility_report()
assert len(audit["modules"]) == 119
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_native_text")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_post_ocr")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_cross_page_tables")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_table_assets")["status"]
    == "intentional_output_difference"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_formula_numbers")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_model_page_assembly")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_model_block_order")["status"]
    == "sampled_differential"
)
assert (
    next(m for m in audit["modules"] if m["id"] == "legacy_paragraphs")["status"]
    == "sampled_differential"
)
assert not any(audit["claims"].values())
assert capabilities()["compatibility"]["python_parser"] is False
assert capabilities()["compatibility"]["native_tables_default"] is False
for module in audit["modules"]:
    for field in ("rust_sources", "python_references"):
        for path, sha in module[field].items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
for name in (
    "paragraphs_lists",
    "geometric_reading_order",
    "native_line_arbitration",
    "native_table_arbitration",
):
    assert (
        next(m for m in audit["modules"] if m["id"] == name)["status"] == "experimental_new_rules"
    )

baseline = load("compatibility")
for p, sha in baseline["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == sha
report = dict(
    scope=baseline["scope"],
    paragraphs=[],
    block_order=[],
    skipped=baseline["skipped"],
    quality_conclusion="not_evaluated",
    default_experiments_enabled=False,
)
for c in baseline["paragraphs"]:
    result = organize_page_text(c["page"], experimental=True, **c["options"])
    actual = [
        dict(kind=p["kind"], sources=[[p["block_index"], i] for i in p["line_indices"]])
        for p in result["paragraphs"]
    ]
    report["paragraphs"].append(
        dict(
            name=c["name"],
            matches_legacy=actual == c["legacy"],
            legacy=c["legacy"],
            experimental=actual,
        )
    )
for c in baseline["block_order"]:
    result = organize_page_text(c["page"], experimental=True, block_order="geometry")
    order = result["block_order"]["indices"]
    report["block_order"].append(
        dict(
            name=c["name"],
            matches_legacy=order == c["legacy"],
            legacy=c["legacy"],
            experimental=order,
        )
    )
report["summary"] = {
    name: dict(
        compared=len(report[name]), different=sum(not x["matches_legacy"] for x in report[name])
    )
    for name in ("paragraphs", "block_order")
}
assert report["summary"]["paragraphs"]["compared"] and report["summary"]["block_order"]["compared"]
target = root / "build/compatibility-differences-current.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# 默认不使用原生表格策略，提供实验许可也不自动覆盖模型结果。
tables = load("native-tables")
case = tables["cases"][0]


class Layout:
    def layout_png(self, *args, **kwargs):
        return case["layout"]


class Classifier:
    def classify_table_png(self, *args, **kwargs):
        raise RuntimeError("model-route-required")


try:
    recognize_pdf_page_hybrid(
        tables["pdf_input"],
        experimental=True,
        layout_model=Layout(),
        detector=object(),
        recognizer=object(),
        classifier=Classifier(),
        dpi=72,
    )
except RuntimeError as e:
    assert str(e) == "model-route-required"
else:
    raise AssertionError("native table arbitration replaced the model by default")
native = recognize_pdf_page_hybrid(
    tables["pdf_input"], experimental=True, prefer_native_tables=True, layout_model=Layout(), dpi=72
)
assert native["blocks"][0]["recognition"]["route"] == "native"
print(
    f"Compatibility gate passed; audited deviations (not quality judgements): {report['summary']}; source hashes and model default verified"
)
