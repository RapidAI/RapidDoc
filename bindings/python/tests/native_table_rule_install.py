"""独立wheel旧原生捷径策略与已有候选仲裁；恢复器不是Rust原生生成后端。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    plan_native_table_rule,
    accept_native_table_rule,
    select_native_table_candidates,
    try_native_table_rule,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-table-rule.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
engines = json.loads(
    (root / "build/migration/native-table-engine-values.json").read_text(encoding="utf-8")
)
assert (
    hashlib.sha256((root / "rapid_doc/model/docvortex_adapter.py").read_bytes()).hexdigest()
    == engines["source_sha256"]
)
for c in engines["cases"]:
    request = copy.deepcopy(f["policy"][0]["request"])
    table = request.pop("table")
    request["config"] = c["config"]
    assert plan_native_table_rule(table, **request)["attempt"] == c["expected"]
for c in f["policy"]:
    request = copy.deepcopy(c["request"])
    table = request.pop("table")
    before = copy.deepcopy(table)
    plan = plan_native_table_rule(table, **request)
    assert plan["attempt"] == c["expected"]["attempt"], c
    if plan["attempt"]:
        assert plan["table_bbox"] == c["expected"]["bbox"]
        accepted = accept_native_table_rule(table, c["result"])
        assert accepted == dict(accepted=c["expected"]["accepted"], table=c["expected"]["table"])
    assert table == before
for c in f["selection"]:
    before = copy.deepcopy(c["request"])
    assert select_native_table_candidates(**c["request"]) == c["expected"]
    assert before == c["request"]
source = (root / "build/migration/model-document/five-pages.pdf").read_bytes()
table = f["policy"][0]["request"]["table"]


class Never:
    def read(self, *a):
        raise AssertionError("guard unexpectedly read input")


for kw, reason in [
    (dict(config={"force_ocr": True}), "force_ocr"),
    (dict(has_source_pdf=False), "missing_pdf"),
    (dict(rotate_label="90"), "page_rotation"),
    (dict(formulas=[dict(bbox=[20, 20, 30, 30])]), "formula_or_checkbox"),
]:
    r = try_native_table_rule(Never(), table, **kw)
    assert not r["accepted"] and r["diagnostic"]["reason"] == reason
assert try_native_table_rule(Never(), table)["diagnostic"]["reason"] == "backend_unavailable"


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


class Provider:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def recover_table(self, data, **kw):
        self.calls.append(kw)
        assert data == source
        if self.error:
            raise self.error
        return self.value


p = Provider(dict(html="<table>raw</table>", confidence=1))
stream = Once(source)
before = copy.deepcopy(table)
r = try_native_table_rule(stream, table, recoverer=p, scale=2, page_index=2)
assert (
    r["accepted"]
    and r["table"]["rule_table_score"] == 1
    and type(r["table"]["rule_table_score"]) is int
)
assert (
    p.calls[0]["bbox"] == [10.0, 10.0, 40.0, 40.0]
    and stream.count == 1
    and not stream.closed
    and table == before
)
for value in [None, dict(html=" ", confidence=1), dict(html="x", confidence=0.899999999999)]:
    assert not try_native_table_rule(source, table, recoverer=Provider(value))["accepted"]
r = try_native_table_rule(source, table, recoverer=Provider(error=RuntimeError("native failed")))
assert (
    r["diagnostic"] == dict(method="model", reason="recovery_error", error_type="RuntimeError")
    and not r["accepted"]
)
try:
    try_native_table_rule(
        source, table, recoverer=Provider(error=ImportError("dependency missing"))
    )
except ImportError:
    pass
else:
    raise AssertionError("native ImportError swallowed")
with ThreadPoolExecutor(max_workers=2) as pool:
    assert all(
        pool.map(
            lambda c: select_native_table_candidates(**c["request"]) == c["expected"],
            f["selection"][:20],
        )
    )
assert (
    capabilities()["legacy_native_table_policy"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    "Installed wheel: 547 old native guards/results and 425 candidate selections; lazy input, stream ownership, provider errors/ImportError and concurrency passed; native generation is not implemented"
)
