"""独立wheel最终表格HTML归一化与真实SLANet最终输出。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import hashlib, io, json, sys
from pathlib import Path
from rapid_doc_rust import (
    normalize_legacy_table_html,
    recognize_legacy_slanet_table_png,
    OnnxSession,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/table-normalize.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in f["html"]:
    if c.get("raises"):
        try:
            normalize_legacy_table_html(c["input"])
        except RapidDocError:
            pass
        else:
            raise AssertionError("old malformed numeric entity accepted")
    else:
        assert normalize_legacy_table_html(c["input"]) == c["expected"], c["input"]
assert normalize_legacy_table_html(None) is None and normalize_legacy_table_html("") == ""
raw = json.loads((root / "build/migration/legacy-table-match.json").read_text(encoding="utf-8"))
models = {
    c["variant"]: OnnxSession(
        root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        root / c["model"],
        c["sha256"],
    )
    for c in raw["real"][::3]
}
for c, expected in zip(raw["real"], f["real"]):
    r = recognize_legacy_slanet_table_png(
        root / c["source"],
        c["ocr_result"],
        structure_model=models[c["variant"]],
        variant=c["variant"],
    )
    assert r["html"] == expected["expected"] and r["raw_html"] == expected["raw_html"]


class Empty:
    def detect_table_png(self, data):
        return []


c = raw["real"][0]
s = io.BytesIO((root / c["source"]).read_bytes())
assert recognize_legacy_slanet_table_png(s, None, detector=Empty())["html"] is None and not s.closed
print(
    "Installed wheel: 2270 old HTML cases and 6 actual SLANet final HTML outputs passed; no-text skip preserved"
)
