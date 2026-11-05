"""独立wheel：实际ftfy6.3.1/旧公式post_process和Unicode M模型CPU，无Python清洗依赖。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys, importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    fix_formula_text,
    OnnxSession,
    execute_model_formulas,
    capabilities,
    RapidDocError,
)
from rapid_doc_rust._native import fix_unicode_text

root = Path(sys.argv[1]).resolve()
source = root / "build/migration/unicode-fix.json"
assert importlib.util.find_spec("ftfy") is None
f = json.loads(source.read_text(encoding="utf-8"))
p = json.loads((root / "build/migration/formula-unicode.json").read_text(encoding="utf-8"))
assert hashlib.sha256(source.read_bytes()).hexdigest() == p["source_sha256"]
assert (
    hashlib.sha256((root / p["reference"]["path"]).read_bytes()).hexdigest()
    == p["reference"]["sha256"]
)
assert len(f["cases"]) == 29517 and len(p["expected"]) == 29517
for name, sha in f["source_hashes"].items():
    path = (
        root
        / ".venv/Lib/site-packages"
        / ((name.replace(".", "/") + ".py") if name != "ftfy" else "ftfy/__init__.py")
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name
for i, (c, e) in enumerate(zip(f["cases"], p["expected"])):
    assert fix_unicode_text(c["text"]) == c["expected"], ("ftfy", i)
    assert fix_formula_text(c["text"]) == e, ("formula", i)


def fails(fn):
    try:
        fn()
    except (ValueError, TypeError, RapidDocError):
        return
    raise AssertionError("ftfy budget accepted")


fails(lambda: fix_unicode_text("x" * 65537))
fails(lambda: fix_unicode_text("&amp;" + "amp;" * 200))
assert fix_unicode_text("ＡＢＣ e\u0301") == "ABC é"
assert fix_unicode_text("&amp;\n<x &amp;\n&amp;") == "&\n<x &amp;\n&amp;"
assert fix_unicode_text("&#59;") == "&#59;"
with ThreadPoolExecutor(max_workers=4) as pool:
    assert list(pool.map(fix_unicode_text, ["sÃ³"] * 12)) == ["só"] * 12
base = json.loads((root / "build/migration/formula-decode.json").read_text(encoding="utf-8"))
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / base["model"]["path"],
    base["model"]["sha256"],
    threads=1,
)
for c in p["real"]:
    images = [(root / path).read_bytes() for path in c["sources"]]
    assert session.recognize_formula_batch_png(images, batch_size=c["batch_size"]) == c["expected"]
    plans = [
        dict(
            model=dict(layout_dets=[dict(category_id=13)]),
            formulas=[dict(layout_index=0, source="formula")],
            images=dict(formula=img),
        )
        for img in images
    ]
    before = copy.deepcopy(plans)
    out = execute_model_formulas(plans, recognizer=session, batch_size=c["batch_size"])
    assert [v["model"]["layout_dets"][0]["latex"] for v in out] == c["expected"] and plans == before
caps = capabilities()
assert caps["formula_full_ftfy"] and caps["formula_ftfy_version"] == "6.3.1"
print(
    "29517 exact old ftfy + 29517 exact full formula post_process cases; 2 real Unicode M CPU batches and cross-page filling; budget recovery, HTML auto state, ownership and concurrency passed"
)
