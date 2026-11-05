"""独立安装：实际旧公式解码、默认ftfy、M模型CPU及跨页公式回填。"""

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
    FormulaDecoder,
    repair_formula_latex,
    OnnxSession,
    execute_model_formulas,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/formula-decode.json").read_text(encoding="utf-8"))
metadata = (root / "build/migration/formula-character.json").read_bytes()
assert hashlib.sha256(metadata).hexdigest() == f["metadata_sha256"]
for path, sha in f["references"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
assert len(f["repairs"]) == 673 and len(f["cases"]) == 1352
decoder = FormulaDecoder(metadata.decode("utf-8"))
configs = json.loads(
    (root / "build/migration/formula-token-configs-v2.json").read_text(encoding="utf-8")
)
for c in configs["cases"]:
    data = json.loads(metadata)
    data["tokenizer_config_file"] = c["config"]
    assert FormulaDecoder(data).decode(c["rows"]) == c["expected"]
for i, c in enumerate(f["repairs"]):
    assert repair_formula_latex(c["text"]) == c["expected"], i
supported = []
unsupported = []
for c in f["cases"]:
    (supported if c["supported"] else unsupported).append(c)
for start in range(0, len(supported), 16):
    rows = supported[start : start + 16]
    assert decoder.decode([c["tokens"] for c in rows]) == [c["expected"] for c in rows]
for c in unsupported:
    assert decoder.decode([c["tokens"]]) == [c["expected"]]


def fails(fn):
    try:
        fn()
    except (ValueError, TypeError, OverflowError, RapidDocError):
        return
    raise AssertionError("invalid formula operation accepted")


for value in ["{}", "x", " " * 8_000_001]:
    fails(lambda: FormulaDecoder(value))
fails(lambda: decoder.decode([[0] * 8193]))
fails(lambda: decoder.decode([[-1]]))
fails(lambda: repair_formula_latex("x" * 65537))
with ThreadPoolExecutor(max_workers=3) as pool:
    assert list(pool.map(lambda _: decoder.decode([[0, 2, 4]]), range(6))) == [[""]] * 6
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
session = OnnxSession(runtime, root / f["model"]["path"], f["model"]["sha256"], threads=1)
assert session.recognize_formula_batch_png([]) == []
for c in f["real"]:
    sources = [(root / p).read_bytes() for p in c["sources"]]
    assert session.recognize_formula_batch_png(sources, batch_size=c["batch_size"]) == c["expected"]
    # 调用同一原生会话，实际PNG→公式→跨页模型回填，旧固定响应回填策略已另行差分。
    plans = [
        dict(
            model=dict(layout_dets=[dict(category_id=13, latex="prior", unknown=2**54 + 1)]),
            formulas=[dict(layout_index=0, source="formula")],
            images={"formula": png},
        )
        for png in sources
    ]
    before = copy.deepcopy(plans)
    out = execute_model_formulas(plans, recognizer=session, batch_size=c["batch_size"])
    assert [p["model"]["layout_dets"][0]["latex"] for p in out] == c["expected"]
    assert plans == before
source = (root / f["real"][0]["sources"][0]).read_bytes()
assert session.recognize_formula_batch_png([source], batch_size=64) == f["real"][0]["expected"]
for batch in [True, 0, 65, "1"]:
    fails(lambda: session.recognize_formula_batch_png([source], batch_size=batch))
fails(lambda: session.recognize_formula_batch_png([b"bad"]))


class Stream(io.BytesIO):
    reads = 0

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


stream = Stream(source)
assert (
    session.recognize_formula_batch_png([stream, stream], batch_size=2)
    == f["real"][0]["expected"] * 2
)
assert stream.reads == 1 and not stream.closed
with ThreadPoolExecutor(max_workers=2) as pool:
    assert (
        list(pool.map(lambda _: session.recognize_formula_batch_png([source]), range(2)))
        == [f["real"][0]["expected"]] * 2
    )
caps = capabilities()
assert caps["formula_model_inference"] and caps["formula_full_ftfy"]
print(
    f"673 actual old repairs, 1352 decoder cases (including {len(unsupported)} formerly unsupported ftfy), 4 real M CPU batches and real cross-page filling; stream ownership, limits and concurrency passed"
)
