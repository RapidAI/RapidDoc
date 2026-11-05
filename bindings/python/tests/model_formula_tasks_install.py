"""独立安装：实际旧公式策略差分、跨页适配器合同与原输入所有权。无公式网络推理声明。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from pathlib import Path
from rapid_doc_rust import (
    apply_model_formula_results,
    execute_model_formulas,
    capabilities,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/model-formula-tasks.json").read_text(encoding="utf-8"))
ref = f["reference"]
assert hashlib.sha256((root / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
png = (root / "build/migration/model-ocr-execution/pixels.png").read_bytes()
assert capabilities()["legacy_model_formula_tasks"] and capabilities()["formula_full_ftfy"]


def same(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


for case in f["cases"]:
    plans = case["request"]["plans"]
    before = copy.deepcopy(plans)
    same(apply_model_formula_results(plans, case["request"]["results"]), case["expected"])
    same(plans, before)
    # 在PNG末尾附加标识，既保持有效像素也让适配器能逐一验证顺序。
    inputs = []
    for page, p in enumerate(plans):
        p["images"] = {}
        for task in p["formulas"]:
            data = png + f"{page}:{task['layout_index']}".encode()
            p["images"][task["source"]] = data
            inputs.append(data)

    class Recognizer:
        calls = 0

        def recognize_formula_batch_png(self, images, *, batch_size):
            self.calls += 1
            assert batch_size == case["batch_size"]
            assert images == inputs
            return iter(case["request"]["results"])

    rec = Recognizer()
    result = execute_model_formulas(plans, recognizer=rec, batch_size=case["batch_size"])
    assert rec.calls == bool(inputs)
    same([{k: v for k, v in p.items() if k != "images"} for p in result], case["expected"])
    same([{k: v for k, v in p.items() if k != "images"} for p in plans], before)

base = copy.deepcopy(next(c["request"]["plans"] for c in f["cases"] if len(c["jobs"]) >= 2))
for p in base:
    p["images"] = {t["source"]: png for t in p["formulas"]}
before = copy.deepcopy(base)


class Bad:
    calls = 0

    def recognize_formula_batch_png(self, images, *, batch_size):
        self.calls += 1
        return ["partial"]


def fails(fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError, RapidDocError):
        return
    raise AssertionError("invalid formula operation succeeded")


bad = Bad()
fails(lambda: execute_model_formulas(base, recognizer=bad))
assert bad.calls == 1
same(base, before)
for batch in [True, 0, 65, "1"]:
    fails(lambda: execute_model_formulas(base, recognizer=bad, batch_size=batch))
fails(lambda: execute_model_formulas(base))
fails(lambda: apply_model_formula_results(base, "x"))
for broken in [b"not PNG", png[:40]]:
    p = copy.deepcopy(base)
    p[0]["images"][p[0]["formulas"][0]["source"]] = broken
    bad = Bad()
    fails(lambda: execute_model_formulas(p, recognizer=bad))
    assert bad.calls == 0
p = copy.deepcopy(base)
del p[-1]["images"][p[-1]["formulas"][-1]["source"]]
bad = Bad()
fails(lambda: execute_model_formulas(p, recognizer=bad))
assert bad.calls == 0


class Stream(io.BytesIO):
    reads = 0

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


stream = Stream(png)
for p in base:
    for key in p["images"]:
        p["images"][key] = stream


class Good:
    def recognize_formula_batch_png(self, images, *, batch_size):
        return [None] * len(images)


result = execute_model_formulas(base, recognizer=Good())
assert stream.reads == 1 and not stream.closed
assert all(isinstance(v, bytes) for p in result for v in p["images"].values())
print(
    "100 actual old formula-stage cases; cross-page order, empty retention, atomic errors and shared streams passed; no formula network inference"
)
