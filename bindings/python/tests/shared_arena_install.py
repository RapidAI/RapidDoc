"""Shared allocator lifetime, separate model state, policies and threaded use."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import sys
import rapid_doc_rust as sdk
from rapid_doc_rust._runtime import LazySession

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/shared-arena-fixture-20261029/models.json").read_text())
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"


def session(index, shared):
    f = fixture[index]
    return sdk.OnnxSession(
        runtime, root / f["path"], f["sha256"], threads=1, cpu_shared_arena=shared
    )


def check(model, multiplier):
    for _ in range(5):
        result = model.run([1, 4], [1.0, 2.0, 3.0, 4.0])["y"]
        assert result["shape"] == [1, 4] and result["values"] == [
            v * multiplier for v in [1.0, 2.0, 3.0, 4.0]
        ]


for _ in range(3):
    a = session(0, True)
    b = session(1, True)
    separate = session(0, False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [
            pool.submit(check, a, 2),
            pool.submit(check, b, 3),
            pool.submit(check, separate, 2),
        ]
        for task in tasks:
            task.result()
    del a
    gc.collect()
    check(b, 3)
    del b
    gc.collect()
    check(separate, 2)
    del separate
    gc.collect()
engine = sdk.RapidDoc(layout_config={"engine_cfg": {"cpu_shared_arena": True}})
models = engine._get_models()
assert models["layout_model"].cpu_shared_arena is True
assert json.loads(models["layout_model"].descriptor())["cpu_shared_arena"] is True
assert models["formula_model"].cpu_shared_arena is None
for bad in ["1", 1, False + 1]:
    try:
        sdk.RapidDoc(layout_config={"engine_cfg": {"cpu_shared_arena": bad}})._get_models()
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
print(
    "Shared arena: separate weights, concurrent sessions, last-session release/re-registration, descriptors and invalid options passed"
)
