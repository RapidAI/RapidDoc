"""Native CPU policy parameters preserve tiny-model and real formula outputs."""

from pathlib import Path
import json, sys
from rapid_doc_rust import RapidDoc, OnnxSession, RapidDocError
from rapid_doc_rust._runtime import LazySession

root = Path(sys.argv[1]).resolve()
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
f = json.loads(sorted((root / "build/migration/onnx-fixtures").glob("*.json"))[0].read_text())
for options in [
    {},
    {"cpu_spinning": False},
    {"cpu_spinning": True, "cpu_dynamic_block_base": 4},
    {"cpu_spinning": False, "cpu_dynamic_block_base": 4},
]:
    model = OnnxSession(runtime, root / f["model"], f["sha256"], threads=1, **options)
    actual = model.run(f["shape"], f["values"])
    for name, expected in f["outputs"].items():
        assert actual[name]["shape"] == expected["shape"]
        assert all(abs(a - b) < 1e-5 for a, b in zip(actual[name]["values"], expected["values"]))
engine = RapidDoc(cpu_spinning=False, cpu_dynamic_block_base=4)
for model in engine._get_models().values():
    if isinstance(model, LazySession):
        assert model.cpu_spinning is False and model.cpu_dynamic_block_base == 4
        descriptor = json.loads(model.descriptor())
        assert descriptor["cpu_spinning"] is False and descriptor["cpu_dynamic_block_base"] == 4
for kwargs in [
    {"cpu_spinning": "0"},
    {"cpu_spinning": 0},
    {"cpu_dynamic_block_base": True},
    {"cpu_dynamic_block_base": -1},
    {"cpu_dynamic_block_base": 2147483648},
]:
    try:
        RapidDoc(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(kwargs)
f = json.loads((root / "build/migration/formula-decode.json").read_text())
for options in [
    {},
    {"cpu_spinning": False},
    {"cpu_dynamic_block_base": 4},
    {"cpu_force_spinning_stop": True},
]:
    model = OnnxSession(
        runtime, root / f["model"]["path"], f["model"]["sha256"], threads=0, **options
    )
    for case in f["real"]:
        assert (
            model.recognize_formula_batch_png(
                [(root / p).read_bytes() for p in case["sources"]], batch_size=1
            )
            == case["expected"]
        )
print(
    "CPU policies: tiny tensors, real formula results, lazy descriptors and invalid arguments passed"
)

engine = RapidDoc(
    cpu_spinning=False,
    formula_config={"engine_cfg": {"cpu_spinning": True}},
    layout_config={"engine_cfg": {"cpu_dynamic_block_base": 4}},
)
models = engine._get_models()
assert models["formula_model"].cpu_spinning is True
assert models["recognizer"].cpu_spinning is False
assert models["layout_model"].cpu_dynamic_block_base == 4
engine = RapidDoc(formula_config={"engine_cfg": {"cpu_force_spinning_stop": True}})
model = engine._get_models()["formula_model"]
assert model.cpu_force_spinning_stop is True
assert json.loads(model.descriptor())["cpu_force_spinning_stop"] is True
assert engine._get_models()["recognizer"].cpu_force_spinning_stop is None
