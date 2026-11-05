"""独立wheel：真实旧PP-FormulaNet输入像素/张量，无NumPy/Pillow/OpenCV依赖。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import concurrent.futures, hashlib, io, json, struct, sys
from pathlib import Path
from rapid_doc_rust import prepare_formula_input, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/formula-preprocess.json").read_text(encoding="utf-8"))
assert len(fixture["cases"]) == 88
for path, sha in fixture["references"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
maximum = 0.0
for i, c in enumerate(fixture["cases"]):
    png = (root / c["source"]).read_bytes()
    out = prepare_formula_input(png, height=c["target_size"][0], width=c["target_size"][1])
    assert out["shape"] == c["shape"] and out["rgb_size"] == list(reversed(c["target_size"]))
    assert out["layout"] == "NCHW" and out["dtype"] == "float32-le"
    assert [out["crop_box"][2] - out["crop_box"][0], out["crop_box"][3] - out["crop_box"][1]] == c[
        "cropped_size"
    ]
    assert hashlib.sha256(out["rgb"]).hexdigest() == c["rgb_sha256"], i
    expected = (root / c["tensor"]).read_bytes()
    assert hashlib.sha256(expected).hexdigest() == c["tensor_sha256"] and len(out["tensor"]) == len(
        expected
    )
    err = max(
        abs(a[0] - b[0])
        for a, b in zip(struct.iter_unpack("<f", out["tensor"]), struct.iter_unpack("<f", expected))
    )
    assert err <= 1e-6, (i, err)
    maximum = max(maximum, err)
    if c["target_size"] == [384, 384]:
        assert prepare_formula_input(png)["tensor"] == out["tensor"]
    if i == 0:
        values = [x[0] for x in struct.iter_unpack("<f", out["tensor"])]
        assert values[31 * 48 :] == [1.0] * 48
        assert all(values[y * 48 + 47] == 1.0 for y in range(31))

source = (root / fixture["cases"][1]["source"]).read_bytes()
stream = io.BytesIO(source)
prepare_formula_input(stream, height=32, width=48)
assert not stream.closed


def fails(call):
    try:
        call()
    except (ValueError, TypeError, RapidDocError):
        return
    raise AssertionError("invalid formula preprocess succeeded")


for size in [0, 1, 2049, True, 2.5]:
    fails(lambda: prepare_formula_input(source, height=size))
for png in [b"bad", source[:40]]:
    fails(lambda: prepare_formula_input(png))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    outputs = list(pool.map(lambda _: prepare_formula_input(source, height=31, width=47), range(6)))
assert all(x == outputs[0] for x in outputs)
assert capabilities()["formula_preprocessing"] and capabilities()["formula_full_ftfy"]
print(
    f"88 old formula preprocessing cases: exact RGB; max FP32 absolute error {maximum:.9g}; dimensions, ownership, budgets and concurrency passed; no formula network inference"
)
