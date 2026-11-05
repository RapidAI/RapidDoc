"""PP-DocLayoutV3 前处理逐位差分，不带 NumPy/OpenCV。"""

from array import array
from pathlib import Path
import json
import sys
import hashlib
from rapid_doc_rust._native import prepare_layout_rgb_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/layout-preprocess.json").read_text(encoding="utf-8"))
for case in fixture["cases"]:
    data = bytes(case["rgb"]) if "rgb" in case else (root / case["rgb_file"]).read_bytes()
    tensor = json.loads(
        prepare_layout_rgb_json(data, case["width"], case["height"], case["target"], case["target"])
    )
    assert tensor["shape"] == case["shape"]
    actual = array("f", tensor["values"])
    if "fp32_le_sha256" in case:
        if sys.byteorder != "little":
            actual.byteswap()
        assert hashlib.sha256(actual.tobytes()).hexdigest() == case["fp32_le_sha256"], case["id"]
        continue
    expected = array("f", case["values"])
    failures = [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    assert actual.tobytes() == expected.tobytes(), (case["id"], len(failures), failures[:8])
print(f"Installed wheel: {len(fixture['cases'])} PP-DocLayoutV3 RGB/cubic/NCHW tensors bit-exact")
