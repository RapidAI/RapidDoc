"""在无 OpenCV 的隔离 wheel 验证 8 连通与外接框。"""

from array import array
from pathlib import Path
import json, sys, hashlib
from rapid_doc_rust._native import table_components_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-components.json").read_text())
for case in fixture["cases"]:
    result = json.loads(
        table_components_json(
            bytes(case["mask"]), case["width"], case["height"], case["axis"], case["line_extent"]
        )
    )
    values = array(
        "f",
        [v for line in result["lines"] for v in line]
        + [v for p in result["polygons"] for xy in p for v in xy],
    )
    if sys.byteorder != "little":
        values.byteswap()
    assert hashlib.sha256(values.tobytes()).hexdigest() == case["sha256"], (
        case["id"],
        result,
        case["lines"],
        case["polygons"],
    )
print(
    f"Installed wheel: {len(fixture['cases'])} connected-component line/region contracts bit-exact"
)
