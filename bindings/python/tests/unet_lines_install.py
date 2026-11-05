"""UNET 从类别图到连接线段，不把线段输出宣称为完整表格。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from array import array
from pathlib import Path
import hashlib, json, sys
from rapid_doc_rust import table_line_geometry, RapidDocError
from rapid_doc_rust._native import prepare_unet_line_masks_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/unet-lines.json").read_text())
for case in fixture["cases"]:
    result = json.loads(
        prepare_unet_line_masks_json(
            bytes(case["mask"]),
            case["width"],
            case["height"],
            case["source_width"],
            case["source_height"],
            case["morph_close"],
        )
    )
    assert (
        hashlib.sha256(bytes(result["horizontal"]) + bytes(result["vertical"])).hexdigest()
        == case["masks_sha256"]
    ), (case["id"], "mask")
    lines = table_line_geometry(
        dict(width=case["width"], height=case["height"], classes=case["mask"]),
        source_width=case["source_width"],
        source_height=case["source_height"],
        morph_close=case["morph_close"],
    )
    values = array("f", [v for line in lines["rows"] + lines["columns"] for v in line])
    if sys.byteorder != "little":
        values.byteswap()
    assert hashlib.sha256(values.tobytes()).hexdigest() == case["lines_sha256"], (
        case["id"],
        lines,
        case["rows"],
        case["columns"],
    )
for mask, w, h, code in [
    (dict(width=1, height=1, classes=[3]), 1, 1, "invalid_input"),
    (dict(width=1, height=1, classes=[0]), 4096, 4096, "resource_limit"),
]:
    try:
        table_line_geometry(mask, source_width=w, source_height=h)
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid UNET line mask accepted")
print(
    f"Installed wheel: {len(fixture['cases'])} UNET resized/closed masks and connected lines bit-exact"
)
