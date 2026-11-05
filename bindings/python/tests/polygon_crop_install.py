"""Native crop masking versus the preserved original cv2.fillPoly adapter."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy
import importlib.util
import io
import json
import sys
from pathlib import Path

from PIL import Image
from rapid_doc_rust import prepare_model_crops
from rapid_doc_rust._native import prepare_model_crops_bytes
from rapid_doc_rust._host_context import host_call

root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(
    "crop_reference", root / "scripts/migration/references/polygon_crops.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
image = Image.new("RGB", (160, 120))
image.putdata(
    [((x * 19) % 256, (y * 11) % 256, (x + y) % 256) for y in range(120) for x in range(160)]
)
buffer = io.BytesIO()
image.save(buffer, format="PNG")
png = buffer.getvalue()
polygons = [
    [[15, 12], [130, 12], [130, 90], [15, 90]],
    [[30, 12], [130, 28], [115, 90], [15, 74]],
    [[15, 12], [130, 90], [15, 90], [130, 12]],
    [[15, 12], [80, 35], [130, 12], [100, 65], [130, 90], [15, 90]],
    [[15.9, 12.8], [130.1, 32.9], [75.7, 90.3]],
    [[-5, 8], [140, 12], [130, 110], [5, 90]],
]


@host_call
def baseline(model, expansion):
    request = json.dumps(dict(model=model, page_index=0, formula_bbox_expand_px=expansion))
    plan, images = prepare_model_crops_bytes(request, png, mask_polygons=False)
    result = json.loads(plan)
    result["images"] = dict(images)
    return reference.mask_crops(result, expansion)


@host_call
def native(model, expansion):
    return prepare_model_crops(png, model, formula_bbox_expand_px=expansion)


count = 0
for polygon in polygons:
    for category in [1, 14, 5]:
        for expansion in [0, 2]:
            model = {
                "layout_dets": [
                    {
                        "category_id": category,
                        "score": 0.99,
                        "poly": [15, 12, 130, 12, 130, 90, 15, 90],
                        "polygon_points": polygon,
                    }
                ]
            }
            a = baseline(copy.deepcopy(model), expansion)
            b = native(copy.deepcopy(model), expansion)
            assert {k: v for k, v in a.items() if k != "images"} == {
                k: v for k, v in b.items() if k != "images"
            }
            assert a["images"].keys() == b["images"].keys()
            for key in a["images"]:
                with (
                    Image.open(io.BytesIO(a["images"][key])) as old,
                    Image.open(io.BytesIO(b["images"][key])) as new,
                ):
                    assert (old.mode, old.size, old.tobytes()) == (
                        new.mode,
                        new.size,
                        new.tobytes(),
                    ), (category, expansion, polygon)
            count += 1
print(f"Original OpenCV polygon crop pixels and task metadata passed: {count} cases")
