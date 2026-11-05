"""在独立 Rust 进程对照字符和矢量几何；不放宽容差来掩盖差异。"""

import argparse
import json
import math
from pathlib import Path
import subprocess
from common import ROOT, digest, write_json
from generate_geometry_corpus import generate
from pdf_corpus import extract_facts


def close(actual, expected, label):
    """以固定 0.001 point 容差比较坐标，不调整顺序或删除字段。"""
    assert len(actual) == len(expected), label
    assert all(math.isfinite(float(a)) and abs(a-b) <= 0.001 for a, b in zip(actual, expected)), (label, actual, expected)


def main():
    """验证初始三份 PDF 加四份 CropBox/Form/旋转样本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT / f"tests/migration/inputs/pdf_{name}.pdf" for name in ("native", "scan", "mixed")]
    paths += generate()
    records = []
    for path in paths:
        expected = extract_facts(path.read_bytes())
        process = subprocess.run([str(args.binary.resolve()), "extract-text", str(path), "--pdfium", str(args.pdfium.resolve()), "--allow-partial"],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
        result = json.loads(process.stdout)["document"]
        assert len(result["pages"]) == len(expected["pages"]), path
        origins = 0
        for actual, golden in zip(result["pages"], expected["pages"]):
            assert actual["text"] == "".join(char["char"] for char in golden["chars"]), path
            assert len(actual["characters"]) == len(golden["chars"]), path
            for char, baseline in zip(actual["characters"], golden["chars"]):
                assert char["is_generated"] == baseline["is_generated"],path
                if baseline.get("origin") is not None:
                    close(char["visual_origin"], baseline["origin"], f"{path.name}:char{char['index']}:origin")
                    origins += 1
                if baseline.get("rotation") == 0:
                    close(char["visual_loose_bounds"], baseline["bbox"], f"{path.name}:char{char['index']}:loose")
            assert len(actual["paths"]) == len(golden["paths"]), path
            for vector, baseline in zip(actual["paths"], golden["paths"]):
                assert len(vector["segments"]) == baseline["segment_count"], path
                assert len(vector["object_path"])-1 == baseline["form_depth"], path
                close(vector["evidence_bounds"], baseline["bbox"], f"{path.name}:path{vector['object_path']}")
                assert vector["fill_visible"] == baseline["fill_visible"], path
                assert vector["stroke_visible"] == baseline["stroke_visible"], path
            assert len(actual["images"]) == len(golden["image_bboxes"]), path
            for image, baseline in zip(actual["images"], golden["image_bboxes"]):
                close(image["visual_bounds"], baseline, f"{path.name}:image")
            assert len(actual["links"]) == len(golden["links"]), path
            for link, baseline in zip(actual["links"],golden["links"]):
                close(link["visual_bounds"],baseline["bboxes"][0],f"{path.name}:link")
                assert link["uri"] == baseline["target"],path
            assert len(actual["drawing_lines"]) == len(golden["drawing_lines"]), path
            for line, baseline in zip(actual["drawing_lines"],golden["drawing_lines"]):
                assert line["orientation"] == baseline["orientation"],path
                close(line["start"],baseline["start"],f"{path.name}:line-start")
                close(line["end"],baseline["end"],f"{path.name}:line-end")
                close(line["bounds"],baseline["bbox"],f"{path.name}:line-bounds")
                close([line["width"]],[baseline["width"]],f"{path.name}:line-width")
        records.append({"id":path.stem,"sha256":digest(path.read_bytes()),"pages":len(result["pages"]),"origins_compared":origins})
    write_json(ROOT / "build/pdf-geometry-report.json", {"absolute_tolerance_pt":0.001,"cases":records,
        "scope":"Generated flags, character origins, zero-rotation loose boxes, path bounds/count/visibility/depth, image bounds, URI link rects, merged drawing line order/bounds/width; arbitrary clipping curves and all font metrics not yet covered"})
    print(f"PDF geometry differential: {len(records)} documents passed")


if __name__ == "__main__":
    main()
