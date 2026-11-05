"""Freeze Python recovery/rendering for an overlapping UNET region capture."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from rapid_doc.model.table.rapid_table_self.table_structure.unet.table_recover import TableRecover
    from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import plot_html_table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    capture = json.loads(args.capture.read_bytes())
    polygons = np.asarray(capture["rotated_polygons"], dtype=np.float64)
    cases = []
    for name, boxes in [("overlapping_regions", polygons),
                        ("translated_regions", polygons + [137, 211]),
                        ("without_overlap", polygons[:-1])]:
        _, points = TableRecover()(boxes)
        texts = [[f"cell-{i}"] for i in range(len(boxes))]
        html = plot_html_table(points, dict(enumerate(texts)), boxes)
        cases.append(dict(name=name, polygons=boxes.tolist(), points=points.tolist(), texts=texts, html=html))
    args.output.write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
