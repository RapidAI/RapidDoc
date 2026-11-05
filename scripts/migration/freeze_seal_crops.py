"""Freeze original Python seal crops and homographies for focused native tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from verify_seal_runtime import crop_fixtures

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import rapid_doc.model.ocr.seal_det_warp as warp
    from rapid_doc.model.ocr.seal_crop import CropByPolys
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pixels, shapes = crop_fixtures(np, cv2)
    (args.output_dir/'input.bgr').write_bytes(pixels.tobytes())
    source = ROOT/'rapid_doc/model/ocr'
    reference = dict(opencv=cv2.__version__, numpy=np.__version__,
                     sources={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                              for name in ['seal_crop.py', 'seal_det_warp.py', 'ocr_patch.py']},
                     width=pixels.shape[1], height=pixels.shape[0], cases=[])
    cropper = CropByPolys(det_box_type='poly')
    original = warp.Homography
    plans = []

    def capture(image, points, width, height, *args, **kwargs):
        dest = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        matrix = cv2.getPerspectiveTransform(np.array(points, np.float32).reshape(4, 2), dest)
        plans.append(dict(points=np.array(points).tolist(), world_w=float(width), world_h=float(height),
                          inverse=cv2.invert(matrix)[1].astype(np.float32).ravel().tolist()))
        return original(image, points, width, height, *args, **kwargs)

    warp.Homography = capture
    try:
        for name, points in shapes:
            plans.clear()
            expected = cropper.get_poly_rect_crop(pixels, points)
            (args.output_dir/(name+'.expected.bgr')).write_bytes(expected.tobytes())
            reference['cases'].append(dict(name=name, points=points.tolist(), expected_width=expected.shape[1],
                                           expected_height=expected.shape[0], patches=list(plans)))
    finally:
        warp.Homography = original
    (args.output_dir/'cases.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
    print(json.dumps(dict(opencv=reference['opencv'], cases=len(reference['cases']))))


if __name__ == '__main__':
    main()
