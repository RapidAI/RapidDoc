"""Differential seal checks against the original Python algorithms and models.

Uses generated geometry (including concave, curved, vertical and invalid rings)
and the existing real seal image. Nothing is tuned to a particular PDF.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def crop_fixtures(np, cv2):
    yy, xx = np.mgrid[:384, :512]
    pixels = np.stack([(xx*7+yy*3) % 256, (xx*11+yy*13) % 256, (xx*2+yy*17) % 256], axis=2).astype(np.uint8)
    shapes = [('rectangle', np.array([[40, 40], [290, 40], [290, 94], [40, 94]], np.int32)),
              ('rotated', cv2.boxPoints(((245., 170.), (210., 40.), 21.)).astype(np.int32)),
              ('invalid', np.array([[40, 40], [290, 94], [290, 40], [40, 94]], np.int32))]
    for count in [8, 16, 31]:
        for angle in [0, 90, 22]:
            theta = np.linspace(-np.pi+.12, -.12, count)
            curve = np.r_[np.c_[145*np.cos(theta), 145*np.sin(theta)],
                          np.c_[96*np.cos(theta[::-1]), 96*np.sin(theta[::-1])]]
            rad = angle*np.pi/180
            curve = curve @ np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
            shapes.append((f'curve-{count}-{angle}', np.rint(curve+[256, 192]).astype(np.int32)))
    return pixels, shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()
    if args.sdk_root:
        sys.path.insert(0, str(args.sdk_root.resolve()))
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    import pyclipper
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapid_doc.model.ocr.seal_crop import CropByPolys, SortPolyBoxes
    from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess
    from rapid_doc_rust import OnnxSession
    from rapid_doc_rust import _native
    from rapid_doc_rust._runtime import geos_runtime, runtime_path

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(sdk=str(Path(_native.__file__).resolve()), cases=[])

    def native(operation, **values):
        return json.loads(_native.seal_geometry_json(json.dumps(dict(operation=operation, **values), allow_nan=False)))

    def add(kind, name, ok, **details):
        report['cases'].append(dict(kind=kind, name=name, passed=bool(ok), **details))

    rng = np.random.default_rng(20261030)
    # Convex and concave shapes, both orientations, varying round offsets.
    polygons = [np.array([[10, 10], [90, 10], [90, 50], [10, 50]], np.int32),
                np.array([[10, 10], [90, 10], [90, 30], [40, 30], [40, 70], [10, 70]], np.int32)]
    for _ in range(24):
        theta = np.sort(rng.uniform(0, 2*np.pi, int(rng.integers(5, 24))))
        radius = rng.uniform(20, 100, len(theta))
        polygons.append(np.rint(np.c_[radius*np.cos(theta)+120, radius*np.sin(theta)+120]).astype(np.int32))
    for i, points in enumerate(polygons):
        for ratio in [0.1, 0.5, 1.8]:
            perimeter = cv2.arcLength(points, True)
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expected = offset.Execute(cv2.contourArea(points)*ratio/perimeter)
            expected = expected[0] if len(expected) == 1 else None
            actual = native('unclip', points=points.tolist(), ratio=ratio)
            add('unclip', f'{i}/{ratio}', expected == actual)

    post = DBPostProcess(thresh=.2, box_thresh=.6, unclip_ratio=.5, use_dilation=False, box_type='poly')
    sorter = SortPolyBoxes()
    maps = [np.zeros((128, 192), np.float32)]
    rectangular = np.zeros((128, 192), np.float32)
    rectangular[12:34, 6:90] = .9
    rectangular[60:88, 108:183] = .7
    maps.append(rectangular)
    for _ in range(12):
        mask = np.zeros((128, 192), np.uint8)
        for _ in range(3):
            center = tuple(map(int, rng.integers([0, 0], [192, 128])))
            cv2.ellipse(mask, center, (int(rng.integers(10, 50)), int(rng.integers(8, 30))),
                        int(rng.integers(0, 180)), 0, 310, 1, int(rng.integers(4, 12)))
        maps.append(mask.astype(np.float32)*.9)
    for i, values in enumerate(maps):
        expected, scores = post(values[None, None], (239, 347))
        expected = [p.tolist() for p in sorter(expected)] if len(expected) else []
        actual = native('detection', values=values.ravel().tolist(), width=values.shape[1],
                        height=values.shape[0], source_width=347, source_height=239)
        add('detection', str(i), expected == [b['points'] for b in actual],
            expected_boxes=len(expected), actual_boxes=len(actual),
            expected=expected if expected != [b['points'] for b in actual] else None,
            actual=actual if expected != [b['points'] for b in actual] else None)

    cropper = CropByPolys(det_box_type='poly')
    pixels, shapes = crop_fixtures(np, cv2)
    for name, points in shapes:
        expected = cropper.get_poly_rect_crop(pixels, points)
        actual = native('crop', bgr=pixels.ravel().tolist(), width=512, height=384,
                        points=points.tolist(), geos_library=geos_runtime())
        actual = np.array(actual['bgr'], np.uint8).reshape(actual['height'], actual['width'], 3)
        same_shape = actual.shape == expected.shape
        diff = np.abs(actual.astype(np.int16)-expected.astype(np.int16)) if same_shape else None
        add('crop', name, same_shape and not np.any(diff), expected_shape=list(expected.shape),
            actual_shape=list(actual.shape), different_values=int(np.count_nonzero(diff)) if same_shape else None,
            max_error=int(diff.max()) if same_shape else None)
        np.savez_compressed(args.output_dir/f'{name}.npz', input=pixels, points=points,
                            expected=expected, actual=actual)

    if args.models:
        # Same detector, same CPU thread count and recognition model; include all
        # text lines, not only a selected successful stamp.
        original = RapidOcrModel(is_seal=True, ocr_config={
            'EngineConfig.onnxruntime.intra_op_num_threads': 1,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1})
        source = ROOT/'demo/images/seal_text_det.png'
        pixels = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        expected = original.ocr(pixels, det=True, rec=True)[0]
        path = ROOT/'rapid_doc/resources/pp-ocrv4_mobile_seal_det.onnx'
        detector = OnnxSession(str(runtime_path()), str(path), hashlib.sha256(path.read_bytes()).hexdigest(), threads=1)
        boxes = detector.detect_seal_png(source.read_bytes())
        crops = []
        for item in boxes:
            result = native('crop', bgr=pixels.ravel().tolist(), width=pixels.shape[1], height=pixels.shape[0],
                            points=item['points'], geos_library=geos_runtime())
            image = np.array(result['bgr'], np.uint8).reshape(result['height'], result['width'], 3)
            ok, encoded = cv2.imencode('.png', image)
            assert ok
            crops.append(encoded.tobytes())
        path = ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx'
        recognizer = OnnxSession(str(runtime_path()), str(path), hashlib.sha256(path.read_bytes()).hexdigest(), threads=1)
        rec = recognizer.recognize_batch_png(crops, batch_size=6) if crops else []
        accepted = [(box, result) for box, result in zip(boxes, rec) if result['confidence'] >= .5]
        actual_text = [r['text'] for _, r in accepted]
        expected_text = [r[1][0] for r in expected or []]
        expected_boxes = [r[0] for r in expected or []]
        add('model', source.name, expected_text == actual_text and expected_boxes == [b['points'] for b, _ in accepted],
            expected_text=expected_text, actual_text=actual_text, detected_regions=len(boxes))

    report['passed'] = sum(c['passed'] for c in report['cases'])
    report['total'] = len(report['cases'])
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], total=report['total'], failures=[c for c in report['cases'] if not c['passed']]), ensure_ascii=False))
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
