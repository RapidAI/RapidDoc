"""Single-table TATR ONNX adapter; runtime requires no torch/transformers."""
from copy import deepcopy
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from ...inference_engine.base import get_engine
from ...utils.typings import EngineType
from . import structure

LABELS = ("table", "table column", "table row", "table column header",
          "table projected row header", "table spanning cell", "no object")
THRESHOLDS = {label: .5 for label in LABELS}
THRESHOLDS["no object"] = 10


def preprocess(image):
    """Microsoft src/inference.py structure_transform: max 1000, PIL bicubic."""
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Table Transformer requires a nonempty HWC BGR image")
    height, width = image.shape[:2]
    scale = 1000 / max(width, height)
    rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
    pixels = np.asarray(rgb, dtype=np.float32) / 255
    pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def ocr_to_tokens(ocr):
    if ocr is None:
        return []
    boxes, texts, scores = ocr
    if not len(boxes) == len(texts) == len(scores):
        raise ValueError("Table Transformer OCR boxes/texts/scores length mismatch")
    tokens = []
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        if x1 > x0 and y1 > y0:
            tokens.append(dict(bbox=[float(x0), float(y0), float(x1), float(y1)], text=str(text),
                               block_num=0, line_num=idx, span_num=0))
    return tokens


def cells_to_result(cells, image_size):
    table = ET.Element("table")
    # Rows fully occupied by rowspan still need a <tr>; otherwise the following
    # row shifts upward and HTML disagrees with the returned logical coordinates.
    rows = {r: ET.SubElement(table, "tr")
            for r in range(max((max(c["row_nums"]) for c in cells), default=-1)+1)}
    boxes, logic = [], []
    for cell in sorted(cells, key=lambda c: (min(c["row_nums"]), min(c["column_nums"]))):
        r0, r1 = min(cell["row_nums"]), max(cell["row_nums"])
        c0, c1 = min(cell["column_nums"]), max(cell["column_nums"])
        attrs = {}
        if r1 > r0:
            attrs["rowspan"] = str(r1-r0+1)
        if c1 > c0:
            attrs["colspan"] = str(c1-c0+1)
        node = ET.SubElement(rows[r0], "th" if cell.get("column header") else "td", attrs)
        node.text = cell.get("cell text", "")
        boxes.append(np.clip(cell["bbox"], [0, 0, 0, 0], [*image_size, *image_size]))
        logic.append([r0, r1, c0, c1])
    return (ET.tostring(table, encoding="unicode", short_empty_elements=False),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(logic, dtype=np.int64).reshape(-1, 4))


def outputs_to_objects(logits, boxes, image_size):
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    if logits.shape != (1, 125, 7) or boxes.shape != (1, 125, 4):
        raise ValueError(f"Unexpected Table Transformer output shapes: {logits.shape}, {boxes.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise ValueError("Table Transformer outputs contain nonfinite values")
    probs = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    labels, scores = probs.argmax(axis=-1), probs.max(axis=-1)
    xy0, xy1 = boxes[0, :, :2] - boxes[0, :, 2:]/2, boxes[0, :, :2] + boxes[0, :, 2:]/2
    xyxy = np.concatenate((xy0, xy1), axis=-1) * np.array([*image_size, *image_size], np.float32)
    # Match upstream inference: skip no-object; individual structure thresholds
    # are applied by its refinement functions, not globally here.
    return [dict(label=LABELS[label], score=float(score), bbox=box.tolist())
            for label, score, box in zip(labels, scores, xyxy) if LABELS[label] != "no object"]


def decode(logits, boxes, image_size, tokens):
    objects = outputs_to_objects(logits, boxes, image_size)
    tables = [obj for obj in objects
              if obj["label"] == "table" and obj["score"] >= THRESHOLDS["table"]]
    if not tables:
        return cells_to_result([], image_size)
    # The caller supplies one table crop; preserve the largest predicted table.
    table = max(tables, key=lambda t: max(0, t["bbox"][2]-t["bbox"][0]) * max(0, t["bbox"][3]-t["bbox"][1]))
    selected = [obj for obj in objects if obj["label"] != "table"] + [table]
    structures = structure.objects_to_structures(deepcopy(selected), deepcopy(tokens), THRESHOLDS)
    if not structures or not structures[0]["rows"] or not structures[0]["columns"]:
        return cells_to_result([], image_size)
    cells, _ = structure.structure_to_cells(structures[0], deepcopy(tokens))
    return cells_to_result(cells, image_size)


class TableTransformerRecognition:
    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["engine_type"] = cfg.get("engine_type") or EngineType.ONNXRUNTIME
        if isinstance(cfg["engine_type"], str):
            cfg["engine_type"] = EngineType(cfg["engine_type"])
        if cfg["engine_type"] != EngineType.ONNXRUNTIME:
            raise ValueError("Table Transformer currently supports ONNXRUNTIME only")
        self.session = get_engine(cfg["engine_type"])(cfg)
        if self.session.get_output_names() != ["logits", "pred_boxes"]:
            raise ValueError("Use scripts/export_table_transformer.py to export the Table Transformer model")

    def __call__(self, images, ocr_results):
        if len(images) != len(ocr_results):
            raise ValueError("Table Transformer image/OCR batch length mismatch")
        htmls, boxes, logic = [], [], []
        for image, ocr in zip(images, ocr_results):
            result = decode(*self.session(preprocess(image)), (image.shape[1], image.shape[0]), ocr_to_tokens(ocr))
            htmls.append(result[0])
            boxes.append(result[1])
            logic.append(result[2])
        return htmls, boxes, logic
