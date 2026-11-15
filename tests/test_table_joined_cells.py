import numpy as np
import pytest

from rapid_doc.model.table.rapid_table_self.table_structure.unet.unet_table_rec import (
    UnetTableRecognition,
)


@pytest.mark.parametrize("case", ["missing", "occupied", "merged", "disagreement"])
def test_joined_ocr_only_fills_verified_empty_cells(monkeypatch, case):
    model = UnetTableRecognition.__new__(UnetTableRecognition)
    first_cell = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    polygons = np.array([first_cell + [10 * col, 0] for col in range(4)])
    logic = np.array([[0, 0, col, col] for col in range(4)])
    ocr_box = np.array([[0, 2], [40, 2], [40, 8], [0, 8]])
    matched = {}
    if case == "occupied":
        matched[1] = [[polygons[1], "existing", 0.99]]
    if case == "merged":
        logic[1, 3] = 2

    def recognize_cells(img, boxes, cell_map):
        assert case in ("missing", "disagreement")
        return {i: [[box, text, 0.98]] for i, (box, text) in enumerate(
            zip(boxes, ["A", "", "C", "D"])
        )}

    monkeypatch.setattr(model, "fill_blank_rec", recognize_cells)
    joined_text = "AXED" if case == "disagreement" else "ABCD"
    model.recover_joined_cells(None, polygons, logic, matched, [(ocr_box, joined_text, 0.99)])

    assert {i: result[0][1] for i, result in matched.items()} == {
        "missing": dict(enumerate("ABCD")),
        "occupied": {1: "existing"},
        "merged": {},
        "disagreement": {},
    }[case]


@pytest.mark.parametrize("case", ["split", "no_line", "occupied", "merged", "disagreement", "protected"])
def test_cross_cell_recognition_requires_confirmed_boundary_and_text(monkeypatch, case):
    model = UnetTableRecognition.__new__(UnetTableRecognition)
    model.ocr_engine = object()
    polygons = np.array([
        [[0, 0], [40, 0], [40, 40], [0, 40]],
        [[40, 0], [80, 0], [80, 40], [40, 40]],
    ])
    logic = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    image = np.full((40, 80, 3), 255, np.uint8)
    if case != "no_line":
        image[:, 39:41] = 0
    box = np.array([[18, 14], [62, 14], [62, 26], [18, 26]])
    tokens = [(box, "甲1", .99)]
    if case == "occupied":
        tokens.append((box - [16, 0], "其他", .99))
    if case == "merged":
        logic[0, 3] = 1

    def recognize_cells(img, boxes, cell_map):
        assert case in ("split", "disagreement")
        text = "1" if case == "split" else "2"
        return {0: [[boxes[0], "甲", .99]], 1: [[boxes[1], text, .99]]}

    monkeypatch.setattr(model, "fill_blank_rec", recognize_cells)
    result = model.recognize_cross_cell_text(
        image, polygons, logic, tokens, protected_token_count=int(case == "protected")
    )
    assert {i: items[0][1] for i, items in result.get(0, {}).items()} == (
        {0: "甲", 1: "1"} if case == "split" else {}
    )
