import numpy as np
import pytest

from collections import Counter

from rapid_doc.model.table.fallback import keeps_candidate_numbers, quality, structure_evidence
from rapid_doc.model.table.table_evidence import TableEvidence, crop_vector_lines


def crop(draw=False):
    image = np.full((120, 140, 3), 255, dtype=np.uint8)
    if draw:
        image[4:112, 67:69] = 0
    boxes = [[(x, y), (x + 24, y), (x + 24, y + 12), (x, y + 12)]
             for y in (15, 50, 85) for x in (12, 87)]
    rows = [boxes, ['item', '42'] * 3, [.99] * 6]
    return image, rows


def grids():
    boxes = [[0, y, 68, y + 35] if col == 0 else [68, y, 140, y + 35]
             for y in (3, 38, 73) for col in (0, 1)]
    split = (boxes, [[row, row, col, col] for row in range(3) for col in (0, 1)])
    merged = ([[0, y, 140, y + 35] for y in (3, 38, 73)],
              [[row, row, 0, 0] for row in range(3)])
    return split, merged


def test_weak_numeric_assignment_does_not_waive_other_instances():
    image, rows = crop()
    split, _ = grids()
    boxes, logic = split
    boxes[1] = [68, 3, 140, 16]  # 仅覆盖第一个 42 的上缘
    source = TableEvidence(image, rows)
    assigned = {i: (i,) for i in range(6)}
    primary = source.score(split, assigned)
    assert primary.weak_numbers == {1}
    candidate = source.score(split, {i: cell for i, cell in assigned.items() if i != 1})
    assert keeps_candidate_numbers(Counter({'42': 3}), Counter({'42': 2}), rows, primary, candidate)
    assert not keeps_candidate_numbers(Counter({'42': 3}), Counter({'42': 1}), rows, primary, candidate)
    candidate = source.score(split, {i: cell for i, cell in assigned.items() if i not in {1, 3}})
    assert not keeps_candidate_numbers(Counter({'42': 3}), Counter({'42': 1}), rows, primary, candidate)


@pytest.mark.parametrize('draw', [False, True])
def test_same_evidence_detects_merged_columns_in_wireless_and_wired_table(draw):
    split, merged = grids()
    evidence = TableEvidence(*crop(draw))
    wrong = evidence.score(merged, {i: (i // 2,) for i in range(6)})
    right = evidence.score(split, {i: (i,) for i in range(6)})
    text, lines, gaps = right.improvement_over(wrong)
    assert wrong.suspicious and not right.suspicious
    assert text == 0 and gaps > 0
    assert (lines > 0) == draw
    assert (right.line_improvement(wrong, 0) > 0) == draw
    assert right.line_improvement(wrong, 1) == 0


def test_partial_ruling_supports_only_its_covered_rows():
    split, merged = grids()
    partial = (merged[0][:1] + split[0][2:], merged[1][:1] + split[1][2:])
    evidence = TableEvidence(*crop(), vector_lines=[(0, 68, 38, 108)])
    assert sum(evidence.score(partial).lines.values()) == 0
    assert sum(evidence.score(merged).lines.values()) > 0


def test_local_line_comparison_ignores_shared_segments():
    split, _ = grids()
    shifted = ([box.copy() for box in split[0]], split[1])
    for box in shifted[0][:4]:
        box[2 if box[0] == 0 else 0] = 50
    evidence = TableEvidence(*crop(), vector_lines=[(0, 68, 3, 108)])
    correct, wrong = evidence.score(split), evidence.score(shifted)
    assert correct.local_line_improvement(wrong) > correct.line_improvement(wrong, 0) > 0
    assert correct.local_line_improvement(correct) == 0


def test_empty_row_count_includes_disjoint_rows_but_excludes_rowspan():
    html = ('<table><tr><td rowspan="2">A</td><td></td></tr><tr><td></td></tr>'
            '<tr><td></td><td></td></tr><tr><td>A</td><td></td></tr>'
            '<tr><td></td><td></td></tr></table>')
    evidence = structure_evidence(html, [[], [], []])
    assert (evidence.empty_rows, evidence.empty_row_run) == (2, 1)


def test_numeric_retention_uses_token_ids_for_duplicate_values():
    split, _ = grids()
    evidence = TableEvidence(*crop())
    score = evidence.score(split, {i: (i,) for i in range(6) if i != 3})
    assert score.missing_numbers == {3}
    assert evidence.score(split).missing_numbers == set()


def test_invalid_grid_preserves_cell_text():
    _, rows = crop()
    html = '<table><tr><td>42</td></tr><tr><td>item</td><td>42</td></tr></table>'
    result = quality(html, rows)
    assert not result.valid
    assert result.retained == 8


@pytest.mark.parametrize('rotation, expected', [
    ('0', (1, 50, 4, 64)), ('90', (0, 50, 36, 96)), ('270', (0, 30, 4, 64)),
])
def test_vector_lines_follow_crop_scale_padding_and_rotation(rotation, expected):
    lines = [dict(x1=10, y1=33, x2=40, y2=33)]
    crop = (4, 4, 20, 20, 112, 92, 100, 80)
    assert crop_vector_lines(lines, crop, 2, rotation) == [expected]


def test_stacked_tables_require_distinct_repeated_headers_and_no_crossing_rule():
    image = np.full((260, 300, 3), 255, np.uint8)
    horizontal = [(1, y, 0, 300) for y in (5, 26, 125, 129, 150, 250)]
    labels = []
    boxes = []
    for y, groups in ((17, [(30, 80), (110, 160), (190, 240)]),
                      (141, [(30, 100), (150, 240)])):
        for lo, hi in groups:
            horizontal.append((1, y - 3, lo, hi))
            for x, text in ((lo + 4, 'Value'), (lo + 25, 'Rate')):
                boxes.append([[x, y], [x+10, y], [x+10, y+8], [x, y+8]])
                labels.append(text)
    rows = [boxes, labels, [1.] * len(labels)]
    evidence = TableEvidence(image, rows, vector_lines=horizontal)
    assert evidence.stacked_table_boundary(rows, *image.shape[:2]) == 127
    evidence = TableEvidence(image, rows, vector_lines=[*horizontal, (0, 120, 5, 250)])
    assert evidence.stacked_table_boundary(rows, *image.shape[:2]) is None
