"""Weak candidates may donate proven annotations without growing a complete grid."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TableAnnotation, _TableCandidate


def candidates(angle=0, kind="caption", core=True):
    def box(value):
        return _rotate_bbox_from_upright(value, (800., 900.), angle)

    target = _TableCandidate(box((40., 50., 540., 250.)), (40., 50., 540., 250.), angle, 111.,
                             core_bbox=box((40., 80., 540., 200.)) if core else None,
                             line_indices={1, 2}, inferred_grid_authoritative=True)
    weak = _TableCandidate(box((40., 100., 540., 700.)), (40., 100., 540., 700.), angle, 18.,
                           core_bbox=box((40., 100., 540., 700.)), line_indices={2, 3, 4})
    # The caption's font box slightly intersects the top border, but its center is outside.
    valid = (100., 60., 250., 80.3) if kind == "caption" else (100., 210., 250., 230.)
    rows = {9: box(valid), 1: box((100., 90., 250., 110.)),
            10: box((100., 115., 250., 125.)), 11: box((100., 260., 250., 280.)),
            12: box((20., 60., 250., 75.))}
    weak.annotations = [_TableAnnotation(kind, box((20., 60., 250., 280.)), set(rows) | {13}, rows)]
    return target, weak


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("kind", ["caption", "footnote"])
def test_enclosed_annotation_survives_without_growing_complete_grid(angle, kind):
    target, weak = candidates(angle, kind)
    original = deepcopy(asdict(target))
    weak_before = deepcopy(asdict(weak))
    merged = table_rules._merge_table_candidates([weak, target])
    assert len(merged) == 1
    actual = asdict(merged[0])
    assert {key: value for key, value in actual.items() if key != "annotations"} == {
        key: value for key, value in original.items() if key != "annotations"}
    assert len(merged[0].annotations) == 1
    annotation = merged[0].annotations[0]
    assert annotation.kind == kind
    assert annotation.line_indices == {9}
    assert annotation.line_bboxes == {9: weak.annotations[0].line_bboxes[9]}
    assert annotation.bbox == weak.annotations[0].line_bboxes[9]
    assert asdict(weak) == weak_before


@pytest.mark.parametrize("reason", ["core_missing", "core_member", "inside_core", "outside_bounds", "missing_box"])
def test_weak_candidate_cannot_donate_unproven_annotation(reason):
    target, weak = candidates(core=reason != "core_missing")
    index = {"core_member": 1, "inside_core": 10, "outside_bounds": 11, "missing_box": 13}.get(reason, 9)
    annotation = weak.annotations[0]
    annotation.line_indices = {index}
    annotation.line_bboxes = {i: b for i, b in annotation.line_bboxes.items() if i == index}
    merged = table_rules._merge_table_candidates([target, weak])
    assert len(merged) == 1
    assert merged[0].annotations == []
    assert merged[0].line_indices == {1, 2}


def test_known_catalog_page_keeps_caption_outside_rotated_header_grid():
    """Replay one real page through the host pipeline, not a preselected table box."""
    from io import BytesIO
    from pathlib import Path
    from pypdf import PdfReader, PdfWriter
    from bs4 import BeautifulSoup
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel

    writer = PdfWriter()
    writer.add_page(PdfReader(Path(__file__).parents[1] / "fixtures/upstream_068/deep_structure_catalog.pdf").pages[53])
    payload = BytesIO()
    writer.write(payload)
    with PDFDocument(payload.getvalue()) as document:
        pages = PdfModel().predict(document)
    tables = [b for b in pages[0] if b["type"] == "table" and "Welcome and Introductions" in b["content"]]
    assert len(tables) == 1
    soup = BeautifulSoup(tables[0]["content"], "html.parser")
    rows = [[c.get_text() for c in row.find_all("td")] for row in soup.find_all("tr")]
    assert len(rows) == 2 and all(len(row) == 2 for row in rows)
    assert [row[0] for row in rows] == ["Day 1", "Day 2"]
    assert "Table 6.6" not in soup.get_text()
    assert sum("Table 6.6: Policy Committee Meeting Agenda" in str(b["content"])
               for b in pages[0] if b["type"] == "caption") == 1
