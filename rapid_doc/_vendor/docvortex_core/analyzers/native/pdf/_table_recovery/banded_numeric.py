"""以重复数值行、共同字形间隙和物理表头证据恢复少线表。"""

from __future__ import annotations

import re
import statistics
from bisect import bisect_left
from collections import Counter
from dataclasses import replace
from typing import Any

from ..spatial_text import project_pdf_table_text
from .candidate import GridCellSpec, build_candidate
from .contracts import NativeTableCandidate, NativeTableInput, NativeTableText
from .geometry import normalize_angle, page_bbox_to_table_local, table_local_size
from .sparse_common import _local_rules


def _numeric_value(value: str) -> bool:
    """接受数值、常见单位后缀或独立状态符号，不把普通短词当数据。"""
    return bool(re.fullmatch(r"(?:[<>≤≥+-]?\d[\d.,\s–—/+-]*(?:%|[kKmMbB])?|[O✓✔✗✘×])", value.strip()))


def _shared_gutter(text: NativeTableText, indices: set[int], lower: float, upper: float) -> float | None:
    """在所有指定行的字形并集中寻找共同空白，拒绝横切单个字形。"""
    intervals = sorted(
        (max(lower, g.bbox[0]), min(upper, g.bbox[2]))
        for g in text.glyphs
        if g.visual_row in indices and g.bbox[2] > lower and g.bbox[0] < upper
    )
    gaps = []
    cursor = lower
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper:
        gaps.append((cursor, upper))
    if not gaps:
        return None
    left, right = max(gaps, key=lambda gap: gap[1] - gap[0])
    return (left + right) / 2 if right - left >= max(0.6, 0.08 * text.median_glyph_height) else None


def _filled_header_tracks(table_input: NativeTableInput, text: NativeTableText, cols: int, width: float) -> list[float] | None:
    """仅采用同高、相邻且覆盖全部表头文本的底色矩形作为列界。"""
    boxes = []
    for rectangle in table_input.rectangles:
        if not rectangle.fill_visible or rectangle.segment_count != 5:
            continue
        box = page_bbox_to_table_local(rectangle.bbox, table_input.table_bbox, normalize_angle(table_input.angle))
        if (
            box
            and box[1] <= text.rows[0].bbox[1]
            and box[3] >= text.rows[0].bbox[3]
            and box[3] - box[1] > text.median_glyph_height
        ):
            boxes.append(box)
    boxes.sort()
    if len(boxes) != cols or any(
        abs(a[2] - b[0]) > 1 or abs(a[1] - b[1]) > 1 or abs(a[3] - b[3]) > 1 for a, b in zip(boxes, boxes[1:])
    ):
        return None
    tracks = [0.0, *(box[2] for box in boxes[:-1]), width]
    if any(
        not tracks[col] <= token.bbox[0] <= token.bbox[2] <= tracks[col + 1] for col, token in enumerate(text.rows[0].tokens)
    ):
        return None
    return tracks


def build_banded_numeric_candidate(
    table_input: NativeTableInput, text: NativeTableText, diagnostics: dict[str, Any]
) -> NativeTableCandidate | None:
    """只在三条以上重复数据行、完整落格和物理边界同时成立时生成候选。"""
    diagnostics.update(evidence="banded_numeric", first_rejection_gate="repeated_data_rows")
    dense = [row for row in text.rows if len(row.tokens) >= 2 and all(_numeric_value(token.text) for token in row.tokens[1:])]
    if len(dense) < 3:
        return None
    cols, count = Counter(len(row.tokens) for row in dense).most_common(1)[0]
    dense = [row for row in dense if len(row.tokens) == cols]
    if count < 3 or cols > 20:
        return None
    first_data = dense[0].row_index
    if not 1 <= first_data <= 3 or len(text.rows) < first_data + 3:
        return None
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    tolerance = max(0.6, 0.12 * text.median_glyph_height)
    horizontal = [rule for rule in rules if rule.orientation == "horizontal"]
    filled = _filled_header_tracks(table_input, text, cols, width)
    outer = [rule for rule in horizontal if rule.end - rule.start >= 0.9 * width]
    if filled is None and not (
        any(rule.coordinate < 3 * tolerance for rule in outer)
        and any(rule.coordinate > height - 3 * tolerance for rule in outer)
    ):
        diagnostics["first_rejection_gate"] = "physical_boundary"
        return None
    centers = [
        statistics.median((row.tokens[col].bbox[0] + row.tokens[col].bbox[2]) / 2 for row in dense) for col in range(cols)
    ]
    if any(b - a < 2 * text.median_glyph_width for a, b in zip(centers, centers[1:])):
        return None
    body_indices = set(range(first_data, len(text.rows)))
    # 叶子表头参与共同间隙验证；更高层表头另由局部横线证明跨列。
    leaf = first_data - 1
    if len(text.rows[leaf].tokens) < cols - 1 and first_data == 2:
        leaf = 0
    grouped_header = first_data == 3 and len(text.rows[0].tokens) == 1 and len(text.rows[2].tokens) == cols - 1
    gutter_rows = body_indices | ({leaf} if grouped_header else set(range(first_data)))
    tracks = filled
    if tracks is None:
        tracks = [0.0]
        for col in range(cols - 1):
            lower = max(row.tokens[col].bbox[2] for row in dense)
            upper = min(row.tokens[col + 1].bbox[0] for row in dense)
            boundary = _shared_gutter(text, gutter_rows, lower, upper)
            if boundary is None:
                diagnostics["first_rejection_gate"] = "shared_gutter"
                return None
            tracks.append(boundary)
        tracks.append(width)
    glyphs = {g.glyph_id: g for g in text.glyphs}
    for row in text.rows[first_data:]:
        values = [[] for _ in range(cols)]
        for glyph_id in row.glyph_ids:
            glyph = glyphs[glyph_id]
            owners = [col for col in range(cols) if tracks[col] <= glyph.bbox[0] and glyph.bbox[2] <= tracks[col + 1]]
            if len(owners) != 1:
                diagnostics["first_rejection_gate"] = "body_glyph_split"
                return None
            values[owners[0]].append(glyph)
        if any(not value for value in values) or any(
            not _numeric_value("".join(g.text for g in sorted(value, key=lambda g: g.bbox[0]))) for value in values[1:]
        ):
            diagnostics["first_rejection_gate"] = "body_occupancy"
            return None
    boundaries = [
        0.0,
        *(sum((a.bbox[1], a.bbox[3], b.bbox[1], b.bbox[3])) / 4 for a, b in zip(text.rows, text.rows[1:])),
        height,
    ]
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    occupied = set()
    # 三层表头仅在组底横线吻合连续叶子列、且首列上下均无标题时合并。
    if grouped_header:
        groups = []
        for token in text.rows[1].tokens:
            if token.bbox[2] < tracks[1]:
                continue
            matches = [
                rule
                for rule in horizontal
                if token.bbox[3] < rule.coordinate < text.rows[2].bbox[1]
                and rule.start <= token.bbox[0]
                and token.bbox[2] <= rule.end
            ]
            if len(matches) != 1:
                diagnostics["first_rejection_gate"] = "group_rules"
                return None
            rule = matches[0]
            columns = [col for col in range(1, cols) if rule.start - tolerance <= centers[col] <= rule.end + tolerance]
            if not columns or columns != list(range(min(columns), max(columns) + 1)):
                return None
            groups.append(columns)
        if sorted(col for group in groups for col in group) != list(range(1, cols)):
            return None
        if any(g.visual_row in {0, 2} and g.bbox[0] < tracks[1] for g in text.glyphs):
            return None
        spans[(0, 0)] = (3, 1)
        spans[(0, 1)] = (1, cols - 1)
        for group in groups:
            spans[(1, group[0])] = (1, len(group))
    specs = []
    for row in range(len(text.rows)):
        for col in range(cols):
            if (row, col) in occupied:
                continue
            rowspan, colspan = spans.get((row, col), (1, 1))
            occupied.update((r, c) for r in range(row, row + rowspan) for c in range(col, col + colspan))
            specs.append(
                GridCellSpec(
                    row, col, rowspan, colspan, (tracks[col], boundaries[row], tracks[col + colspan], boundaries[row + rowspan])
                )
            )
    # 高层标题若横切列界而无横线合并证据，拒绝猜测；同列跨行文字保持原文。
    for spec in specs:
        for glyph in text.glyphs:
            if (
                spec.row <= glyph.visual_row < spec.row + spec.rowspan
                and spec.bbox[0] < (glyph.bbox[0] + glyph.bbox[2]) / 2 < spec.bbox[2]
            ):
                if glyph.bbox[0] < spec.bbox[0] or glyph.bbox[2] > spec.bbox[2]:
                    diagnostics["first_rejection_gate"] = "header_glyph_split"
                    return None
    record: dict[str, object] = {}
    candidate = build_candidate(
        source="sparse_hybrid",
        rows=len(text.rows),
        cols=cols,
        specs=tuple(specs),
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=banded_numeric",),
        use_grid_index=True,
        diagnostics=record,
    )
    diagnostics.update(x_tracks=tracks, candidate_diagnostics=record)
    if (
        candidate is None
        or candidate.score < 0.98
        or candidate.text_capture < 1.0
        or record.get("ambiguous_glyph_ratio", 1) > 0
    ):
        diagnostics["first_rejection_gate"] = "verified_integrity"
        return None
    # 表头 loose 字框可能重叠；公共空间投影按原始基线恢复各格物理行。
    raw = {int(char.get("char_idx", index)): char for index, char in enumerate(table_input.chars)}
    visible = sorted(index for index, char in raw.items() if str(char.get("char", "")).strip())
    cells = []
    for cell in candidate.cells:
        sources = set(cell.source_char_indices)
        selected = [raw[index] for index in sources if index in raw]
        for index, char in raw.items():
            if not str(char.get("char", "")).isspace():
                continue
            position = bisect_left(visible, index)
            if 0 < position < len(visible) and visible[position - 1] in sources and visible[position] in sources:
                selected.append(char)
        selected.sort(key=lambda char: int(char.get("char_idx", 0)))
        content = " ".join(project_pdf_table_text(selected, table_input.table_bbox, table_input.angle).split())
        if "".join(content.split()) != "".join(cell.content.split()) and cell.row >= first_data:
            diagnostics["first_rejection_gate"] = "body_text_integrity"
            return None
        cells.append(replace(cell, content=content))
    diagnostics["first_rejection_gate"] = None
    return replace(candidate, cells=tuple(cells))
