"""以三线表物理表头边界恢复换行表头和同步重启的描述列。"""

from __future__ import annotations

from typing import Any

from .candidate import GridCellSpec, build_candidate
from .contracts import NativeTableCandidate, NativeTableInput, NativeTableText
from .geometry import normalize_angle, table_local_size
from .sparse_common import _local_rules
from .sparse_multiline import (
    _build_column_hypothesis,
    _has_overlapping_formula_rows,
    _local_rectangles,
    _row_occupancy,
    _stable_gutters,
)


def build_rule_band_candidates(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[NativeTableCandidate]:
    """只在三条完整横线及无歧义列占用共同成立时恢复特殊三线表。"""

    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    tolerance = max(1.0, 0.15 * text.median_glyph_height)
    horizontal = sorted(
        {rule.coordinate for rule in rules if rule.orientation == "horizontal" and rule.end - rule.start >= 0.9 * width}
    )
    if (
        len(horizontal) != 3
        or horizontal[0] > tolerance
        or height - horizontal[-1] > tolerance
        or any(rule.orientation == "vertical" and tolerance < rule.coordinate < width - tolerance for rule in rules)
    ):
        return []
    header_bottom = horizontal[1]
    header = [row for row in text.rows if row.bbox[3] <= header_bottom + 0.5]
    body = [row for row in text.rows if row.bbox[1] >= header_bottom - 0.5]
    if not 1 <= len(header) <= 3 or len(body) < 3 or len(header) + len(body) != len(text.rows):
        return []
    if _has_overlapping_formula_rows(text, header_bottom):
        return []
    hypothesis = _build_column_hypothesis(text, width, rules, _local_rectangles(table_input, width, height), None)
    if hypothesis is None:
        return []
    tracks = hypothesis.x_tracks
    cols = len(tracks) - 1
    glyphs = {glyph.glyph_id: glyph for glyph in text.glyphs}
    occupancy = [_row_occupancy(row, glyphs, tracks) for row in body]
    full = set(range(cols))
    common = set.intersection(*occupancy)
    prefix_cols = min(common) if common else 0
    groups: list[tuple[int, int]] = []
    if prefix_cols:
        # 至少两个描述列同步重启，右侧逐行完整，才能区分 rowspan 与普通空格。
        suffix = set(range(prefix_cols, cols))
        starts = [index for index, occupied in enumerate(occupancy) if occupied == full]
        ends = starts[1:] + [len(body)]
        if (
            prefix_cols < 2
            or len(suffix) < 2
            or common != suffix
            or len(starts) < 3
            or starts[0] != 0
            or any(occupied not in (full, suffix) for occupied in occupancy)
            or any(end - start < 2 for start, end in zip(starts, ends))
        ):
            return []
        groups = list(zip(starts, ends))
    elif any(occupied != full for occupied in occupancy):
        return []
    if not groups and len(header) == 1:
        # 普通单行表头三线表仍走既有策略，避免无关候选和金标漂移。
        return []
    header_occupancy = [_row_occupancy(row, glyphs, tracks) for row in header]
    if set.union(*header_occupancy) != full:
        return []
    if len(header) > 1 and header_occupancy[-1] != full:
        # 多级表头可能有仅出现在上层的 rowspan 标签，不能误折叠成普通换行表头。
        return []
    margin = max(0.25, 0.05 * text.median_glyph_width)
    if any(
        token.bbox[0] + margin < boundary < token.bbox[2] - margin
        for row in header
        for token in row.tokens
        for boundary in tracks[1:-1]
    ):
        return []
    if not _stable_gutters(text, tracks, hypothesis.physical_boundaries, header_bottom, None):
        return []
    boundaries = [header_bottom]
    for previous, current in zip(body, body[1:]):
        if previous.bbox[3] > current.bbox[1] + 0.5:
            return []
        boundaries.append((previous.bbox[3] + current.bbox[1]) / 2.0)
    boundaries.append(height)
    specs = [GridCellSpec(0, col, 1, 1, (tracks[col], 0.0, tracks[col + 1], header_bottom)) for col in range(cols)]
    spans = {start: end for start, end in groups}
    for index in range(len(body)):
        for col in range(cols):
            if col < prefix_cols and index not in spans:
                continue
            end = spans[index] if col < prefix_cols else index + 1
            specs.append(
                GridCellSpec(index + 1, col, end - index, 1, (tracks[col], boundaries[index], tracks[col + 1], boundaries[end]))
            )
    record: dict[str, Any] = {}
    candidate = build_candidate(
        source="sparse_multiline",
        rows=len(body) + 1,
        cols=cols,
        specs=tuple(specs),
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=physical_header_band", f"wrapped_header_rows={len(header)}", f"descriptor_columns={prefix_cols}"),
        use_grid_index=True,
        diagnostics=record,
    )
    if diagnostics is not None:
        diagnostics.append({"source": "sparse_multiline", "evidence": "physical_header_band", **record})
    if (
        candidate is None
        or candidate.text_capture < 1.0
        or candidate.order_consistency < 1.0
        or candidate.score < 0.98
        or record.get("ambiguous_glyph_ratio", 1.0) > 0.0
    ):
        return []
    return [candidate]
