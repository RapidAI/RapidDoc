"""用独立横线带恢复正文中的整行分组标题，不放宽普通单元格安全门。"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
from typing import Any

from ..spatial_text import project_pdf_table_text
from .candidate import GridCellSpec, build_candidate
from .contracts import NativeTableCandidate, NativeTableInput, NativeTableText
from .geometry import normalize_angle, table_local_size
from .sparse_common import _local_rules, cluster_members
from .sparse_multiline import (
    _has_overlapping_formula_rows,
    _infer_target_columns,
    _infer_text_tracks,
    _row_occupancy,
    _stable_gutters,
)


def _select_rows(text: NativeTableText, indices: set[int]) -> NativeTableText:
    """建立仅供列证据检查的局部行视图，保留字符来源并同步视觉行索引。"""
    rows = [row for row in text.rows if row.row_index in indices]
    mapping = {row.row_index: index for index, row in enumerate(rows)}
    return replace(
        text,
        rows=tuple(replace(row, row_index=mapping[row.row_index]) for row in rows),
        glyphs=tuple(
            replace(glyph, visual_row=mapping[glyph.visual_row]) for glyph in text.glyphs if glyph.visual_row in mapping
        ),
    )


def build_grouped_rule_band_candidate(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: dict[str, Any],
) -> NativeTableCandidate | None:
    """只在重复独立分组带与稳定数据列共同成立时生成正文 colspan。"""
    diagnostics.update(evidence="grouped_rule_band", first_rejection_gate="group_bands")
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    tolerance = max(0.5, 0.10 * text.median_glyph_height)
    horizontal = [
        coordinate
        for coordinate, _aliases in cluster_members(
            [
                rule.coordinate
                for rule in rules
                if rule.orientation == "horizontal"
                and rule.end - rule.start >= 0.90 * width
                and -3 * tolerance <= rule.coordinate <= height + 3 * tolerance
            ],
            tolerance,
        )
    ]
    # 模型框可能裁掉末条外线；跨列证据必须来自组内上下横线，不依赖框外补线。
    if len(horizontal) < 4 or horizontal[0] > 3 * tolerance:
        return None
    group_bands: dict[int, tuple[float, float]] = {}
    for upper, lower in zip(horizontal, horizontal[1:]):
        members = [row for row in text.rows if upper < (row.bbox[1] + row.bbox[3]) / 2.0 < lower]
        if len(members) != 1:
            continue
        row = members[0]
        if row.row_index == 0 or len(row.tokens) != 1:
            continue
        if row.bbox[1] < upper - tolerance or row.bbox[3] > lower + tolerance:
            return None
        if any(
            rule.orientation == "vertical"
            and tolerance < rule.coordinate < width - tolerance
            and min(lower, rule.end) - max(upper, rule.start) > tolerance
            for rule in rules
        ):
            return None
        group_bands[row.row_index] = (upper, lower)
    if len(group_bands) < 2 or min(group_bands) != 1:
        return None
    if _has_overlapping_formula_rows(text, -1.0):
        diagnostics["first_rejection_gate"] = "overlapping_formula_rows"
        return None
    group_indices = sorted(group_bands)
    if any(end - start < 2 for start, end in zip(group_indices, group_indices[1:] + [len(text.rows)])):
        return None
    data_indices = set(range(1, len(text.rows))) - group_bands.keys()
    data = _select_rows(text, data_indices)
    cols = _infer_target_columns(data)
    if cols is None or cols < 3:
        diagnostics["first_rejection_gate"] = "column_count"
        return None
    tracks = _infer_text_tracks(data, width, cols)
    if tracks is None:
        diagnostics["first_rejection_gate"] = "column_tracks"
        return None
    diagnostics.update(x_tracks=list(tracks), group_rows=group_indices)
    glyphs = {glyph.glyph_id: glyph for glyph in text.glyphs}
    full = set(range(cols))
    for row in data.rows:
        occupied = _row_occupancy(row, glyphs, tracks)
        if occupied not in (full, full - {0}) or len(row.tokens) != len(occupied):
            diagnostics["first_rejection_gate"] = "data_occupancy"
            return None
    # 表头可以有明确空白的首列；单层列标题不能横切任何叶子列边界。
    header = text.rows[0]
    if _row_occupancy(header, glyphs, tracks) not in (full, full - {0}):
        diagnostics["first_rejection_gate"] = "header_occupancy"
        return None
    margin = max(0.15, 0.03 * text.median_glyph_width)
    if any(token.bbox[0] + margin < boundary < token.bbox[2] - margin for token in header.tokens for boundary in tracks[1:-1]):
        diagnostics["first_rejection_gate"] = "header_token_split"
        return None
    if not _stable_gutters(data, tracks, frozenset(), -1.0, diagnostics):
        diagnostics["first_rejection_gate"] = "gutter_support"
        return None
    boundaries = [0.0]
    for previous, current in zip(text.rows, text.rows[1:]):
        if current.row_index in group_bands:
            boundary = group_bands[current.row_index][0]
        elif previous.row_index in group_bands:
            boundary = group_bands[previous.row_index][1]
        else:
            # 字体 loose 框可轻微交叠；独立视觉行按中心分界，最终仍要求零字符落格歧义。
            boundary = sum((previous.bbox[1], previous.bbox[3], current.bbox[1], current.bbox[3])) / 4.0
        if (current.row_index in group_bands or previous.row_index in group_bands) and not (
            previous.bbox[3] - tolerance <= boundary <= current.bbox[1] + tolerance
        ):
            diagnostics["first_rejection_gate"] = "row_boundary"
            return None
        boundaries.append(boundary)
    boundaries.append(height)
    specs = []
    for row in text.rows:
        index = row.row_index
        if index in group_bands:
            specs.append(GridCellSpec(index, 0, 1, cols, (0.0, boundaries[index], width, boundaries[index + 1])))
        else:
            specs.extend(
                GridCellSpec(index, col, 1, 1, (tracks[col], boundaries[index], tracks[col + 1], boundaries[index + 1]))
                for col in range(cols)
            )
    record: dict[str, object] = {}
    candidate = build_candidate(
        source="sparse_multiline",
        rows=len(text.rows),
        cols=cols,
        specs=tuple(specs),
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=grouped_rule_band", f"group_rows={','.join(map(str, group_indices))}"),
        use_grid_index=True,
        diagnostics=record,
    )
    diagnostics["candidate_diagnostics"] = record
    if (
        candidate is None
        or candidate.text_capture < 1.0
        or candidate.order_consistency < 1.0
        or candidate.score < 0.98
        or record.get("ambiguous_glyph_ratio", 1.0) > 0.0
        or record.get("token_split_count", 1) > 0
    ):
        diagnostics["first_rejection_gate"] = "verified_integrity"
        return None
    # 新跨列标题复用原生投影的紧字框间距；不改变普通格和既有恢复路径的文本行为。
    raw_by_source = {}
    for index, char in enumerate(table_input.chars):
        try:
            source_index = int(char.get("char_idx", index))
        except (TypeError, ValueError):
            source_index = index
        raw_by_source[source_index] = char
    visible_sources = sorted(index for index, char in raw_by_source.items() if str(char.get("char", "")).strip())
    cells = []
    for cell in candidate.cells:
        if cell.colspan == cols:
            sources = set(cell.source_char_indices)
            first_source, last_source = min(sources), max(sources)
            selected_chars = []
            for index, char in raw_by_source.items():
                if index in sources:
                    selected_chars.append(char)
                elif first_source < index < last_source and str(char.get("char", "")).isspace():
                    # PDF 空格框可退化为点；只认领左右可见源字符都属于本格的显式空白。
                    position = bisect_left(visible_sources, index)
                    if (
                        0 < position < len(visible_sources)
                        and visible_sources[position - 1] in sources
                        and visible_sources[position] in sources
                    ):
                        selected_chars.append(char)
            content = " ".join(project_pdf_table_text(selected_chars, table_input.table_bbox, table_input.angle).split())
            if "".join(content.split()) != "".join(cell.content.split()):
                diagnostics["first_rejection_gate"] = "group_text_integrity"
                return None
            cell = replace(cell, content=content)
        cells.append(cell)
    diagnostics.update(first_rejection_gate=None, grid={"rows": candidate.rows, "cols": candidate.cols})
    return replace(candidate, cells=tuple(cells))
