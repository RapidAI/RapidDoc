"""以重复横线分组和独立数值行证明居中首列的 rowspan。"""

from __future__ import annotations

import statistics

from .banded_numeric import _numeric_value
from .candidate import GridCellSpec, build_candidate
from .contracts import NativeTableCandidate, NativeTableInput, NativeTableText
from .geometry import normalize_angle, table_local_size
from .sparse_common import _local_rules
from .sparse_multiline import _build_column_hypothesis, _local_rectangles


def build_banded_rowspan_candidate(
    table_input: NativeTableInput,
    original: NativeTableText,
    refined: NativeTableText,
) -> NativeTableCandidate | None:
    """仅修复至少三组重复的两行数值记录；证据不足时保留原有候选路径。"""

    # 正常视觉行不受影响；居中文字桥接会在独立中心组行后多出行。
    if len(refined.rows) <= len(original.rows):
        return None
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    tolerance = max(0.6, 0.12 * refined.median_glyph_height)
    boundaries = sorted({r.coordinate for r in rules if r.orientation == "horizontal" and r.end - r.start >= 0.9 * width})
    if (
        len(boundaries) < 5
        or boundaries[0] > tolerance
        or height - boundaries[-1] > tolerance
        or any(r.orientation == "vertical" and tolerance < r.coordinate < width - tolerance for r in rules)
    ):
        return None
    # 局部横线可能表示更复杂的单元格拓扑，不能用跨行推测覆盖它。
    if any(r.orientation == "horizontal" and min(abs(r.coordinate - y) for y in boundaries) > tolerance for r in rules):
        return None
    hypothesis = _build_column_hypothesis(original, width, rules, _local_rectangles(table_input, width, height), None)
    if hypothesis is None or len(hypothesis.x_tracks) < 4:
        return None
    tracks = hypothesis.x_tracks
    cols = len(tracks) - 1
    header = [r for r in refined.rows if r.bbox[3] <= boundaries[1]]
    if len(header) != 1 or header[0].bbox[1] < boundaries[0]:
        return None
    # 不允许字形跨列；字符来源最终还需全量、唯一落格。
    if any(not any(tracks[c] <= g.bbox[0] and g.bbox[2] <= tracks[c + 1] for c in range(cols)) for g in refined.glyphs):
        return None
    specs = [GridCellSpec(0, c, 1, 1, (tracks[c], 0.0, tracks[c + 1], boundaries[1])) for c in range(cols)]
    logical_row = 1
    for top, bottom in zip(boundaries[1:-1], boundaries[2:]):
        members = [g for g in refined.glyphs if top < (g.bbox[1] + g.bbox[3]) / 2 < bottom]
        labels = [g for g in members if g.bbox[2] <= tracks[1]]
        data = [g for g in members if g.bbox[0] >= tracks[1]]
        indices = sorted({g.visual_row for g in data})
        if not labels or len(indices) != 2 or len({g.visual_row for g in labels}) != 1:
            return None
        rows = [[g for g in data if g.visual_row == i] for i in indices]
        centers = [statistics.median((g.bbox[1] + g.bbox[3]) / 2 for g in row) for row in rows]
        label_center = statistics.median((g.bbox[1] + g.bbox[3]) / 2 for g in labels)
        if (
            not centers[0] < label_center < centers[1]
            or abs(label_center - sum(centers) / 2) > tolerance
            or any(g.bbox[1] < top or g.bbox[3] > bottom for g in members)
        ):
            return None
        # 每个右侧列都必须独立支持两条数值/状态记录，不拆普通段落换行。
        for row in rows:
            for col in range(1, cols):
                owned = sorted(
                    [g for g in row if tracks[col] <= g.bbox[0] and g.bbox[2] <= tracks[col + 1]],
                    key=lambda g: g.bbox[0],
                )
                if not owned or not _numeric_value("".join(g.text for g in owned)):
                    return None
        upper_bottom = max(g.bbox[3] for g in rows[0])
        lower_top = min(g.bbox[1] for g in rows[1])
        if upper_bottom >= lower_top:
            return None
        cut = (upper_bottom + lower_top) / 2
        specs.append(GridCellSpec(logical_row, 0, 2, 1, (tracks[0], top, tracks[1], bottom)))
        for offset, (y0, y1) in enumerate(((top, cut), (cut, bottom))):
            for col in range(1, cols):
                specs.append(GridCellSpec(logical_row + offset, col, 1, 1, (tracks[col], y0, tracks[col + 1], y1)))
        logical_row += 2
    diagnostics: dict[str, object] = {}
    candidate = build_candidate(
        source="sparse_hybrid",
        rows=logical_row,
        cols=cols,
        specs=tuple(specs),
        text=refined,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=banded_rowspan",),
        use_grid_index=True,
        diagnostics=diagnostics,
    )
    if (
        candidate is None
        or candidate.text_capture < 1.0
        or candidate.order_consistency < 1.0
        or candidate.score < 0.98
        or diagnostics.get("ambiguous_glyph_ratio", 1.0) > 0.0
    ):
        return None
    sources = [index for cell in candidate.cells for index in cell.source_char_indices]
    if len(sources) != len(set(sources)) or len(sources) != len(refined.glyphs):
        return None
    return candidate
