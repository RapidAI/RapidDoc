"""以局部物理分隔和独立子表证据修复少线参数表的层级网格。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter

from .....schema import BBox
from .candidate import GridCellSpec, build_candidate
from .contracts import NativeTableCandidate, NativeTableInput, NativeTableRule, NativeTableText
from .geometry import bbox_union, normalize_angle, table_local_size
from .sparse_common import _local_rules, cluster_members
from .sparse_hybrid import build_sparse_hybrid_candidates


@dataclass(frozen=True, slots=True)
class NestedTableEvidence:
    """保存局部竖线右侧由少线恢复器独立验证的逐格子表。"""

    bbox: BBox
    child: NativeTableCandidate | None


def _subset_text(text: NativeTableText, bbox: BBox) -> NativeTableText | None:
    """按中心选取完整文本项并平移到子表坐标，保留原始字符身份。"""
    left, top, right, bottom = bbox
    glyph_ids = {
        glyph.glyph_id
        for glyph in text.glyphs
        if left <= (glyph.bbox[0] + glyph.bbox[2]) / 2 < right and top <= (glyph.bbox[1] + glyph.bbox[3]) / 2 < bottom
    }
    rows = []
    mapping = {}
    for row in text.rows:
        tokens = [token for token in row.tokens if glyph_ids.intersection(token.glyph_ids)]
        if not tokens:
            continue
        if any(not set(token.glyph_ids).issubset(glyph_ids) for token in tokens):
            return None
        mapping[row.row_index] = len(rows)
        rows.append(
            replace(
                row,
                row_index=len(rows),
                bbox=_shift_bbox(bbox_union(token.bbox for token in tokens), left, top),
                tokens=tuple(replace(token, bbox=_shift_bbox(token.bbox, left, top)) for token in tokens),
                glyph_ids=tuple(gid for gid in row.glyph_ids if gid in glyph_ids),
            )
        )
    if not rows:
        return None
    return replace(
        text,
        rows=tuple(rows),
        glyphs=tuple(
            replace(
                glyph,
                bbox=_shift_bbox(glyph.bbox, left, top),
                advance_bbox=_shift_bbox(glyph.advance_bbox, left, top) if glyph.advance_bbox is not None else None,
                origin=(glyph.origin[0] - left, glyph.origin[1] - top) if glyph.origin is not None else None,
                visual_row=mapping[glyph.visual_row],
            )
            for glyph in text.glyphs
            if glyph.glyph_id in glyph_ids
        ),
    )


def _shift_bbox(bbox: BBox, left: float, top: float) -> BBox:
    """平移局部几何，不缩放或改写原生字符的排版证据。"""
    return bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top


def _subtable_candidate(local_input: NativeTableInput, text: NativeTableText, bbox: BBox) -> NativeTableCandidate | None:
    """复用少线恢复器验证分区，拒绝子表自身仍有跨行跨列或缺字的候选。"""
    subset = _subset_text(text, bbox)
    if subset is None or len(subset.rows) < 3:
        return None
    candidates = build_sparse_hybrid_candidates(replace(local_input, table_bbox=bbox), subset)
    accepted = [
        candidate
        for candidate in candidates
        if candidate.score >= 0.98
        and candidate.text_capture == 1.0
        and candidate.order_consistency == 1.0
        and candidate.rows == len(subset.rows)
        and candidate.cols >= 2
        and all(cell.rowspan == cell.colspan == 1 and cell.content.strip() for cell in candidate.cells)
    ]
    if not accepted or len({candidate.topology for candidate in accepted}) != 1:
        return None
    return max(accepted, key=lambda candidate: candidate.score)


def _local_input(table_input: NativeTableInput) -> NativeTableInput:
    """构造正向物理线证据供分区复用，字符仍由同一次预处理结果提供。"""
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    converted = []
    for rule in rules:
        half = max(0.05, rule.width / 2)
        box = (
            (rule.start, rule.coordinate - half, rule.end, rule.coordinate + half)
            if rule.orientation == "horizontal"
            else (rule.coordinate - half, rule.start, rule.coordinate + half, rule.end)
        )
        converted.append(NativeTableRule(box, rule.width, rule.orientation))
    return NativeTableInput((0.0, 0.0, width, height), (width, height), 0, (), tuple(converted))


def find_nested_table_evidence(table_input: NativeTableInput, text: NativeTableText) -> tuple[NestedTableEvidence, ...]:
    """仅在局部竖线、重复右侧横线和完整子列同时成立时认定嵌套区域。"""
    # 层级网格依赖横向文本行；旋转或未定向字符交给其他候选，避免把跨方向字形当成重复子行。
    if any(glyph.writing_angle != 0 for glyph in text.glyphs):
        return ()
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    vertical = [rule for rule in rules if rule.orientation == "vertical"]
    if not vertical:
        return ()
    tolerance = max(0.5, 0.12 * text.median_glyph_height)
    horizontal = [rule for rule in rules if rule.orientation == "horizontal"]
    full = [rule for rule in horizontal if rule.end - rule.start >= 0.9 * width]
    local_input = _local_input(table_input)
    output = []
    for line in vertical:
        if not 0.1 * width < line.coordinate < 0.8 * width:
            continue
        partial = [
            rule
            for rule in horizontal
            if abs(rule.start - line.coordinate) <= tolerance
            and width - rule.end <= 3 * tolerance
            and line.start - tolerance <= rule.coordinate <= line.end + tolerance
        ]
        separators = cluster_members([rule.coordinate for rule in partial], tolerance)
        if len(separators) < 3:
            continue
        upper = [rule.coordinate for rule in full if line.start - 2 * tolerance <= rule.coordinate <= line.start]
        lower = [rule.coordinate for rule in full if line.end <= rule.coordinate <= line.end + 2 * tolerance]
        if not upper or not lower:
            continue
        box = (line.coordinate, max(upper), width, min(lower))
        if any(abs(box[0] - item.bbox[0]) <= tolerance and abs(box[1] - item.bbox[1]) <= tolerance for item in output):
            continue
        subset = _subset_text(text, box)
        if subset is None or len(subset.rows) != len(separators) + 1:
            continue
        child = _subtable_candidate(local_input, text, box)
        if child is None and not _has_repeated_child_columns(subset):
            continue
        # 跨越两侧的真实文本或另一条贯穿子列的竖线不是此窄入口的层级参数表。
        parent = _subset_text(text, (0.0, box[1], box[0], box[3]))
        if parent is None or len(parent.rows) >= len(subset.rows):
            continue
        output.append(NestedTableEvidence(box, child))
    return tuple(output)


def _has_repeated_child_columns(text: NativeTableText) -> bool:
    """子表有空白或局部歧义时，重复列首仍可否定把所有子列粘成一格的旧结果。"""
    counts = Counter(len(row.tokens) for row in text.rows if len(row.tokens) >= 2)
    if not counts:
        return False
    cols, support = counts.most_common(1)[0]
    if support < 3 or support < 0.6 * len(text.rows):
        return False
    rows = [row for row in text.rows if len(row.tokens) == cols]
    tolerance = max(0.5, 0.2 * text.median_glyph_height)
    return all(
        max(row.tokens[col].bbox[0] for row in rows) - min(row.tokens[col].bbox[0] for row in rows) <= tolerance
        for col in range(cols)
    ) and all(
        min(row.tokens[col + 1].bbox[0] for row in rows) - max(row.tokens[col].bbox[2] for row in rows)
        >= 2 * text.median_glyph_width
        for col in range(cols - 1)
    )


def preserves_nested_cells(candidate: NativeTableCandidate, evidence: tuple[NestedTableEvidence, ...]) -> bool:
    """所有后备候选均须保留已独立验证的子表格子，不能把拒绝的欠分割换名接受。"""
    if not evidence:
        return True
    owners = {index: cell_index for cell_index, cell in enumerate(candidate.cells) for index in cell.source_char_indices}
    for region in evidence:
        if region.child is None:
            return False
        child_owners = []
        for cell in region.child.cells:
            occupied = {owners.get(index) for index in cell.source_char_indices}
            if None in occupied or len(occupied) != 1:
                return False
            child_owners.append(next(iter(occupied)))
        if len(set(child_owners)) != len(child_owners):
            return False
    return True


def build_hierarchical_candidate(
    table_input: NativeTableInput, text: NativeTableText, evidence: tuple[NestedTableEvidence, ...]
) -> NativeTableCandidate | None:
    """只组合一个末尾子表及其上方完整参数行，其他大区域保持拒绝并交给宿主回退。"""
    if len(evidence) != 1:
        return None
    region = evidence[0]
    split, top, width, bottom = region.bbox
    if region.child is None:
        return None
    local_input = _local_input(table_input)
    height = local_input.page_size[1]
    if any((glyph.bbox[1] + glyph.bbox[3]) / 2 >= bottom for glyph in text.glyphs):
        return None
    upper = _subtable_candidate(local_input, text, (0.0, 0.0, width, top))
    if upper is None or upper.cols != 2:
        return None
    child = region.child
    cols = child.cols + 1
    specs = []
    for cell in upper.cells:
        left, right = (0.0, split) if cell.col == 0 else (split, width)
        specs.append(
            GridCellSpec(
                cell.row,
                0 if cell.col == 0 else 1,
                1,
                1 if cell.col == 0 else child.cols,
                (left, cell.bbox[1], right, cell.bbox[3]),
            )
        )
    specs.append(GridCellSpec(upper.rows, 0, child.rows, 1, (0.0, top, split, height)))
    for cell in child.cells:
        box = _shift_bbox(cell.bbox, -split, -top)
        if cell.row == child.rows - 1:
            box = (*box[:3], height)
        specs.append(GridCellSpec(cell.row + upper.rows, cell.col + 1, 1, 1, box))
    candidate = build_candidate(
        source="sparse_hybrid",
        rows=upper.rows + child.rows,
        cols=cols,
        specs=tuple(specs),
        text=text,
        structure_support=min(upper.score, child.score),
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=hierarchical_rule_bands",),
        use_grid_index=True,
        require_atomic_tokens=True,
    )
    if candidate is None or candidate.text_capture != 1.0 or candidate.order_consistency != 1.0:
        return None
    # 组合后的父列边界必须与上部独立恢复的成员一致，禁止借层级证据移动原参数值。
    for original in upper.cells:
        combined = next(cell for cell in candidate.cells if cell.row == original.row and cell.col == original.col)
        if set(original.source_char_indices) != set(combined.source_char_indices):
            return None
    return candidate if preserves_nested_cells(candidate, evidence) else None
