"""用原生数值列、表头及字符净空恢复少线数据表，不依赖文档内容。"""

from __future__ import annotations

from dataclasses import replace
import re
import statistics
from .contracts import NativeTableText, NativeTableTextRow
from .text import _tokenize_row
from .geometry import bbox_union
from .candidate import GridCellSpec, build_candidate
from .geometry import table_local_size, normalize_angle
from .sparse_common import _local_rules


def regroup_numeric_text(text):
    """金额处于换行标签中间时按独立字形中心组行，避免高数字框桥接两行汉字。"""
    groups = []
    centers = []
    em = text.median_glyph_height
    for glyph in sorted(text.glyphs, key=lambda g: ((g.bbox[1] + g.bbox[3]) / 2, g.bbox[0])):
        y = (glyph.bbox[1] + glyph.bbox[3]) / 2
        if not groups or y - centers[-1] > max(0.75, 0.3 * em):
            groups.append([glyph])
            centers.append(y)
        else:
            groups[-1].append(glyph)
            centers[-1] = statistics.median((g.bbox[1] + g.bbox[3]) / 2 for g in groups[-1])
    glyphs, rows = [], []
    for index, group in enumerate(groups):
        members = sorted([replace(g, visual_row=index) for g in group], key=lambda g: g.bbox[0])
        glyphs.extend(members)
        rows.append(
            NativeTableTextRow(
                index,
                bbox_union(g.bbox for g in members),
                _tokenize_row(members, 0.35 * text.median_glyph_width, em),
                tuple(g.glyph_id for g in members),
            )
        )
    return NativeTableText(tuple(sorted(glyphs, key=lambda g: g.glyph_id)), tuple(rows), text.median_glyph_width, em)


_VALUE = re.compile(r"(?:[+−-]?\(?\d+(?:[.,]\d+)*\)?\s*[%‰]?|[-–—])$")


def numeric_token(token):
    """接受金额、负数和缺省横线，年份日期中的文字不会作为金额种子。"""
    return bool(_VALUE.fullmatch(token.text.strip()))


def numeric_row_groups(text):
    """按反复对齐的数值右缘分组；跨越全宽正文或不同列槽时关闭当前表。"""
    em = text.median_glyph_height
    groups = []
    active = []
    anchors = []
    for row in text.rows:
        values = [token for token in row.tokens if numeric_token(token)]
        if len(values) < 2:
            continue
        positions = [token.bbox[2] for token in values]
        prefix = [t for t in row.tokens if not numeric_token(t) and t.bbox[2] < values[0].bbox[0]]
        if not prefix and row.row_index:
            prior = text.rows[row.row_index - 1]
            if row.bbox[1] - prior.bbox[3] < em:
                prefix = [t for t in prior.tokens if not numeric_token(t) and t.bbox[2] < values[0].bbox[0]]
        if not active and not prefix:
            continue
        if any(b - a < 2 * em for a, b in zip(positions, positions[1:])):
            continue
        matches = bool(anchors) and all(min(abs(x - a) for a in anchors) <= 0.65 * em for x in positions)
        between = text.rows[active[-1] + 1 : row.row_index] if active else ()
        prefix_limit = min(token.bbox[0] for token in values) - 0.5 * em
        barrier = any(
            any(token.bbox[0] < prefix_limit and token.bbox[2] > prefix_limit + em for token in prior.tokens)
            for prior in between
        )
        if active and (not matches or barrier or row.bbox[1] - text.rows[active[-1]].bbox[3] > 22 * em):
            groups.append((active, anchors))
            active = []
        if not active:
            anchors = positions
        elif len(positions) > len(anchors):
            anchors = positions
        active.append(row.row_index)
    if active:
        groups.append((active, anchors))
    return groups


def numeric_region_bounds(text, rows, anchors, rules):
    """从数据行向上收集贴邻二维表头，正文题名和跨列段落构成硬屏障。"""
    em = text.median_glyph_height
    first, last = rows[0], rows[-1]
    minimum = min(t.bbox[0] for i in rows for t in text.rows[i].tokens if numeric_token(t))
    left_labels = [
        t.bbox for row in text.rows[first : last + 1] for t in row.tokens if t.bbox[2] < minimum and not numeric_token(t)
    ]
    if not left_labels or len(anchors) < 2:
        return None
    left = min(box[0] for box in left_labels)
    prefix_boundary = (max(box[2] for box in left_labels) + minimum) / 2
    right = max(anchors)
    # 只有重复数据列上方多列文字或同宽横线存在，才能从正文数值推导表头。
    top = first
    while top:
        prior = text.rows[top - 1]
        current = text.rows[top]
        if current.bbox[1] - prior.bbox[3] > 5 * em:
            break
        if any(t.bbox[0] < left - 3 * em or t.bbox[2] > right + em for t in prior.tokens):
            break
        if any(t.bbox[0] < prefix_boundary - em and t.bbox[2] > prefix_boundary + em for t in prior.tokens):
            break
        if len(prior.tokens) == 1 and prior.bbox[0] >= minimum and prior.bbox[2] - prior.bbox[0] > 8 * em:
            break
        top -= 1
    header = text.rows[top:first]
    # 数据前的空值小节可能向左悬挂；完整项目列必须包含它的首字，不能只沿金额行裁框。
    left = min([left, *[t.bbox[0] for row in header for t in row.tokens if t.bbox[2] < minimum]])
    # 完整顶边横线分隔表头和其上方正文尾行，不将“如下”等段尾带入表格。
    if header:
        first_multi = next((row for row in header if len(row.tokens) >= 2), None)
        if first_multi is not None:
            tops = [
                r.coordinate
                for r in rules
                if r.orientation == "horizontal"
                and r.start <= minimum
                and r.end >= right - em
                and 0 <= first_multi.bbox[1] - r.coordinate <= em
            ]
            if tops:
                boundary = max(tops)
                top = next(row.row_index for row in header if row.bbox[3] > boundary)
                header = text.rows[top:first]
    numeric_width = right - minimum
    has_header = any(len(row.tokens) >= 2 and any(not numeric_token(t) for t in row.tokens) for row in header)
    physical = [
        r
        for r in rules
        if r.orientation == "horizontal"
        and r.end - r.start >= max(3 * em, 0.3 * numeric_width)
        and r.start <= right
        and r.end >= minimum
        and text.rows[top].bbox[1] - em <= r.coordinate <= text.rows[last].bbox[3] + em
    ]
    if not has_header or len(physical) < 2:
        return None
    # 数据最后一行后的贴邻左列续行仍属于当前项目，不能剪掉换行标签。
    while last + 1 < len(text.rows):
        next_row = text.rows[last + 1]
        if next_row.bbox[1] - text.rows[last].bbox[3] > 0.8 * em or any(t.bbox[2] >= minimum for t in next_row.tokens):
            break
        last += 1
    right = max(right, *(t.bbox[2] for row in text.rows[top : last + 1] for t in row.tokens))
    top_y, bottom_y = text.rows[top].bbox[1], text.rows[last].bbox[3]
    borders = [
        r
        for r in rules
        if r.orientation == "horizontal"
        and r.start <= left + em
        and r.end >= right - em
        and top_y - em <= r.coordinate <= bottom_y + em
        and r.end - r.start <= 1.15 * (right - left) + 2 * em
    ]
    left = min([left, *[r.start for r in borders]])
    right = max([right, *[r.end for r in borders]])
    near_top = [
        r.coordinate
        for r in rules
        if r.orientation == "horizontal" and r.start <= minimum and r.end >= right - em and 0 <= top_y - r.coordinate <= em
    ]
    near_bottom = [
        r.coordinate
        for r in rules
        if r.orientation == "horizontal" and r.start <= minimum and r.end >= right - em and 0 <= r.coordinate - bottom_y <= em
    ]
    return (
        left - 0.2 * em,
        min(near_top, default=top_y - 0.2 * em),
        right + 0.2 * em,
        max(near_bottom, default=bottom_y + 0.2 * em),
    )


def _column_tracks(text, data, header, width, rules):
    """用数据共同净空建立列轨，仅在表头证明额外文字列时扩展列数。"""
    em = text.median_glyph_height
    values = [[t for t in row.tokens if numeric_token(t)] for row in data]
    anchors = max(values, key=len)
    if len(anchors) < 2:
        return None
    buckets = [[] for _ in anchors]
    for row_values in values:
        for token in row_values:
            owner = min(range(len(anchors)), key=lambda i: abs(token.bbox[2] - anchors[i].bbox[2]))
            if abs(token.bbox[2] - anchors[owner].bbox[2]) > 0.65 * em:
                return None
            buckets[owner].append(token.bbox)
    if any(len(items) < 2 for items in buckets):
        return None
    first_value = min(b[0] for b in buckets[0])
    prefix = [
        t.bbox[2]
        for row in text.rows[data[0].row_index :]
        for t in row.tokens
        if t.bbox[2] < first_value and not numeric_token(t)
    ]
    if not prefix and data[0].row_index:
        prefix = [
            t.bbox[2]
            for row in text.rows[max(0, data[0].row_index - 2) : data[0].row_index]
            for t in row.tokens
            if t.bbox[2] < first_value and not numeric_token(t)
        ]
    if not prefix or max(prefix) >= first_value:
        return None
    tracks = [0.0, (max(prefix) + first_value) / 2]
    for a, b in zip(buckets, buckets[1:]):
        lower, upper = max(box[2] for box in a), min(box[0] for box in b)
        if upper - lower < 0.3 * em:
            return None
        tracks.append((lower + upper) / 2)
    # 同一行覆盖全部金额列的表头提供额外净空，避免较宽日期被金额右缘的中点切开。
    header_columns = [[] for _ in anchors]
    centers = [(token.bbox[0] + token.bbox[2]) / 2 for token in anchors]
    for row in header:
        slots = {}
        for token in row.tokens:
            center = (token.bbox[0] + token.bbox[2]) / 2
            if center < tracks[1] or center > anchors[-1].bbox[2] + em:
                continue
            owner = min(range(len(centers)), key=lambda i: abs(center - centers[i]))
            slots.setdefault(owner, []).append(token.bbox)
        if len(slots) == len(anchors) and all(len(items) == 1 for items in slots.values()):
            for owner, items in slots.items():
                header_columns[owner].extend(items)
    for index in range(len(anchors) - 1):
        lower = max(box[2] for box in [*buckets[index], *header_columns[index]])
        upper = min(box[0] for box in [*buckets[index + 1], *header_columns[index + 1]])
        if upper - lower >= 0.3 * em:
            tracks[index + 2] = (lower + upper) / 2
    last_right = max(b[2] for b in buckets[-1])
    # 原三线表中金额右侧独立的状态/原因表头形成额外文字列；要求正文也有对应成员。
    extra = sorted([t for row in header for t in row.tokens if t.bbox[0] > last_right + 0.5 * em], key=lambda t: t.bbox[0])
    extra_centers = []
    for token in extra:
        center = (token.bbox[0] + token.bbox[2]) / 2
        if not extra_centers or center - extra_centers[-1] > 2 * em:
            extra_centers.append(center)
    if extra_centers:
        candidates = [t for row in text.rows[len(header) :] for t in row.tokens if t.bbox[0] > last_right + 0.5 * em]
        if not candidates:
            return None
        first_extra = min(t.bbox[0] for t in candidates)
        tracks.append((last_right + first_extra) / 2)
        for index in range(len(extra_centers) - 1):
            buckets = [[], []]
            for token in candidates:
                center = (token.bbox[0] + token.bbox[2]) / 2
                owner = min(range(len(extra_centers)), key=lambda i: abs(center - extra_centers[i]))
                if owner in {index, index + 1}:
                    buckets[owner - index].append(token.bbox)
            if not all(buckets):
                return None
            lower = max(box[2] for box in buckets[0])
            upper = min(box[0] for box in buckets[1])
            if upper - lower < 0.3 * em:
                return None
            tracks.append((lower + upper) / 2)
    tracks.append(width)
    return tracks


def _logical_body_rows(text, first_data, tracks):
    """按金额行及紧邻换行标签组装逻辑行，保留独立的空金额小节行。"""
    em = text.median_glyph_height
    groups = []
    pending = []
    for row in text.rows[first_data:]:
        numeric = any(numeric_token(t) and t.bbox[0] >= tracks[1] for t in row.tokens)
        label_only = any(t.bbox[2] <= tracks[1] for t in row.tokens) and not numeric
        if pending and (numeric or not label_only):
            # 数值行已有完整左列文字时，带小节号且明显隔开的前导行是独立空值行。
            section = any(re.match(r"(?:[一二三四五六七八九十]+|\d+)[、.]", t.text) for r in pending for t in r.tokens)
            current_label = any(t.bbox[2] <= tracks[1] for t in row.tokens)
            unit_section = (
                current_label
                and len(pending) == 1
                and any(
                    re.search(r"[（(].+[）)]$", t.text)
                    and t.bbox[2] <= tracks[1]
                    and min(c.bbox[0] for c in row.tokens if c.bbox[2] <= tracks[1]) > t.bbox[0] + 0.5 * em
                    for t in pending[0].tokens
                )
            )
            if (
                row.bbox[1] - pending[-1].bbox[3] > em
                or (section and current_label and row.bbox[1] - pending[-1].bbox[3] > 0.5 * em)
                or (unit_section and row.bbox[1] - pending[-1].bbox[3] > 0.25 * em)
            ):
                groups.append(pending)
                pending = []
            groups.append([*pending, row])
            pending = []
        elif numeric:
            groups.append([row])
        elif label_only:
            if (
                groups
                and not pending
                and row.bbox[1]
                < max((t.bbox[3] for r in groups[-1] for t in r.tokens if numeric_token(t)), default=-1) - 0.1 * em
            ):
                # 数值在两行标签中间时，下方缩进续行仍归前一项目。
                prior_labels = [t for r in groups[-1] for t in r.tokens if t.bbox[2] <= tracks[1]]
                if prior_labels and row.bbox[0] > min(t.bbox[0] for t in prior_labels) + 0.5 * em:
                    groups[-1].append(row)
                    continue
            prior_section = pending and any(
                re.match(r"(?:[一二三四五六七八九十]+|\d+)[、.]", t.text) for r in pending for t in r.tokens
            )
            if pending and (
                row.bbox[1] - pending[-1].bbox[3] > em or prior_section and row.bbox[1] - pending[-1].bbox[3] > 0.5 * em
            ):
                groups.append(pending)
                pending = []
            pending.append(row)
        elif groups:
            # 数据侧的合并原因说明稍后按列跨度绑定，仍保留在当前物理行组。
            groups[-1].append(row)
    if pending:
        groups.append(pending)
    return groups


def build_aligned_numeric_candidate(table_input, text):
    """只有物理表头和重复数值列共同成立时，恢复金额表完整行列及多层表头。"""
    width, height = table_local_size(table_input.table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    if len([r for r in rules if r.orientation == "horizontal"]) < 2:
        return None
    # 完整物理格线已经限定单元格时，不以文字行重写真实合并单元格。
    full_horizontal = [r for r in rules if r.orientation == "horizontal" and r.end - r.start >= 0.8 * width]
    full_vertical = [r for r in rules if r.orientation == "vertical" and r.end - r.start >= 0.6 * height]
    if len(full_horizontal) >= 4 or len(full_horizontal) >= 3 and len(full_vertical) >= 3:
        return None
    groups = numeric_row_groups(text)
    if len(groups) != 1 or len(groups[0][0]) < 2:
        return None
    indices, anchors = groups[0]
    # 当前恢复的证据模型限定一个项目列及二至四个金额列；更宽的多维表保留既有专门恢复。
    if len(anchors) > 4:
        return None
    first = indices[0]
    header = text.rows[:first]
    if not header or first > 8:
        return None
    if any(
        g.text.isdigit() and g.bbox[3] - g.bbox[1] < 0.65 * text.median_glyph_height
        for g in text.glyphs
        if g.visual_row < first
    ):
        return None
    data = [text.rows[i] for i in indices]
    tracks = _column_tracks(text, data, header, width, rules)
    if tracks is None:
        return None
    # 表头明确给出多个左侧字段且正文反复对齐时，不能压成一个换行项目列。
    left_headers = [row for row in header if sum(token.bbox[2] <= tracks[1] for token in row.tokens) >= 2]
    if left_headers and sum(sum(token.bbox[2] <= tracks[1] for token in row.tokens) >= 2 for row in data) >= 3:
        return None
    # 时间维度与指标构成嵌套行表头，需要专门的行跨度恢复，不能当作一列换行项目重写。
    if re.search(r"\d{1,2}[：:]\d{2}", "".join(glyph.text for glyph in text.glyphs if glyph.bbox[0] < tracks[1])):
        return None
    em = text.median_glyph_height
    # 跨列连续文字不是表头；只有横线证明的组标题可占多个数值列。
    if not any(len(row.tokens) >= 2 for row in header):
        return None
    # 贴邻金额之前的独立左列小节是表体，不能反向并入跨列表头。
    header_end = max((row.row_index + 1 for row in header if any(t.bbox[0] >= tracks[1] for t in row.tokens)), default=first)
    header = text.rows[:header_end]
    body = _logical_body_rows(text, header_end, tracks)
    if len(body) < 2:
        return None
    header_labels = [row for row in header if all(t.bbox[2] <= tracks[1] for t in row.tokens)]
    header = tuple(row for row in header if row not in header_labels)
    header_logical = [[row] for row in header]
    if any(a.bbox[3] > b.bbox[1] + 0.4 * em for a, b in zip(header, header[1:])):
        # 各列垂直居中的字段与换行日期交错，只能作为同一完整表头带，而非相交的物理行。
        header_logical = [list(header)]
    header_count = len(header_logical)
    logical = header_logical + body
    boundaries = [0.0]
    for a, b in zip(logical, logical[1:]):
        lower = max(t.bbox[3] for row in a for t in row.tokens if row in header or t.bbox[2] <= tracks[len(anchors) + 1])
        upper = min(t.bbox[1] for row in b for t in row.tokens if row in header or t.bbox[2] <= tracks[len(anchors) + 1])
        if lower > upper + 0.4 * em:
            return None
        boundaries.append((lower + upper) / 2)
    boundaries.append(height)
    cols = len(tracks) - 1
    if not 4 <= cols <= 5:
        return None
    spans = {}
    header_label_tokens = [t for row in header_labels for t in row.tokens]
    if header_label_tokens:
        spans[0, 0] = (header_count, 1)
    for row_index, members in enumerate(header_logical):
        for token in [t for row in members for t in row.tokens]:
            crossed = [c for c in range(cols) if token.bbox[0] < tracks[c + 1] and token.bbox[2] > tracks[c]]
            if len(crossed) == 2 and abs((token.bbox[0] + token.bbox[2]) / 2 - tracks[crossed[1]]) <= 0.65 * em:
                spans[row_index, crossed[0]] = (1, len(crossed))
            matching = [
                r
                for r in rules
                if r.orientation == "horizontal"
                and token.bbox[3] - 0.2 * em
                <= r.coordinate
                <= (logical[row_index + 1][0].bbox[1] + 0.25 * em if row_index + 1 < len(logical) else height)
                and r.start <= token.bbox[0]
                and r.end >= token.bbox[2]
                and r.end - r.start < 0.85 * width
                and abs((r.start + r.end - token.bbox[0] - token.bbox[2]) / 2) <= 0.75 * em
            ]
            if matching:
                rule = min(matching, key=lambda r: r.end - r.start)
                covered = [c for c in range(1, cols) if rule.start - em <= (tracks[c] + tracks[c + 1]) / 2 <= rule.end + em]
                if 1 < len(covered) < cols and covered == list(range(covered[0], covered[-1] + 1)):
                    spans[row_index, covered[0]] = (1, len(covered))
    # 原因/状态只有一个跨两项的说明时，沿表体全部行保留 rowspan。
    number_cols = len(anchors) + 1
    for col in range(number_cols, cols):
        occupied = [
            i
            for i, rows in enumerate(body)
            if any(tracks[col] <= (t.bbox[0] + t.bbox[2]) / 2 <= tracks[col + 1] for row in rows for t in row.tokens)
        ]
        if occupied:
            spans[header_count, col] = (len(body), 1)
    specs, occupied = [], set()
    for row in range(len(logical)):
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
    candidate = build_candidate(
        source="sparse_multiline",
        rows=len(logical),
        cols=cols,
        specs=tuple(specs),
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=("evidence=aligned_numeric_grid",),
        use_grid_index=True,
    )
    # 表头文本项已由独立净空组装；若仍被切入多个单元格，结构证据不足，保留旧恢复路径。
    if candidate is not None:
        for row in header:
            for token in row.tokens:
                sources = set(token.source_char_indices)
                owners = sum(bool(sources.intersection(cell.source_char_indices)) for cell in candidate.cells)
                if owners > 1:
                    return None
    return candidate
