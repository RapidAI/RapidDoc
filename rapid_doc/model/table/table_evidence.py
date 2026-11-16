"""Independent crop evidence shared by the primary and fallback grids."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from .rapid_table_self.table_matcher.main import _normalize_cell_bboxes


def crop_vector_lines(lines, crop, scale, rotation):
    from .rule_table import _segments

    px, py, x0, y0, x1, y1, width, height = crop
    horizontal, vertical = _segments(lines, (x0 / scale, y0 / scale, x1 / scale, y1 / scale))
    result = []
    # 原生线先映射到带 padding 的裁图，再与同方向的 OCR 框一起旋转。
    for axis, segments in enumerate((vertical, horizontal)):
        along, across = (y0 - py, x0 - px) if axis == 0 else (x0 - px, y0 - py)
        for start, end, pos in segments:
            start, end, pos = start * scale - along, end * scale - along, pos * scale - across
            if rotation == '90':
                segment = (1, width - pos, start, end) if axis == 0 else (0, pos, width - end, width - start)
            elif rotation == '270':
                segment = (1, pos, height - end, height - start) if axis == 0 else (0, height - pos, start, end)
            else:
                segment = (axis, pos, start, end)
            result.append(segment)
    return result


def _raster_lines(image, em):
    # 方向开运算保留比字高长得多的连续笔画，适用于扫描和半框表。
    height, width = image.shape[:2]
    factor = min(1., 1400 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if factor < 1:
        gray = cv2.resize(gray, (max(1, round(width * factor)), max(1, round(height * factor))),
                          interpolation=cv2.INTER_AREA)
    window = max(15, round(2 * em * factor) | 1)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, window, 8)
    for axis in (0, 1):
        length = max(12, round(2.5 * em * factor), gray.shape[axis] // 25)
        size = (1, length) if axis == 0 else (length, 1)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            start, end, pos, thickness = (y, y + h, x + w / 2, w) if axis == 0 else (x, x + w, y + h / 2, h)
            if end - start >= max(length, 6 * thickness):
                yield axis, pos / factor, start / factor, end / factor


def _interval_length(intervals):
    end = total = 0.
    for lo, hi in sorted(intervals):
        if hi > end:
            total += hi - max(end, lo)
            end = hi
    return total


def _gutters(boxes, em):
    bands = []
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    for i in np.argsort(centers[:, 1], kind='stable'):
        y = centers[i, 1]
        if not bands or y - bands[-1][0] > .45 * em:
            bands.append((y, []))
        bands[-1][1].append(i)
    gaps, pairs = [], []
    for band_id, (_, band) in enumerate(bands):
        band.sort(key=lambda i: boxes[i, 0])
        for a, b in zip(band, band[1:]):
            lo, hi = boxes[a, 2], boxes[b, 0]
            overlap = min(boxes[a, 3], boxes[b, 3]) - max(boxes[a, 1], boxes[b, 1])
            if hi >= lo and overlap >= .5 * min(boxes[a, 3] - boxes[a, 1], boxes[b, 3] - boxes[b, 1]):
                pairs.append((1, a, b, False))
            if hi - lo >= .45 * em:
                gaps.append((hi, lo, band_id, a, b))

    # 用共同空带和对齐的文字边缘建立列证据，限制在相邻行段，保留合并表头。
    band_ids = [gap[2] for gap in gaps]
    for hi, lo, band_id, a, b in gaps:
        start_index = bisect_left(band_ids, band_id - 6)
        end_index = bisect_right(band_ids, band_id + 6)
        support = {
            other_band for end, start, other_band, _, _ in gaps[start_index:end_index]
            if min(hi, end) - max(lo, start) >= .3 * em
            and min(abs(hi - end), abs(lo - start)) <= .65 * em
        }
        if len(support) >= 3:
            pairs.append((0, a, b, True))

    # 多列共同出现上下对齐和空隙时才建立行证据，单列换行不计入。
    for (y, above), (next_y, below) in zip(bands, bands[1:]):
        if next_y - y < 1.2 * em:
            continue
        aligned = []
        for a in above:
            matches = [b for b in below if boxes[b, 1] - boxes[a, 3] >= .12 * em
                       and min(abs(boxes[a, 0] - boxes[b, 0]), abs(boxes[a, 2] - boxes[b, 2]),
                               abs(centers[a, 0] - centers[b, 0])) < .45 * em
                       and min(boxes[a, 2], boxes[b, 2]) > max(boxes[a, 0], boxes[b, 0])]
            if len(matches) == 1:
                aligned.append((a, matches[0]))
        if len({b for _, b in aligned}) >= 2:
            pairs.extend((1, a, b, True) for a, b in aligned)
    return pairs


def _boundary_tracks(boxes, logic, axis, em):
    start_index = 2 if axis == 0 else 0
    tracks = defaultdict(lambda: ([], []))
    for (start, end), box in zip(logic[:, start_index:start_index + 2], boxes):
        lo, hi = box[1 - axis], box[3 - axis]
        tracks[start][1].append((lo, hi, box[axis]))
        tracks[end + 1][0].append((lo, hi, box[axis + 2]))
    for before, after in tracks.values():
        # 内部边界要求两侧有不同的逻辑单元格，避免跨行/列的格子被虚拟线切开。
        after.sort()
        starts = [span[0] for span in after]
        for lo, hi, pos in before:
            for a, b, edge in after[:bisect_left(starts, hi)]:
                left, right = max(lo, a), min(hi, b)
                if right > left and abs(pos - edge) <= 2 * em:
                    yield (pos + edge) / 2, left, right


@dataclass
class EvidenceScore:
    text: dict
    lines: dict
    gaps: dict
    exact: bool
    missing_numbers: set
    text_count: int
    missing: set
    weak_numbers: set

    def separator_gains(self, primary, axis):
        return sum(primary.gaps[key] > 0 and self.gaps[key] == 0
                   for key in primary.gaps.keys() & self.gaps.keys() if key[0] == axis and key[3])

    def line_improvement(self, primary, axis):
        keys = [key for key in self.lines if key[0] == axis and key in primary.lines]
        return float(sum(primary.lines[key] - self.lines[key] for key in keys)) / max(4, len(keys)) ** .5

    def local_line_improvement(self, primary):
        # 两个候选都支持的长段不参与择优。只累计覆盖明显不同的原图线段。
        differences = [primary.lines[key] - value for key, value in self.lines.items() if key in primary.lines]
        changes = [change for change in differences if abs(change) >= .15]
        return float(sum(changes)) / max(1, len(changes)) ** .5

    def loses_supported_lines(self, primary, line_gain):
        # 多条原先匹配的长线同时消失，才视为明确退步。短线和轻微偏移不触发保护。
        return (line_gain <= -.8
                and sum(self.lines[k] - v >= .6 for k, v in primary.lines.items()
                        if k in self.lines and v <= .1) >= 4)

    @property
    def suspicious(self):
        return (sum(self.lines.values()) >= 1.3 or sum(self.gaps.values()) >= 2
                or self.exact and sum(self.text.values()) >= max(2, .025 * self.text_count))

    def improvement_over(self, primary):
        # 同一组证据项直接比较得失。按层有效数量缩放，避免文字或长表压过其他证据。
        changes = []
        for before, after, count in ((primary.text, self.text, self.text_count),
                                     (primary.lines, self.lines, len(self.lines)),
                                     (primary.gaps, self.gaps, len(self.gaps))):
            keys = before.keys() & after.keys()
            changes.append(float(sum(before[k] - after[k] for k in keys)) / max(4, count) ** .5)
        return tuple(changes)

    def summary(self):
        return ((round(float(sum(self.text.values())), 2), self.text_count),
                (round(float(sum(self.lines.values())), 2), len(self.lines)),
                (round(float(sum(self.gaps.values())), 2), len(self.gaps)))


class TableEvidence:
    def __init__(self, image, rows, excluded=(), vector_lines=()):
        from .fallback import _NUMERIC_CELL

        ids, boxes, confidence, numbers = [], [], [], set()
        height, width = image.shape[:2]
        for i, (points, text, score) in enumerate(zip(*rows)):
            if points is None or not len(points) or not text.strip() or text in excluded or score < .5:
                continue
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            if not np.isfinite(points).all():
                continue
            low = np.clip(points.min(axis=0), 0, (width, height))
            high = np.clip(points.max(axis=0), 0, (width, height))
            if np.any(high <= low):
                continue
            if score >= .85 and _NUMERIC_CELL.fullmatch(text.strip()):
                numbers.add(len(ids))
            ids.append(i)
            boxes.append([*low, *high])
            confidence.append(score)
        self.ids = ids
        self.boxes = np.asarray(boxes).reshape(-1, 4)
        self.confidence = confidence
        self.numbers = numbers
        self.em = max(2., median(self.boxes[:, 3] - self.boxes[:, 1])) if ids else 10.
        self.gaps = _gutters(self.boxes, self.em)
        self.lines = []
        centers = (self.boxes[:, :2] + self.boxes[:, 2:]) / 2
        native = list(vector_lines)
        tracks = []
        self.ruling = ([], [])
        for axis, pos, lo, hi in [*native, *_raster_lines(image, self.em)]:
            axis_limit, along_limit = (width, height) if axis == 0 else (height, width)
            lo, hi = max(0, lo), min(hi, along_limit)
            if not .05 * self.em < pos < axis_limit - .05 * self.em or hi - lo < 2.5 * self.em:
                continue
            self.ruling[axis].append((pos, lo, hi))
            if len(ids) < 2:
                continue
            near = (centers[:, 1-axis] >= lo - self.em) & (centers[:, 1-axis] <= hi + self.em)
            # 只比较两边都有内容的局部线段，忽略外框、页眉下划线和装饰线。
            before = near & (centers[:, axis] < pos - .25 * self.em)
            after = near & (centers[:, axis] > pos + .25 * self.em)
            crossing = near & (self.boxes[:, axis] < pos - .2 * self.em) & (
                self.boxes[:, axis + 2] > pos + .2 * self.em
            )
            if not before.any() or not after.any() or crossing.sum() > min(before.sum(), after.sum()):
                continue
            track = next((parts for other_axis, other_pos, parts in tracks
                          if other_axis == axis and abs(other_pos - pos) <= .2 * self.em), None)
            if track is None:
                tracks.append((axis, pos, [(lo, hi)]))
            else:
                track.append((lo, hi))
        # 矢量和像素常看到同一条线，合并覆盖区间，但保留中间实际断开的部分。
        for axis, pos, intervals in tracks:
            for lo, hi in sorted(intervals):
                if self.lines and self.lines[-1][:2] == (axis, pos) and lo <= self.lines[-1][3]:
                    lo, hi = self.lines[-1][2], max(hi, self.lines[-1][3])
                    self.lines.pop()
                self.lines.append((axis, pos, lo, hi))

    def ruled_empty_rows(self, grid, empty_indices):
        """用横线和相交竖线验证可填写的空行，不要求框线两边都有文字。"""
        if grid is None or not empty_indices or not all(self.ruling):
            return set()
        boxes, logic = grid
        boxes = _normalize_cell_bboxes(boxes)
        empty = set(empty_indices)
        cells_by_row = defaultdict(list)
        for box, (r0, r1, _, _) in zip(boxes, logic):
            if r0 == r1 and r0 in empty:
                cells_by_row[r0].append(box)
        supported = set()
        for row, cells in cells_by_row.items():
            top, bottom = np.median(cells, axis=0)[1::2]
            left = min(c[0] for c in cells)
            right = max(c[2] for c in cells)
            if right <= left or bottom <= top:
                continue
            # 分割模型的框可能止于线内侧，按行高容许向外的小间隙。
            tolerance = min(.4 * self.em, .25 * (bottom - top))
            outer = .3 * (bottom - top)
            top_lines = [(lo, hi) for y, lo, hi in self.ruling[1] if top - outer <= y <= top + tolerance]
            bottom_lines = [(lo, hi) for y, lo, hi in self.ruling[1] if bottom - tolerance <= y <= bottom + outer]
            # 两条横线需覆盖同一个足够宽的区段，且至少两条竖线贯穿此行。
            for a, b in top_lines:
                for c, d in bottom_lines:
                    lo, hi = max(a, c, left - tolerance), min(b, d, right + tolerance)
                    if hi - lo < max(2.5 * self.em, .5 * (right - left)):
                        continue
                    crossings = [x for x, y0, y1 in self.ruling[0]
                                 if lo - tolerance <= x <= hi + tolerance
                                 and y0 <= top + tolerance and y1 >= bottom - tolerance]
                    if crossings and max(crossings) - min(crossings) >= 2.5 * self.em:
                        supported.add(row)
                        break
                if row in supported:
                    break
        return supported

    def stacked_table_boundary(self, rows, height, width):
        """识别双横线两侧列结构不同、各自有分组表头的独立表段。"""
        em = self.em
        horizontal = sorted(self.ruling[1])
        full = [y for y, left, right in horizontal if right - left >= .85 * width]
        partial = [(y, left, right) for y, left, right in horizontal
                   if .12 * width <= right - left <= .5 * width]
        if len(full) < 4 or len(partial) < 4:
            return None

        def header_groups(top, bottom):
            # 分组表头下方有多条并排的子横线。在它们与下一条全宽线之间取子表头。
            for y in sorted({p[0] for p in partial if top < p[0] < min(bottom, top + 2 * em)}):
                spans = sorted((left, right) for pos, left, right in partial if abs(pos - y) <= .15 * em)
                groups = []
                for left, right in spans:
                    if groups and left < groups[-1][1]:
                        groups[-1][1] = max(groups[-1][1], right)
                    else:
                        groups.append([left, right])
                if len(groups) < 2:
                    continue
                labels = []
                for left, right in groups:
                    members = [(box[0], rows[1][i]) for i, box in zip(self.ids, self.boxes)
                               if y < (box[1] + box[3]) / 2 < bottom
                               and left <= (box[0] + box[2]) / 2 <= right]
                    labels.append(tuple(''.join(text.casefold().split()) for _, text in sorted(members)))
                if len(labels[0]) >= 2 and all(label == labels[0] for label in labels):
                    return labels[0], len(labels)
            return None

        first_top = full[0]
        first_bottom = next((y for y in full if y > first_top + em), None)
        if first_top > 2 * em or first_bottom is None:
            return None
        upper = header_groups(first_top, first_bottom)
        if upper is None:
            return None
        splits = []
        for a, b in zip(full, full[1:]):
            cut = (a + b) / 2
            if b - a > .5 * em or min(cut, height - cut) < 5 * em:
                continue
            if any(.1 * width < x < .9 * width and y0 < cut - .5 * em and y1 > cut + .5 * em
                   for x, y0, y1 in self.ruling[0]):
                continue
            if any(box[1] < cut < box[3] for box in self.boxes):
                continue
            end = next((y for y in full if y > b + em), None)
            lower = header_groups(b, end) if end is not None else None
            if lower and lower[0] == upper[0] and lower[1] != upper[1]:
                splits.append(round(cut))
        return splits[0] if len(splits) == 1 else None

    def unsupported_row_splits(self, main, candidate, main_tokens, candidate_tokens):
        main_boxes = _normalize_cell_bboxes(main[0])
        candidate_boxes = _normalize_cell_bboxes(candidate[0])
        main_logic = np.asarray(main[1])
        candidate_logic = np.asarray(candidate[1])
        tracks = list(_boundary_tracks(candidate_boxes, candidate_logic, 1, self.em))
        boxes_by_id = dict(zip(self.ids, self.boxes))
        grouped = defaultdict(list)
        for token, cells in main_tokens.items():
            for cell in set(cells):
                if token in boxes_by_id:
                    grouped[cell].append(token)
        witnesses = defaultdict(list)
        for cell, ids in grouped.items():
            if not 0 <= cell < len(main_logic) or main_logic[cell, 0] != main_logic[cell, 1]:
                continue
            bounds = main_boxes[cell]
            assigned = []
            for token in ids:
                cells = set(candidate_tokens.get(token, ()))
                if len(cells) != 1:
                    continue
                other = next(iter(cells))
                if not 0 <= other < len(candidate_logic):
                    continue
                row, last, _, _ = candidate_logic[other]
                if row != last:
                    continue
                box = boxes_by_id[token]
                overlap = np.maximum(0, np.minimum(bounds[2:], box[2:]) - np.maximum(bounds[:2], box[:2]))
                if np.prod(overlap) >= .65 * np.prod(box[2:] - box[:2]):
                    assigned.append(((box[1] + box[3]) / 2, row, box))
            assigned.sort(key=lambda item: item[0])
            # 同一主候选格的相邻文字被拆到两行时，记录局部边界证据。
            for (y0, before, upper), (y1, after, lower) in zip(assigned, assigned[1:]):
                if after == before + 1 and y1 - y0 >= .5 * self.em:
                    witnesses[after].append((upper[3], lower[1], bounds[0], bounds[2]))

        occupied = {i for token, cells in candidate_tokens.items() if token in boxes_by_id for i in cells}
        spans = [(box, (start, end, left, right))
                 for i, (box, (start, end, left, right)) in enumerate(zip(candidate_boxes, candidate_logic))
                 if i in occupied and start < end]
        group_ends = {(end, left, right) for _, (_, end, left, right) in spans}
        unsupported = 0
        for row, segments in witnesses.items():
            if len(segments) < 2:
                continue
            # 带内容的跨行格可以为相邻列的多条明细提供分组锚点。
            if any((start < row <= end or start == row and (row - 1, c0, c1) in group_ends)
                   and all(min(box[2], right) <= max(box[0], left) + .2 * self.em
                           for _, _, left, right in segments)
                   for box, (start, end, c0, c1) in spans):
                continue
            ruled = sum(any(
                min(top, bottom) - .2 * self.em <= y <= max(top, bottom) + .2 * self.em
                and min(hi, right) - max(lo, left) >= .5 * (right - left)
                and any(abs(edge - y) <= .5 * self.em and min(b, hi, right) > max(a, lo, left)
                        for edge, a, b in tracks)
                for y, lo, hi in self.ruling[1]
            ) for top, bottom, left, right in segments)
            gap = min(s[1] for s in segments) - max(s[0] for s in segments)
            unsupported += ruled * 2 < len(segments) and gap < .55 * self.em
        return unsupported

    def score(self, grid, token_cells=None):
        if grid is None:
            return None
        boxes, logic = grid
        try:
            boxes, logic = _normalize_cell_bboxes(boxes), np.asarray(logic, dtype=float)
        except (TypeError, ValueError):
            return None
        if not len(boxes) or logic.shape != (len(boxes), 4) or not np.isfinite(boxes).all():
            return None
        if not np.isfinite(logic).all() or np.any(logic < 0) or np.any(logic[:, ::2] > logic[:, 1::2]):
            return None
        exact = token_cells is not None
        assignments, text, missing_numbers, missing, weak_numbers = [], {}, set(), set(), set()
        for i, (token_id, box, confidence) in enumerate(zip(self.ids, self.boxes, self.confidence)):
            if exact:
                cells = tuple(c for c in token_cells.get(token_id, ()) if 0 <= c < len(boxes))
                coverage = 0.
                if len(cells) == 1:
                    region = boxes[cells[0]]
                    overlap = np.maximum(0, np.minimum(region[2:], box[2:]) - np.maximum(region[:2], box[:2]))
                    coverage = float(np.prod(overlap) / np.prod(box[2:] - box[:2]))
            else:
                overlap = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]))
                ratios = np.prod(overlap, axis=1) / np.prod(box[2:] - box[:2])
                best = int(ratios.argmax())
                coverage = float(ratios[best])
                cells = (best,) if coverage >= .65 else ()
            assignments.append(cells)
            text[token_id] = 0.
            if not cells:
                text[token_id] = confidence * (1 if exact else .35)
                if exact:
                    missing.add(token_id)
                if exact and i in self.numbers:
                    missing_numbers.add(token_id)
            elif len(cells) == 1 and coverage < .7:
                # 单格只覆盖部分 OCR 框时，框与内容可能错位，或候选边界切穿文字。
                text[token_id] = confidence * (1 - coverage)
                if exact and i in self.numbers and coverage < .5:
                    weak_numbers.add(token_id)
        gaps = {}
        for axis, a, b, separated in self.gaps:
            left, right = assignments[a], assignments[b]
            if not left or not right:
                continue
            start = 2 if axis == 0 else 0
            left_spans, right_spans = logic[list(left), start:start + 2], logic[list(right), start:start + 2]
            if separated:
                conflicting = max(left_spans[:, 1]) >= min(right_spans[:, 0])
            else:
                # 同一视觉行的并排文本应共享逻辑行，防止错位行边界靠增加分隔得分。
                conflicting = (max(left_spans[:, 1]) < min(right_spans[:, 0])
                               or max(right_spans[:, 1]) < min(left_spans[:, 0]))
            gaps[axis, a, b, separated] = min(self.confidence[a], self.confidence[b]) if conflicting else 0.

        tracks = [sorted(_boundary_tracks(boxes, logic, axis, self.em)) for axis in (0, 1)]
        positions = [[pos for pos, _, _ in track] for track in tracks]
        lines = {}
        for i, (axis, pos, lo, hi) in enumerate(self.lines):
            tolerance = .5 * self.em
            first = bisect_left(positions[axis], pos - tolerance)
            last = bisect_right(positions[axis], pos + tolerance)
            covered = _interval_length((max(lo, a), min(hi, b)) for _, a, b in tracks[axis][first:last]
                                       if min(hi, b) > max(lo, a))
            # 只有真实线覆盖的区段计分。较短的半框线仍可独立支持局部分隔。
            weight = min(1., (hi - lo) / (5 * self.em))
            lines[axis, i] = weight * max(0., .8 - covered / (hi - lo))
        return EvidenceScore(text, lines, gaps, exact, missing_numbers, len(self.ids), missing, weak_numbers)
