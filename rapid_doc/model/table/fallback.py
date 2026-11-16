"""Opt-in table recovery; independent of the Rust package."""
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
import re
from statistics import median


def _counts(text):
    return Counter(c for c in text if not c.isspace())


def _table_text(html, row_widths=None):
    stack, active, text = [], [], []
    row = col = cells = tables = last = 0
    width = None
    for token in re.finditer(r"<[^>]*>", html or "", re.S):
        chunk = html[last:token.start()]
        if any(s in ("td", "th") for s in stack):
            text.append(chunk)
        elif unescape(chunk).strip():
            return None
        last = token.end()
        inner = token.group()[1:-1].strip()
        closing = inner.startswith("/")
        parts = inner.lstrip("/").split()
        if not parts:
            return None
        name = parts[0].rstrip("/").lower()
        if closing:
            if not stack or stack.pop() != name:
                return None
            if name == "tr":
                while True:
                    end = next((b for a, b, end in active if end > row and a <= col < b), None)
                    if end is None:
                        break
                    col = end
                if not col or (row_widths is None and width is not None and width != col):
                    return None
                if row_widths is not None:
                    row_widths.append(col)
                width = col
                row += 1
        else:
            if name == "table":
                tables += 1
                if tables != 1 or "table" in stack:
                    return None
            elif name == "tr":
                if not stack or stack[-1] not in ("table", "thead", "tbody", "tfoot"):
                    return None
                col = 0
                active = [v for v in active if v[2] > row]
            elif name in ("td", "th"):
                if not stack or stack[-1] != "tr":
                    return None
                while True:
                    end = next((b for a, b, _ in active if a <= col < b), None)
                    if end is None:
                        break
                    col = end
                rs = cs = 1
                for key, value in re.findall(r'''\b(rowspan|colspan)\s*=\s*["']?([0-9]+)''', inner, re.I):
                    n = int(value)
                    if not 0 < n <= 2**64-1:
                        return None
                    if key.lower() == "rowspan":
                        rs = n
                    else:
                        cs = n
                end = col + cs
                if end > 2**64-1 or row + rs > 2**64-1 or any(a < end and col < b for a, b, _ in active):
                    return None
                active.append((col, end, row + rs))
                col = end
                cells += 1
            elif name in ("thead", "tbody", "tfoot"):
                if not stack or stack[-1] != "table":
                    return None
            elif name not in ("html", "body") and not any(s in ("td", "th") for s in stack):
                return None
            if name not in ("br", "img", "hr", "wbr", "meta", "input"):
                stack.append(name)
    if stack or tables != 1 or not cells or not row or any(v[2] > row for v in active) or (html or "")[last:].strip():
        return None
    return "".join(text)


def repair_local_grid(html):
    """修复少数行中多出来的空占位格，同时保留有内容的格与合并关系。"""
    from bs4 import BeautifulSoup

    widths = []
    if _table_text(html, widths) is None or len(widths) < 3:
        return None
    width, count = Counter(widths).most_common(1)[0]
    affected = [i for i, w in enumerate(widths) if w != width]
    if not affected or count < .9 * len(widths):
        return None
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.find_all('tr')
    if len(rows) != len(widths):
        return None
    # 只接受唯一可解释的修正。多个空格都可删时，无法确定原来的列归属。
    for i in affected:
        excess = widths[i] - width
        if excess != 1:
            return None
        removable = [c for c in rows[i].find_all(['td', 'th'], recursive=False)
                     if str(c.get('rowspan', 1)) == '1' and str(c.get('colspan', 1)) == '1'
                     and not c.get_text(strip=True)
                     and all(t.name in ('br', 'wbr') for t in c.find_all(True))]
        if len(removable) != 1:
            return None
        removable[0].decompose()
    candidate = str(soup)
    if _table_text(candidate) is None:
        return None
    return candidate


def restore_trailing_row(primary, alternative, primary_grid, candidate_grid, primary_tokens,
                         candidate_tokens, source, lost):
    """用主候选保留的整宽尾行补回候选裁掉的文字，表内缺失留给归属判断。"""
    from bs4 import BeautifulSoup
    from .table_evidence import _normalize_cell_bboxes

    if not lost or not all(g is not None for g in (primary_grid, candidate_grid, primary_tokens, candidate_tokens)):
        return None
    boxes, logic = primary_grid
    alt_boxes, alt_logic = candidate_grid
    if not len(logic) or not len(alt_logic):
        return None
    row = max(p[1] for p in logic)
    width = max(p[3] for p in logic) + 1
    terminal = [i for i, p in enumerate(logic) if p[1] == row]
    if len(terminal) != 1 or list(logic[terminal[0]]) != [row, row, 0, width - 1]:
        return None
    cell_id = terminal[0]
    assigned = {i for i, cells in primary_tokens.items() if cell_id in cells}
    if assigned != lost or any(tuple(primary_tokens[i]) != (cell_id,) for i in assigned):
        return None
    # 只有全部遗失的文字都位于备用网格的最下方，才允许补接独立尾行。
    cutoff = max(_normalize_cell_bboxes(alt_boxes)[:, 3])
    token_boxes = [b for i, b in zip(source.ids, source.boxes) if i in lost]
    if len(token_boxes) != len(lost) or min(b[1] for b in token_boxes) < cutoff - .2 * source.em:
        return None
    primary_rows = BeautifulSoup(primary, 'html.parser').find_all('tr')
    tail = primary_rows[-1].find_all(['td', 'th'], recursive=False) if primary_rows else []
    soup = BeautifulSoup(alternative, 'html.parser')
    candidate_rows = soup.find_all('tr')
    alt_row = max(p[1] for p in alt_logic)
    if len(primary_rows) != row + 1 or len(candidate_rows) != alt_row + 1 or len(tail) != 1:
        return None
    alt_width = max(p[3] for p in alt_logic) + 1
    new_row = primary_rows[-1]
    tail[0]['colspan'] = str(alt_width)
    candidate_rows[-1].insert_after(new_row)
    html = str(soup)
    if _table_text(html) is None:
        return None
    combined_grid = ([*_normalize_cell_bboxes(alt_boxes), _normalize_cell_bboxes(boxes)[cell_id]],
                     [*alt_logic, [alt_row + 1, alt_row + 1, 0, alt_width - 1]])
    tokens = {**candidate_tokens, **{i: (len(alt_boxes),) for i in lost}}
    return html, combined_grid, tokens


def restore_internal_rows(html, grid, tokens, source, missing, ocr_rows):
    """为候选网格中完全漏掉的连续文字带补行，保留跨越空档的合并格。"""
    import numpy as np
    from bs4 import BeautifulSoup
    from .table_evidence import _normalize_cell_bboxes

    if len(missing) < 6 or _table_text(html) is None:
        return None
    boxes, logic = grid
    boxes = _normalize_cell_bboxes(boxes)
    logic = np.asarray(logic, dtype=int)
    # HTML 与逻辑网格必须逐格对应，否则无法安全调整 rowspan。
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.find_all('tr')
    if len(rows) != int(logic[:, 1].max()) + 1:
        return None
    order = sorted(range(len(logic)), key=lambda i: (logic[i, 0], logic[i, 2]))
    nodes = [c for r in rows for c in r.find_all(['td', 'th'], recursive=False)]
    if len(nodes) != len(order):
        return None
    nodes = dict(zip(order, nodes))
    if any((str(nodes[i].get('rowspan', 1)), str(nodes[i].get('colspan', 1)))
           != (str(r1 - r0 + 1), str(c1 - c0 + 1))
           for i, (r0, r1, c0, c1) in enumerate(logic)):
        return None
    ready = {i: b for i, b, confidence in zip(source.ids, source.boxes, source.confidence)
             if i in missing and confidence >= .85}
    if set(ready) != missing:
        return None
    width = int(logic[:, 3].max()) + 1
    if width > len(logic):
        return None
    # 同列坐标由多个独立单元格共同确定，避免把跨列框当作单列使用。
    column_boxes = {}
    for c in range(width):
        same = boxes[(logic[:, 2] == c) & (logic[:, 3] == c)]
        if len(same):
            column_boxes[c] = np.median(same[:, ::2], axis=0)
    if len(column_boxes) != width:
        return None
    positioned = []
    for i, b in ready.items():
        matches = [c for c, (l, r) in column_boxes.items()
                   if min(r, b[2]) - max(l, b[0]) >= .75 * (b[2] - b[0])]
        if len(matches) != 1:
            return None
        positioned.append(((b[1]+b[3])/2, matches[0], i))
    bands = []
    for y, col, i in sorted(positioned):
        if not bands or y - bands[-1][0] > .4 * source.em:
            bands.append((y, {}))
        band = bands[-1][1]
        if col in band:
            return None
        band[col] = i
    if len(bands) < 2 or any(len(band) < 2 for _, band in bands):
        return None
    # 至少两列重复对齐。仅一段被拆散的多行文字不能证明存在数据行。
    repeated = Counter(col for _, band in bands for col in band)
    if sum(n >= 2 for n in repeated.values()) < 2:
        return None
    ymin = min(b[1] for b in ready.values())
    ymax = max(b[3] for b in ready.values())
    slots = []
    for r in range(1, len(rows)):
        above = boxes[(logic[:, 0] == r-1) & (logic[:, 1] == r-1)]
        below = boxes[(logic[:, 0] == r) & (logic[:, 1] == r)]
        if len(above) and len(below):
            a, b = float(np.median(above[:, 3])), float(np.median(below[:, 1]))
            if a <= ymin and ymax <= b and b-a >= 1.5 * source.em:
                slots.append(r)
    if len(slots) != 1:
        return None
    r = slots[0]
    spanning = np.flatnonzero((logic[:, 0] < r) & (logic[:, 1] >= r))
    occupied = {c for j in spanning for c in range(logic[j, 2], logic[j, 3]+1)}
    if occupied & repeated.keys():
        return None
    rebuilt_tokens = dict(tokens)
    new_logic = logic.copy()
    new_logic[:, :2] += (new_logic[:, :2] >= r) * len(bands)
    for j in spanning:
        nodes[j]['rowspan'] = str(new_logic[j, 1] - new_logic[j, 0] + 1)
    new_boxes, new_spans = list(boxes), list(new_logic)
    for offset, (_, band) in enumerate(bands):
        tr = soup.new_tag('tr')
        top = min(ready[i][1] for i in band.values())
        bottom = max(ready[i][3] for i in band.values())
        for col, (left, right) in column_boxes.items():
            if col in occupied:
                continue
            td = soup.new_tag('td')
            if col in band:
                i = band[col]
                td.string = unescape(ocr_rows[1][i])
                rebuilt_tokens[i] = (len(new_boxes),)
            tr.append(td)
            new_boxes.append([left, top, right, bottom])
            new_spans.append([r+offset, r+offset, col, col])
        rows[r].insert_before(tr)
    result = str(soup)
    if _table_text(result) is None:
        return None
    return result, (new_boxes, new_spans), rebuilt_tokens


class _CellText(HTMLParser):
    # 网格不合法时仍统计能读到的单元格文字，排除表格外的内容。
    def __init__(self, html):
        super().__init__()
        self.in_cell = False
        self.text = []
        self.feed(html or "")
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th", "tr", "table"):
            self.in_cell = False

    def handle_data(self, data):
        if self.in_cell:
            self.text.append(data)


@dataclass
class Quality:
    valid: bool
    retained: int
    total: int

    def needs_fallback(self, threshold):
        return self.total > 0 and (not self.valid or self.retained == 0 or self.total - self.retained > self.total * threshold)

    def can_replace(self, primary, threshold, fallback_to_text):
        # When prose recovery is disabled, retain the better valid table even
        # if both candidates still lose more text than the trigger threshold.
        return self.valid and self.retained > primary.retained and (
            not fallback_to_text or not self.needs_fallback(threshold)
        )


def quality(html, rows, excluded=()):
    expected = Counter()
    for text, score in zip(rows[1], rows[2]):
        if score >= .5 and text not in excluded:
            expected.update(_counts(unescape(text)))
    parsed = _table_text(html)
    ordinary = unescape(parsed) if parsed is not None else "".join(_CellText(html).text)
    for text in excluded:
        ordinary = ordinary.replace(unescape(text), "")
    actual = _counts(ordinary)
    return Quality(parsed is not None, sum((expected & actual).values()), sum(expected.values()))


_NUMBER_PATTERN = r'[+−-]?\d+(?:,\d{3})*(?:\.\d+)?%?'
_VALUE_PATTERN = rf'(?:\({_NUMBER_PATTERN}\)|{_NUMBER_PATTERN})'
_NUMBER = re.compile(rf'(?<![A-Za-z0-9_.,]){_VALUE_PATTERN}(?![A-Za-z0-9_]|[.,]\d)')
_NUMERIC_CELL = re.compile(_VALUE_PATTERN)

def join_aligned_row_bands(html, grid, assignments, source, rows):
    """用重复出现的跨行锚点和文字对齐恢复多行单元格。"""
    import numpy as np
    from bs4 import BeautifulSoup
    from .table_evidence import _normalize_cell_bboxes

    boxes, logic = grid
    boxes = _normalize_cell_bboxes(boxes)
    logic = np.asarray(logic)
    if not len(logic) or not assignments:
        return None
    width = int(logic[:, 3].max()) + 1
    if width < 3 or width > len(logic) or not np.any(logic[:, 1] > logic[:, 0]):
        return None

    bands = []
    for i, box in sorted(zip(source.ids, source.boxes), key=lambda pair: (pair[1][1] + pair[1][3]) / 2):
        y = (box[1] + box[3]) / 2
        if not bands or y - bands[-1][0] > .4 * source.em:
            bands.append((y, []))
        bands[-1][1].append((i, box, unescape(rows[1][i])))

    def pair_record(first, second):
        y0, upper = first
        y1, lower = second
        upper = sorted(upper, key=lambda t: t[1][0])
        if len(upper) != width or not 2 <= len(lower) < width:
            return None
        # 文字框边缘可重叠，横线和文字中心共同决定是否存在真实行边界。
        if not .6 * source.em <= y1 - y0 <= 1.6 * source.em:
            return None
        positions = [(box[0] + box[2]) / 2 for _, box, _ in upper]
        matched = {}
        for i, box, text in lower:
            center = (box[0] + box[2]) / 2
            fits = [j for j, x in enumerate(positions) if abs(center - x) <= .65 * source.em]
            if len(fits) != 1 or fits[0] in matched:
                return None
            upper_box = upper[fits[0]][1]
            left, right = min(box[0], upper_box[0]), max(box[2], upper_box[2])
            if any(y0 < y < y1 and min(hi, right) - max(lo, left) > .5 * (right - left)
                   for y, lo, hi in source.ruling[1]):
                return None
            matched[fits[0]] = (i, text)

        # 至少一个只有首行文字的列，已被候选明确识别为跨两条文字带。
        # 仅凭第二行缺几个值，无法区分多行单元格和真正的稀疏数据行。
        anchors = set()
        cell_rows = []
        for col, (i, _, _) in enumerate(upper):
            cells = assignments.get(i, ())
            if len(cells) != 1:
                return None
            cell = cells[0]
            r0, r1, c0, c1 = logic[cell]
            if c0 != col or c1 != col:
                return None
            cell_rows.append(int(r0))
            if col not in matched and r1 == r0 + 1 and boxes[cell][1] <= y0 < y1 <= boxes[cell][3]:
                anchors.add(col)
        return (upper, matched, positions, anchors, min(cell_rows)) if anchors else None

    records = []
    # 从末尾回溯一段连续重复的记录，保留前面的表头。
    for j in range(len(bands) - 2, 0, -2):
        record = pair_record(bands[j], bands[j + 1])
        if record is None or record[4] == 0 or (records and record[1].keys() != records[-1][1].keys()):
            break
        records.append(record)
    if len(records) < 3:
        return None
    records.reverse()
    anchors = set.intersection(*(record[3] for record in records))
    if not anchors or any(
            abs(x - ref) > .8 * source.em
            for record in records[1:] for x, ref in zip(record[2], records[0][2])):
        return None
    header_end = records[0][4]
    records = [(upper, matched) for upper, matched, *_ in records]
    # 保留原候选的表头。只有每个高置信 OCR 实例都能归到保留表头或重建记录时才采用。
    soup = BeautifulSoup(html, 'html.parser')
    tr = soup.find_all('tr')
    if not 0 < header_end < len(tr) or any(r0 < header_end <= r1 for r0, r1, _, _ in logic):
        return None
    data_ids = {i for mean, _ in records for i, *_ in mean}
    data_ids.update(i for _, matched in records for i, _ in matched.values())
    if any(not assignments.get(i) or any(logic[j, 1] >= header_end for j in assignments[i])
           for i in source.ids if i not in data_ids):
        return None
    insertion = tr[header_end]
    for mean, matched in records:
        row = soup.new_tag('tr')
        for col, (_, _, text) in enumerate(mean):
            cell = soup.new_tag('td')
            cell.append(text)
            if col in matched:
                cell.append(soup.new_tag('br'))
                cell.append(matched[col][1])
            row.append(cell)
        insertion.insert_before(row)
    for row in tr[header_end:]:
        row.decompose()
    result = str(soup)
    return result if _table_text(result) is not None else None


def _compact(text):
    return ''.join(text.split())


def keeps_candidate_numbers(main, alternative, rows, primary_score, candidate_score):
    deficits = main - alternative
    if not (candidate_score and primary_score and candidate_score.exact and primary_score.exact):
        return not deficits
    lost = candidate_score.missing_numbers - primary_score.missing_numbers
    if lost - primary_score.weak_numbers:
        return False
    # 仅扣除已证实归属不可靠的实例。相同数值在其他格的缺失仍由计数保护。
    weak_losses = Counter(_compact(unescape(rows[1][token])) for token in lost)
    return not (deficits - weak_losses)


@dataclass
class GridEvidence:
    conflicts: set[tuple[int, int]]
    supported: set[tuple[int, int]]
    width: int

    def compare(self, primary):
        """Count gains and losses on the same OCR pairs. Missing evidence cannot earn a gain."""
        return (len(primary.conflicts & self.supported),
                len(primary.supported - self.supported))


def grid_evidence(grid, rows, excluded=()):
    """Check the candidate against separations inferred from OCR geometry."""
    if grid is None:
        return None
    import numpy as np
    from .rapid_table_self.table_matcher.main import _normalize_cell_bboxes
    from .rapid_table_self.table_structure.table_transformer.postprocess import slot_into_containers

    boxes, logic = grid
    boxes = _normalize_cell_bboxes(boxes)
    logic = np.asarray(logic)
    if not len(boxes) or logic.shape != (len(boxes), 4) or not np.isfinite(boxes).all():
        return None
    tokens = []
    for index, (points, text, score) in enumerate(zip(*rows)):
        if points is None or not len(points) or not text.strip() or text in excluded or score < .5:
            continue
        x, y = zip(*points)
        box = [min(x), min(y), max(x), max(y)]
        if box[2] > box[0] and box[3] > box[1]:
            tokens.append({'bbox': box, 'index': index})
    if not tokens:
        return GridEvidence(set(), set(), int(logic[:, 3].max()) + 1)
    _, assignments, _ = slot_into_containers([{'bbox': box} for box in boxes], tokens, .65)
    em = median(t['bbox'][3] - t['bbox'][1] for t in tokens)
    bands = []
    for j in sorted(range(len(tokens)), key=lambda j: sum(tokens[j]['bbox'][1::2])):
        center = sum(tokens[j]['bbox'][1::2]) / 2
        if not bands or center - bands[-1][0] > .35 * em:
            bands.append((center, []))
        bands[-1][1].append(j)

    gaps = []
    for row_index, (_, band) in enumerate(bands):
        band.sort(key=lambda j: tokens[j]['bbox'][0])
        for a, b in zip(band, band[1:]):
            left, right = tokens[a]['bbox'], tokens[b]['bbox']
            if right[0] - left[2] >= .5 * em:
                gaps.append((right[0], row_index, a, b))
    conflicts, supported = set(), set()
    gaps.sort()
    positions = [gap[0] for gap in gaps]
    for x, _, a, b in gaps:
        start = bisect_left(positions, x - .6 * em)
        end = bisect_right(positions, x + .6 * em)
        supporting = {other_row for _, other_row, _, _ in gaps[start:end]}
        if len(supporting) < 3 or not assignments[a] or not assignments[b]:
            continue
        ca, cb = assignments[a][0], assignments[b][0]
        pair = tokens[a]['index'], tokens[b]['index']
        if ca == cb:
            conflicts.add(pair)
        elif logic[ca, 3] < logic[cb, 2]:
            supported.add(pair)
    for i, (y, band) in enumerate(bands[:-1]):
        next_y, next_band = bands[i + 1]
        if next_y - y < 1.25 * em:
            continue
        for a in band:
            for b in next_band:
                left, right = tokens[a]['bbox'], tokens[b]['bbox']
                if (abs(left[0] - right[0]) > .4 * em or right[1] < left[3]
                        or not assignments[a] or not assignments[b]):
                    continue
                ca, cb = assignments[a][0], assignments[b][0]
                pair = tokens[a]['index'], tokens[b]['index']
                if ca == cb:
                    conflicts.add(pair)
                elif logic[ca, 1] < logic[cb, 0]:
                    supported.add(pair)
    return GridEvidence(conflicts, supported, int(logic[:, 3].max()) + 1)


@dataclass
class StructureEvidence:
    split_lines: int
    merged_values: int
    numbers: Counter
    full_width_rows: int
    row_misalignment: int
    collapsed_rows: int
    row_count: int
    empty_row_run: int
    empty_rows: int = 0
    empty_indices: list[int] = field(default_factory=list)

    @property
    def suspicious(self):
        return self.suspicious_with_ruling(())

    def suspicious_with_ruling(self, ruled_rows):
        return (self.split_lines >= 2 or self.merged_values + self.row_misalignment >= 2
                or self.unruled_empty_run(ruled_rows) >= 4)

    def unruled_empty_run(self, ruled_rows):
        longest = run = 0
        previous = -2
        for row in self.empty_indices:
            if row in ruled_rows:
                continue
            run = run + 1 if row == previous + 1 else 1
            longest = max(longest, run)
            previous = row
        return longest

    def improves(self, primary):
        # A few extra rows can be justified when the alternative resolves
        # repeated cross-column row conflicts and retains the numeric values.
        if (primary.row_misalignment >= 5 and self.row_misalignment <= 1
                and self.row_count <= 1.1 * primary.row_count
                and self.split_lines <= primary.split_lines
                and self.merged_values <= primary.merged_values
                and self.numbers == primary.numbers):
            return True
        if (self.row_count > primary.row_count
                and primary.collapsed_rows - self.collapsed_rows < 2):
            return False
        if (primary.row_misalignment - self.row_misalignment >= 2
                and self.merged_values <= primary.merged_values
                and self.full_width_rows <= primary.full_width_rows + 1):
            return True
        return (self.split_lines <= primary.split_lines
                and self.merged_values <= primary.merged_values
                and self.row_misalignment <= primary.row_misalignment
                and self.full_width_rows <= primary.full_width_rows + 1
                and (primary.split_lines - self.split_lines >= 2
                     or (primary.merged_values - self.merged_values
                         + primary.row_misalignment - self.row_misalignment) >= 2))

    def removes_empty_rows(self, primary, ruled_rows=()):
        return (primary.unruled_empty_run(ruled_rows) >= 4 and self.empty_row_run <= 1
                and self.row_count <= .75 * primary.row_count)


def structure_evidence(html, rows, excluded=()):
    """Use identifiable OCR boxes to check how the HTML grouped their text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or '', 'html.parser')
    cell_texts = []
    cell_rows = []
    row_spans = []
    empty_run = longest_empty_run = occupied_until = 0
    empty_indices = []
    for row_index, row in enumerate(soup.find_all('tr')):
        cells = row.find_all(['td', 'th'], recursive=False)
        spans = [str(cell.get('colspan', '1')) for cell in cells]
        row_spans.append([int(span) if re.fullmatch(r'[0-9]{1,4}', span) else 1 for span in spans])
        occupied = row_index < occupied_until
        for cell in cells:
            text = cell.get_text(' ', strip=True)
            has_content = bool(text or cell.find('img'))
            for marker in excluded:
                text = text.replace(unescape(marker), '')
            cell_texts.append(text)
            cell_rows.append(row_index)
            if has_content:
                occupied = True
                rowspan = str(cell.get('rowspan', '1'))
                if re.fullmatch(r'[0-9]{1,4}', rowspan):
                    occupied_until = max(occupied_until, row_index + int(rowspan))
        empty_run = empty_run + 1 if cells and not occupied else 0
        if empty_run:
            empty_indices.append(row_index)
        longest_empty_run = max(longest_empty_run, empty_run)

    compact_cells = [_compact(text) for text in cell_texts]
    cell_numbers = [_NUMBER.findall(text) for text in cell_texts]
    actual_numbers = Counter(number for values in cell_numbers for number in values)
    number_locations = defaultdict(list)
    for index, numbers in enumerate(cell_numbers):
        for number in set(numbers):
            number_locations[number].append(index)
    by_cell = defaultdict(list)
    located = []
    prose = []
    tokens = []
    for points, text, confidence in zip(*rows):
        if confidence < .5 or not text.strip() or text in excluded or points is None or not len(points):
            continue
        text = unescape(text).strip()
        compact = _compact(text)
        numeric = bool(_NUMERIC_CELL.fullmatch(compact))
        tokens.append((points, compact, numeric))

    frequencies = Counter(compact for _, compact, _ in tokens)
    repeated_numbers = defaultdict(list)
    for token_index, (points, compact, numeric) in enumerate(tokens):
        if numeric and frequencies[compact] > 1:
            x, y = zip(*points)
            repeated_numbers[compact].append((min(y), min(x), token_index))
    number_assignments = {}
    for number, occurrences in repeated_numbers.items():
        locations = number_locations[number]
        # Identical values can be paired in reading order only when each
        # occurrence appears in a distinct cell and none are missing.
        if len(occurrences) == actual_numbers[number] == len(locations):
            number_assignments.update((index, [cell]) for (_, _, index), cell in zip(sorted(occurrences), locations))
    for token_index, (points, compact, numeric) in enumerate(tokens):
        if len(compact) < 2 and not numeric:
            continue
        if frequencies[compact] != 1:
            matches = number_assignments.get(token_index, [])
        elif numeric:
            matches = number_locations[compact]
        else:
            matches = [i for i, cell in enumerate(compact_cells) if compact in cell]
        if len(matches) != 1:
            continue
        x, y = zip(*points)
        box = (min(x), min(y), max(x), max(y))
        index = matches[0]
        if box[2] > box[0] and box[3] > box[1]:
            located.append((box, index, cell_rows[index]))
        by_cell[index].append((box, numeric, len(compact)))
        if not numeric and box[2] - box[0] >= 4 * (box[3] - box[1]):
            prose.append((box[1], cell_rows[index], box))

    # Closely spaced, left-aligned lines in a wide cell can be mistaken for
    # separate rows. Require repeated splits before routing.
    prose.sort()
    split_rows = set()
    for _, row, box in prose:
        height = box[3] - box[1]
        preceding = [
            (prior, previous_row) for y, previous_row, prior in prose
            if y < box[1] and -.2 * height <= box[1] - prior[3] <= .4 * height
            and abs(prior[0] - box[0]) <= .5 * height
        ]
        if preceding and max(preceding, key=lambda item: item[0][1])[1] != row:
            split_rows.add(row)

    merged_values = 0
    for boxes in by_cell.values():
        boxes.sort(key=lambda item: item[0][1])
        conflicting = set()
        separated_value = False
        for i, (a, numeric_a, length_a) in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                b, numeric_b, length_b = boxes[j]
                if b[1] >= a[3]:
                    break
                height = min(a[3] - a[1], b[3] - b[1])
                if height <= 0:
                    continue
                y_overlap = min(a[3], b[3]) - max(a[1], b[1])
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                if numeric_a != numeric_b and y_overlap >= .5 * height and x_overlap < -3 * height:
                    separated_value = True
                if ((numeric_a and numeric_b and y_overlap >= .5 * height and x_overlap < 0)
                        or (max(length_a, length_b) <= 5
                            and y_overlap >= .7 * height and x_overlap < -.6 * height)):
                    conflicting.update((i, j))
        merged_values += max(int(separated_value), len(conflicting) - 1)
    expected = Counter(compact for _, compact, numeric in tokens if numeric)
    numbers = expected & actual_numbers
    width = max(map(sum, row_spans), default=0)
    # Several new full-width rows can hide a column merge behind a better header.
    full_width_rows = row_spans.count([width]) if width >= 3 else 0
    split, collapsed = _row_misalignment(located)
    return StructureEvidence(len(split_rows), merged_values, numbers, full_width_rows,
                             split + collapsed, collapsed, len(row_spans), longest_empty_run,
                             len(empty_indices), empty_indices)


def _row_misalignment(located):
    """Compare occupied HTML rows with aligned text in multiple columns."""
    if not located:
        return 0, 0
    height = median(box[3] - box[1] for box, _, _ in located)
    bands = []
    for box, cell, row in sorted(located, key=lambda item: (item[0][1] + item[0][3]) / 2):
        center = (box[1] + box[3]) / 2
        if not bands or center - bands[-1][0] > .38 * height:
            bands.append((center, []))
        bands[-1][1].append((box, cell, row))

    parallel = defaultdict(list)
    split = 0
    for center, items in bands:
        items.sort(key=lambda item: item[0][0])
        split_band = False
        for i, (left, left_cell, left_row) in enumerate(items):
            for right, right_cell, right_row in items[i + 1:]:
                if right[0] - left[2] < .6 * height:
                    continue
                if left_row != right_row:
                    split_band = True
                elif left_cell != right_cell:
                    parallel[left_cell, right_cell].append(center)
        split += split_band
    # Tight multiline cells are normal. Separated lines repeated across the
    # same pair of cells indicate a stronger row-collapse signal.
    collapsed = max((sum(b - a >= 1.15 * height for a, b in zip(centers, centers[1:]))
                     for centers in parallel.values()), default=0)
    return split, collapsed


class TextFallback(str):
    """Markup carrier; the pipeline records that this is prose, not a table."""


def plain_html(rows):
    # Use all corners: box tops differ slightly even on the same printed line.
    # A fixed seed per line prevents transitive overlaps from joining two rows.
    boxes = [(min(p[0] for p in q), min(p[1] for p in q),
              max(p[0] for p in q), max(p[1] for p in q)) for q in rows[0]]
    order = sorted((i for i, text in enumerate(rows[1]) if text),
                   key=lambda i: ((boxes[i][1]+boxes[i][3])/2, boxes[i][0], i))
    lines = []
    for i in order:
        b = boxes[i]
        center = (b[1]+b[3])/2
        candidates = []
        for k, (seed, _) in enumerate(lines):
            a = boxes[seed]
            height = min(a[3]-a[1], b[3]-b[1])
            distance = abs(center-(a[1]+a[3])/2)
            overlap = min(a[3], b[3])-max(a[1], b[1])
            if height > 0 and distance <= height*.5 and overlap >= height*.5:
                candidates.append((distance/height, k))
        if candidates:
            lines[min(candidates)[1]][1].append(i)
        else:
            lines.append((i, [i]))
    result = []
    for _, indices in lines:
        indices.sort(key=lambda i: (boxes[i][0], (boxes[i][1]+boxes[i][3])/2, i))
        # The Markdown exporter uses a text fence to preserve ordinary spaces.
        result.append('<p>' + '  '.join(rows[1][i] for i in indices) + '</p>\n')
    return TextFallback(''.join(result))


def prose(markup):
    # Keep escaped text and asset <img> elements intact for Markdown rendering.
    return markup.replace("<p>", "").replace("</p>", "\n").replace('\u2003', '  ').strip()


def markdown_prose(markup):
    text = prose(markup)
    # Keep existing rich rendering, including legacy results without hint flags.
    if '<' in text or '$' in text or r'\(' in text or r'\[' in text:
        return text
    text = unescape(text)
    if not text:
        return ''
    # OCR can itself contain Markdown fences. Choose a fence it cannot close.
    length = max([2] + [len(m.group()) for m in re.finditer(r'`+', text)]) + 1
    fence = '`' * length
    return f'{fence}text\n{text}\n{fence}'
