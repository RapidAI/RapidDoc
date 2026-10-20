"""用真实字形行距恢复字体宽松框失真的中文标题及自然段。"""

import re
import statistics

from .geometry import _bbox_union_many, _bbox_center_y, _bbox_overlap_in_first
from .line_layout import _infer_text_lanes
from .table_filled_grid import has_inflated_native_cjk_metrics
from .title_analysis.structural import _line_has_bold_font


def _visible_line_bbox(line):
    """仅用于本阶段推断的墨迹包络，不改写现有输出框和字符身份。"""
    boxes = [char["tight_bbox"] for char in line.chars or [] if char.get("tight_bbox")]
    return _bbox_union_many(boxes) if boxes else line.ink_bbox or line.bbox


def restore_native_cjk_bands(lines, page_size, containers, page_index):
    """异常度量页以短编号标题、墨迹行距、留白和末行宽度建立持续段界。"""
    chars = [char for line in lines for char in line.chars or []]
    if not has_inflated_native_cjk_metrics(chars):
        return
    cjk = [char for char in chars if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", ""))]
    em = statistics.median(char["bbox"][2] - char["bbox"][0] for char in cjk)
    heading = re.compile(r"^\s*(?:\d+[.、]\s*(?=[\u4e00-\u9fff])|[（(][一二三四五六七八九十]+[）)]|[一二三四五六七八九十]+、)")
    for line in lines:
        if line.angle != 0 or line.caption_start or line.title_suppressed:
            continue
        box = _visible_line_bbox(line)
        if page_index > 0 and line.semantic_type == "doc_title" and len(line.text) > 25 and not _line_has_bold_font(line):
            line.semantic_type = None
        if line.semantic_type is not None or any(_bbox_overlap_in_first(box, container) >= 0.8 for container in containers):
            continue
        numbered = bool(heading.match(line.text)) and len(line.text) <= 40 and not re.search(r"[。！？;；]", line.text)
        display = _line_has_bold_font(line) and 1.5 * em <= box[3] - box[1] and len(line.text) <= 40
        if not numbered and not display:
            continue
        if not 0.07 * page_size[1] <= _bbox_center_y(box) <= 0.92 * page_size[1]:
            continue
        followers = [
            other
            for other in lines
            if other is not line
            and other.angle == 0
            and other.semantic_type is None
            and len(other.text) >= 15
            and 0 < _visible_line_bbox(other)[1] - box[3] <= 8 * em
            and abs(other.bbox[0] - line.bbox[0]) <= 3 * em
        ]
        if not followers and not display:
            continue
        line.semantic_type = "paragraph_title"
        line.structural_title = True
        line.explicit_section_title = True
        line.title_band_id = line.source_index

    eligible = [
        (line, _visible_line_bbox(line))
        for line in lines
        if line.angle == 0
        and line.semantic_type is None
        and line.paragraph_group is None
        and not line.caption_start
        and (
            len(re.findall(r"[\u4e00-\u9fff]", line.text)) >= 2
            # 短末行靠前一满行、相同字体和实际行距认领；孤立数字不会形成正文行组。
            or (len(line.text.strip()) <= 10 and re.search(r"[^\W_]", line.text))
        )
    ]
    if not eligible:
        return
    lanes = _infer_text_lanes(eligible, page_size[0], em)
    for lane in lanes:
        rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
        gaps = [b[1][1] - a[1][1] for a, b in zip(rows, rows[1:]) if em <= b[1][1] - a[1][1] <= 3 * em]
        if len(gaps) < 2:
            continue
        pitch = statistics.median(gaps)
        width = lane.right - lane.left
        run = []
        for line, box in rows:
            if run:
                previous, before = run[-1]
                if (
                    box[1] - before[1] > 1.25 * pitch
                    or abs(box[0] - before[0]) > 1.2 * em
                    or before[2] - before[0] < 0.8 * width
                    or line.font_signature != previous.font_signature
                ):
                    _mark_prose_run(run)
                    run = []
            run.append((line, box))
        _mark_prose_run(run)


def _mark_prose_run(run):
    """只为多行连续正文传递段组，不改变孤立短行或已确认标题。"""
    if len(run) < 2 or sum(len(line.text) for line, _ in run) < 35:
        return
    group = 1_000_000 + run[0][0].source_index
    for line, _ in run:
        line.paragraph_group = group
        line.semantic_type = "text"
        line.title_suppressed = True
