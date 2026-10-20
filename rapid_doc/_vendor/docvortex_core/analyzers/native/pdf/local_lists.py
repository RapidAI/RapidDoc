"""在已经聚合的单个连续正文区域中识别列表，不跨文本块拼凑条目。"""

import re
import statistics

from .geometry import _bbox_union_many
from .line_layout import _line_effective_height
from .text_assembly.common import _merge_text_line_content


def _vector_bullet(line, paths, em):
    """以实心近圆小路径、正文左缘和同排中心证明圆点，不接受分隔线或图表刻度。"""
    return next(
        (
            path.bbox
            for path in paths
            if path.fill_visible
            and not path.stroke_visible
            and not path.rectangle_bboxes
            and 8 <= path.segment_count <= 100
            and 0.12 * em <= path.bbox[2] - path.bbox[0] <= 0.5 * em
            and 0.12 * em <= path.bbox[3] - path.bbox[1] <= 0.5 * em
            and 0.8 <= (path.bbox[2] - path.bbox[0]) / (path.bbox[3] - path.bbox[1]) <= 1.2
            and 0.2 * em <= line.bbox[0] - path.bbox[2] <= 2 * em
            and abs((path.bbox[1] + path.bbox[3] - line.bbox[1] - line.bbox[3]) / 2) <= 0.4 * em
        ),
        None,
    )


def split_local_list_blocks(blocks, paths):
    """完整连续区域内的递增字母或重复圆点恢复条目；续行沿用同区域字符及行框。"""
    output = []
    for block in blocks:
        rows = sorted(block.get("_text_lines", []), key=lambda line: (line.bbox[1], line.bbox[0]))
        if block["type"] != "text" or len(rows) < 3 or any(line.angle for line in rows):
            output.append(block)
            continue
        em = statistics.median(_line_effective_height(line, line.bbox) for line in rows)
        if em <= 0 or any(not 0.85 * em <= _line_effective_height(line, line.bbox) <= 1.15 * em for line in rows):
            output.append(block)
            continue
        # 同一区域内也必须有稳定行距，真实段界和内嵌小字不得被吸入列表。
        if any(not -0.3 * em <= b.bbox[1] - a.bbox[3] <= 0.7 * em for a, b in zip(rows, rows[1:])):
            output.append(block)
            continue
        letters = [re.match(r"^\s*([a-z])[.)]\s+\S", line.text) for line in rows]
        dots = [_vector_bullet(line, paths, em) for line in rows]
        starts = [i for i, match in enumerate(letters) if match]
        lettered = (
            len(starts) >= 3
            and starts[0] == 0
            and all(ord(letters[b][1]) == ord(letters[a][1]) + 1 for a, b in zip(starts, starts[1:]))
        )
        if not lettered:
            starts = [i for i, dot in enumerate(dots) if dot]
        if len(starts) < 3 or starts[0] != 0 or any(abs(rows[i].bbox[0] - rows[0].bbox[0]) > 0.2 * em for i in starts):
            output.append(block)
            continue
        if any(rows[i].font_signature != rows[0].font_signature for i in starts):
            output.append(block)
            continue
        items = []
        for start, end in zip(starts, [*starts[1:], len(rows)]):
            members = rows[start:end]
            if any(not -0.2 * em <= line.bbox[0] - rows[start].bbox[0] <= 2 * em for line in members):
                break
            bounds = [line.bbox for line in members]
            if not lettered:
                bounds.append(dots[start])
            items.append(
                {
                    **block,
                    "type": "text",
                    "content": _merge_text_line_content([line.text for line in members]),
                    "bbox": _bbox_union_many(bounds),
                    "_text_lines": members,
                    "_local_line_bboxes": [line.bbox for line in members],
                    "_local_output_line_bboxes": [line.bbox for line in members],
                    "_line_heights": [_line_effective_height(line, line.bbox) for line in members],
                    "_output_bbox_repaired": False,
                }
            )
        if len(items) != len(starts):
            output.append(block)
            continue
        output.append({"type": "list", "content": "", "bbox": _bbox_union_many([item["bbox"] for item in items]), "angle": 0})
        output.extend(items)
    return output
