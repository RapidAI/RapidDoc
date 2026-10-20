# Portions derived from pdftext 0.7.1, Copyright Vik Paruchuri, Apache-2.0.
# Changed in DocVortex: grouping uses owned dictionaries without upstream container adapters.
"""基础文本行与上下标分组；保留已验证的几何判断。"""

from __future__ import annotations

import math
import unicodedata

from ._contracts import Line, Lines, Spans


def is_math_symbol(char: str) -> bool:
    """判断单字符数学符号。"""
    if len(char) != 1:
        return False

    category = unicodedata.category(char)
    return category == "Sm"


def _top2(values: list[float]) -> tuple[float, int, float]:
    # Returns (max1, max1_idx, max2) so that max-excluding-index can be answered in O(1)
    """在线性时间内找到两个最大值。"""
    max1 = max2 = float("-inf")
    max1_idx = -1
    for idx, v in enumerate(values):
        if v > max1:
            max2 = max1
            max1 = v
            max1_idx = idx
        elif v > max2:
            max2 = v
    return max1, max1_idx, max2


def _bottom2(values: list[float]) -> tuple[float, int, float]:
    """在线性时间内找到两个最小值。"""
    min1 = min2 = float("inf")
    min1_idx = -1
    for idx, v in enumerate(values):
        if v < min1:
            min2 = min1
            min1 = v
            min1_idx = idx
        elif v < min2:
            min2 = v
    return min1, min1_idx, min2


def assign_scripts(lines: Lines, height_threshold: float = 0.8, line_distance_threshold: float = 0.1) -> None:
    """根据邻接片段几何设置基础上下标提示。"""
    for line in lines:
        spans = line["spans"]
        if len(spans) < 2:
            continue

        line_bbox = line["bbox"].bbox
        line_height = line_bbox[3] - line_bbox[1]
        # Skip vertical lines
        if line_height > line_bbox[2] - line_bbox[0]:
            continue

        # Precompute per-span geometry once; the loop below would otherwise
        # recompute these via Bbox properties O(n^2) times per line
        heights = []
        y_starts = []
        y_ends = []
        v_above = []
        v_below = []
        for s in spans:
            bbox = s["bbox"].bbox
            height = bbox[3] - bbox[1]
            heights.append(height)
            y_starts.append(bbox[1])
            y_ends.append(bbox[3])
            v_above.append(bbox[1] - height * line_distance_threshold)
            v_below.append(bbox[3] + height * line_distance_threshold)

        above_max1, above_max1_idx, above_max2 = _top2(v_above)
        below_min1, below_min1_idx, below_min2 = _bottom2(v_below)
        max_line_height = max(1, line_height)
        last_idx = len(spans) - 1

        for i, span in enumerate(spans):
            is_first = i == 0 or not spans[i - 1]["text"].strip()
            is_last = i == last_idx or not spans[i + 1]["text"].strip()
            span_height = heights[i]
            span_top = y_starts[i]
            span_bottom = y_ends[i]

            line_fullheight = span_height / max_line_height <= height_threshold
            next_fullheight = is_last or span_height / max(1, heights[i + 1]) <= height_threshold
            prev_fullheight = is_first or span_height / max(1, heights[i - 1]) <= height_threshold

            # any(span_top < v_above[j] for j != i) == span_top < max(v_above excluding i)
            above = span_top < (above_max2 if i == above_max1_idx else above_max1)
            prev_above = is_first or span_top < y_starts[i - 1]
            next_above = is_last or span_top < y_starts[i + 1]

            below = span_bottom > (below_min2 if i == below_min1_idx else below_min1)
            prev_below = is_first or span_bottom > y_ends[i - 1]
            next_below = is_last or span_bottom > y_ends[i + 1]

            span_text = span["text"].strip()
            span_text_okay = all(
                [
                    (len(span_text) == 1 or span_text.isdigit()),  # Ensure that the span text is a single char or a number
                    span_text.isalnum()
                    or is_math_symbol(span_text),  # Ensure that the span text is an alphanumeric or a math symbol
                ]
            )

            if all([(prev_fullheight or next_fullheight), (prev_above or next_above), above, line_fullheight, span_text_okay]):
                span["superscript"] = True
            elif all(
                [(prev_fullheight or next_fullheight), (prev_below or next_below), below, line_fullheight, span_text_okay]
            ):
                span["subscript"] = True


def get_lines(spans: Spans) -> Lines:
    """按换行、角度和位置将片段聚合为行。"""
    lines: Lines = []
    line: Line = None

    def line_break() -> None:
        """以当前片段开始一个新的文本行。"""
        lines.append({"spans": [span], "bbox": span["bbox"].copy(), "rotation": span["rotation"]})

    for span in spans:
        if lines:
            line = lines[-1]

        if not line:
            line_break()
            continue

        # we break if the previous span ends with a linebreak
        last_text = line["spans"][-1]["text"]
        if any(last_text.endswith(suffix) for suffix in ["\n", "\x02"]):
            line_break()
            continue

        # rotations are radians from FPDFText_GetCharAngle; compare circularly.
        # Only break on roughly perpendicular text: pdfium reports a 180-degree
        # flip for ordinary text rendered with negative-scale matrices, which
        # still belongs to the same visual line
        if span["rotation"] != line["rotation"]:
            rotation_diff = abs(span["rotation"] - line["rotation"]) % (2 * math.pi)
            rotation_diff = min(rotation_diff, 2 * math.pi - rotation_diff)
            if math.radians(45) <= rotation_diff <= math.radians(135):
                line_break()
                continue

        # sometimes pdfium doesn't inject a linebreak, so we check the span positions
        if span["bbox"].y_start > line["bbox"].y_end:
            line_break()
            continue

        line["spans"].append(span)
        line["bbox"].merge_inplace(span["bbox"])

    return lines
