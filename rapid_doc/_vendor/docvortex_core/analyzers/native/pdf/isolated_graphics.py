"""恢复没有原生文字成员的独立矢量图形，装饰背景和正文仍由原规则处理。"""

import statistics
import math
from ...._compute_backend import get_native

from .geometry import _bbox_area, _bbox_distance, _bbox_overlap_in_first, _bbox_overlap_in_smaller, _bbox_union_many
from .line_layout import _line_effective_height


def _isolated_path_groups(source, paths, em):
    """仅缓存当前页的矩形几何与字号；颜色、文字和最终图体仍每次独立核验。"""
    boxes = tuple(path.bbox for path in paths)
    ordinary = (
        type(em) in (int, float)
        and math.isfinite(em)
        and abs(em) <= 2**50
        and all(
            type(box) is tuple
            and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2**50 for v in box)
            for box in boxes
        )
    )
    cache = getattr(source, "isolated_path_cache", {}) if ordinary else {}
    key = (boxes, em) if ordinary else None
    indices = cache.get(key) if ordinary else None
    native = get_native()
    if indices is None and ordinary and native is not None:
        indices = native.isolated_path_groups(boxes, em)
    if indices is None:
        indices = []
        for i, path in enumerate(paths):
            matches = [group for group in indices if any(_bbox_distance(path.bbox, paths[j].bbox) <= 0.35 * em for j in group)]
            if not matches:
                indices.append([i])
            else:
                merged = [i, *(j for group in matches for j in group)]
                indices = [group for group in indices if all(group is not match for match in matches)]
                indices.append(merged)
    if ordinary:
        cache[key] = indices
    return [[paths[i] for i in group] for group in indices]


def isolated_vector_components(source):
    """由复杂二维路径组成独立图片，不认领正文；框缘轻微相交可保留完整多色插画。"""
    heights = [_line_effective_height(line, line.bbox) for line in source.lines if line.angle == 0]
    em = statistics.median(heights) if heights else 10.0
    width, height = source.page_size
    area = width * height
    paths = [
        path
        for path in source.path_infos
        if path.form_depth == 0
        and (path.fill_visible or path.stroke_visible)
        and not path.rectangle_bboxes
        and path.segment_count >= 8
        and 0 < _bbox_area(path.bbox) <= 0.06 * area
    ]
    groups = _isolated_path_groups(source, paths, em)
    output = []
    for group in groups:
        box = _bbox_union_many([path.bbox for path in group])
        if (
            not 0.0005 <= _bbox_area(box) / area <= 0.06
            or min(box[2] - box[0], box[3] - box[1]) < 2.5 * em
            or sum(path.segment_count for path in group) < 30
            or not any(
                path.segment_count >= 20 and min(path.bbox[2] - path.bbox[0], path.bbox[3] - path.bbox[1]) >= 1.5 * em
                for path in group
            )
        ):
            continue
        edges = sum((box[0] <= 0.2 * em, box[1] <= 0.2 * em, box[2] >= width - 0.2 * em, box[3] >= height - 0.2 * em))
        # 多个不同填色的复杂轮廓可形成贴页边的完整插画，单一角饰不能借此扩框。
        colors = {path.fill_rgba for path in group if path.fill_visible}
        if (edges >= 2 and (len(group) < 8 or len(colors) < 3)) or any(
            _bbox_overlap_in_first(line.bbox, box) > 0.1 for line in source.lines
        ):
            continue
        if any(_bbox_overlap_in_smaller(box, raster) > 0.2 for raster in source.image_bboxes):
            continue
        output.append(box)
    return output


def meaningful_small_raster(source, box):
    """小栅格图的两个方向均明显大于正文，且没有覆盖原生字行时保留；薄线及行内字图拒绝。"""
    heights = [_line_effective_height(line, line.bbox) for line in source.lines if line.angle == 0]
    if not heights:
        return False
    em = statistics.median(heights)
    return (
        min(box[2] - box[0], box[3] - box[1]) >= 3 * em
        and _bbox_area(box) >= 0.0005 * source.page_size[0] * source.page_size[1]
        and not any(_bbox_overlap_in_first(line.bbox, box) > 0.01 for line in source.lines)
    )
