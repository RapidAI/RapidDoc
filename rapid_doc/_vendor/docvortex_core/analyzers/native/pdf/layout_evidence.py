"""按当前阶段的实际文字构建栏证据；不缓存可变成员，也不假设页面中线是栏沟。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ....schema import BBox
from .geometry import _bbox_center_x, _bbox_overlap_in_first, _rotate_bbox_to_upright
from .line_layout import _infer_text_lanes, _line_effective_height
from .models import _LineItem, _TextLane


@dataclass
class LayoutEvidence:
    """保存一个分析阶段的正向栏带、局部尺度与视觉屏障。"""

    lanes: list[_TextLane]
    width: float
    em: float
    barriers: tuple[BBox, ...] = ()
    geometry: tuple = ()

    def corridor(self, bbox: BBox) -> tuple[float, float] | None:
        """用实际栏间空白分配区域，覆盖多栏的块独立使用通栏区域。"""
        if not self.lanes:
            return None
        overlaps = [max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) for lane in self.lanes]
        substantial = [
            i
            for i, overlap in enumerate(overlaps)
            if overlap >= 0.25 * min(bbox[2] - bbox[0], self.lanes[i].right - self.lanes[i].left)
        ]
        if len(substantial) > 1:
            return (0.0, self.width)
        center = _bbox_center_x(bbox)
        edges = [0.0] + [(a.right + b.left) / 2 for a, b in zip(self.lanes, self.lanes[1:])] + [self.width]
        for left, right in zip(edges, edges[1:]):
            if left <= center <= right:
                return left, right
        return None

    def separated(self, first: BBox, second: BBox) -> bool:
        """仅明确属于不同非跨栏区域时阻止认领，缺少证据不创造虚假栏沟。"""
        a, b = self.corridor(first), self.corridor(second)
        if a is not None and b is not None and a != b and (0.0, self.width) not in (a, b):
            return True
        # 全页有通栏正文时，图旁的局部窄栏仍须由同高度附近的重复边缘独立确认。
        left, right = sorted((first, second), key=lambda bounds: bounds[0])
        if right[0] - left[2] <= 0.5 * self.em or min(left[2] - left[0], right[2] - right[0]) < 3 * self.em:
            return False
        top, bottom = min(first[1], second[1]) - 3 * self.em, max(first[3], second[3]) + 3 * self.em
        local = [(line, bounds) for line, bounds in self.geometry if top <= (bounds[1] + bounds[3]) / 2 <= bottom]
        if not local:
            return False
        # 图注有显式首行身份，短尾行也能支持局部栏；不要求它达到正文长行阈值。
        for seed, bounds in local:
            if not seed.caption_start:
                continue
            followers = [
                b
                for line, b in local
                if abs(b[0] - bounds[0]) <= 0.75 * self.em and bounds[1] <= b[1] <= bounds[3] + 4 * self.em
            ]
            if len(followers) >= 2:
                if left[2] <= bounds[0] - 0.5 * self.em and abs(right[0] - bounds[0]) <= self.em:
                    return True
                edge = max(b[2] for b in followers)
                if right[0] >= edge + 0.5 * self.em and abs(left[0] - bounds[0]) <= self.em:
                    return True
        lanes = sorted(
            [lane for lane in _infer_text_lanes(local, self.width, self.em) if not lane.is_span and len(lane.lines) >= 3],
            key=lambda lane: lane.left,
        )
        return any(
            left[2] <= previous.right + self.em and right[0] >= following.left - self.em and previous.right < following.left
            for previous, following in zip(lanes, lanes[1:])
        )


def build_layout_evidence(
    lines: list[_LineItem], page_size: tuple[float, float], *, angle: int = 0, barriers=()
) -> LayoutEvidence:
    """复用稳定左右边缘推断，过滤视觉容器内部标签；成员变化后由调用方重新构建。"""
    width = page_size[1] if angle in {90, 270} else page_size[0]
    geometry = [
        (line, _rotate_bbox_to_upright(line.ink_bbox or line.bbox, page_size, angle))
        for line in lines
        if line.angle == angle and not any(_bbox_overlap_in_first(line.bbox, b) >= 0.8 for b in barriers)
    ]
    em = statistics.median(_line_effective_height(line, bbox) for line, bbox in geometry) if geometry else 10.0
    if not geometry:
        return LayoutEvidence([], width, em, tuple(barriers))
    lanes = _infer_text_lanes(geometry, width, em, recalculate_intervals=False)
    stable = sorted([lane for lane in lanes if not lane.is_span and len(lane.lines) >= 3], key=lambda lane: lane.left)
    # 重叠局部带不能被错误解释为并列栏；优先保留拥有更多成员的栏证据。
    distinct = []
    for lane in stable:
        if distinct and lane.left < distinct[-1].right:
            if len(lane.lines) > len(distinct[-1].lines):
                distinct[-1] = lane
        else:
            distinct.append(lane)
    return LayoutEvidence(distinct, width, em, tuple(barriers), tuple(geometry))
