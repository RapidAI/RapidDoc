"""统计全文和栏内正文排版基线。"""

from __future__ import annotations

import statistics
import math

from ..geometry import _rotate_bbox_to_upright
from .. import line_layout as _profile_line_layout
from ..inline.types import PDF_FONT_FORCE_BOLD_FLAG, PDF_FONT_ITALIC_FLAG
from ..line_layout import _estimate_lane_gap, _font_signatures_share_family, _line_canonical_style_scale, _line_effective_height
from ..models import _DocumentBodyProfile, _LaneBodyProfile, _LineItem, _PreparedPage, _TextLane


def _plain_profile_number(value) -> bool:
    """仅让有限内置数值进入无回调的阶段缓存，巨大整数保持原异常路径。"""
    return type(value) is float and math.isfinite(value) or type(value) is int and -(2**53) <= value <= 2**53


_CONTEXT_REFERENCE_FUNCTIONS = (
    _plain_profile_number,
    _line_effective_height,
    _profile_line_layout._line_style_scale,
    _LineItem.__getattribute__,
    _TextLane.__getattribute__,
)


class _LaneProfileContext:
    """在一次标题判定中复用只读行尺度；语义写入由调用方立即通知失效。"""

    def __init__(self, lanes, line_geometry=None):
        """只接纳普通栏与行，记录所有身份归属以处理重复成员的失效。"""
        self.profiles = {}
        self.heights = {}
        self.memberships = {}
        self.plain = True
        if line_geometry is not None:
            if type(line_geometry) is not list:
                self.plain = False
                return
            lanes = [*lanes, _TextLane(0.0, 0.0, line_geometry)]
        from ....._compute_backend import get_native

        native = get_native()
        if (
            native is not None
            and (
                _plain_profile_number,
                _line_effective_height,
                _profile_line_layout._line_style_scale,
                _LineItem.__getattribute__,
                _TextLane.__getattribute__,
            )
            == _CONTEXT_REFERENCE_FUNCTIONS
        ):
            packed = native.lane_profile_context_owned(lanes, _TextLane, _LineItem)
            if packed is not None:
                self.heights, self.memberships = packed
                return
        for lane in lanes:
            if (
                type(lane) is not _TextLane
                or type(lane.lines) is not list
                or not all(_plain_profile_number(value) for value in (lane.left, lane.right))
            ):
                self.plain = False
                break
            for row in lane.lines:
                if type(row) is not tuple or len(row) != 2:
                    self.plain = False
                    break
                line, box = row
                if (
                    type(line) is not _LineItem
                    or type(line.text) is not str
                    or type(line.angle) is not int
                    or type(box) not in (tuple, list)
                    or len(box) != 4
                    or not all(_plain_profile_number(value) for value in box)
                    or not all(
                        _plain_profile_number(value) for value in (line.em_height, line.effective_height, line.font_coverage)
                    )
                    or line.dominant_font_weight is not None
                    and not _plain_profile_number(line.dominant_font_weight)
                    or line.semantic_type is not None
                    and type(line.semantic_type) is not str
                    or line.visual_row_id is not None
                    and type(line.visual_row_id) is not int
                    or type(line.source_index) is not int
                    or any(
                        type(value) is not bool
                        for value in (line.style_scale_repaired, line.restored_inline_cluster, line.split_from_row)
                    )
                    or line.font_signature is not None
                    and (
                        type(line.font_signature) is not tuple
                        or len(line.font_signature) != 2
                        or type(line.font_signature[0]) is not str
                        or type(line.font_signature[1]) is not int
                    )
                ):
                    self.plain = False
                    break
                self.memberships.setdefault(id(line), set()).add(id(lane))
                self.heights[id(line), id(box)] = _line_effective_height(line, box)
            if not self.plain:
                break
        if not self.plain:
            self.heights.clear()
            self.memberships.clear()

    def profile(self, lane):
        """保留首次统计导致的排序副作用，不把排序前结果缓存到排序后状态。"""
        if not self.plain:
            return _infer_lane_body_profile(lane)
        cached = self.profiles.get(id(lane))
        if cached is not None:
            return cached
        order = tuple(id(row) for row in lane.lines)
        result = _infer_lane_body_profile(lane)
        if order == tuple(id(row) for row in lane.lines):
            self.profiles[id(lane)] = result
        return result

    def height(self, line, box):
        """只复用阶段内已准备的原行框尺度，框或行身份不同则现场计算。"""
        value = self.heights.get((id(line), id(box)))
        return value if value is not None else _line_effective_height(line, box)

    def invalidate(self, line):
        """语义变化后清除所有包含该行的栏统计，保留未变动的只读几何。"""
        for lane_id in self.memberships.get(id(line), ()):
            self.profiles.pop(lane_id, None)


def _stage_profile_context(lanes, line_geometry, container_bboxes=(), document_body_profile=None, scalars=()):
    """特殊容器或文档字体可能通过回调修改普通行，整阶段恢复原实时统计。"""
    context = _LaneProfileContext(lanes, line_geometry)
    boxes_plain = type(container_bboxes) in (tuple, list) and all(
        type(box) in (tuple, list) and len(box) == 4 and all(_plain_profile_number(value) for value in box)
        for box in container_bboxes
    )
    profile_plain = document_body_profile is None or (
        type(document_body_profile) is _DocumentBodyProfile
        and _plain_profile_number(document_body_profile.body_height)
        and (document_body_profile.body_weight is None or _plain_profile_number(document_body_profile.body_weight))
        and type(document_body_profile.regular_fonts) is frozenset
        and type(document_body_profile.has_style_scale_repairs) is bool
        and all(
            type(value) is tuple and len(value) == 2 and type(value[0]) is str and type(value[1]) is int
            for value in document_body_profile.regular_fonts
        )
    )
    if not boxes_plain or not profile_plain or any(value is not None and not _plain_profile_number(value) for value in scalars):
        context.plain = False
        context.profiles.clear()
        context.heights.clear()
        context.memberships.clear()
    return context


def _infer_document_body_profile(
    prepared_pages: list[_PreparedPage],
    *,
    use_canonical_scale: bool = False,
) -> _DocumentBodyProfile | None:
    """按跨页覆盖和累计行宽推断全文正文行高及常规字体集合。"""

    samples: list[
        tuple[
            float,
            int,
            float,
            tuple[str, int] | None,
            float | None,
        ]
    ] = []
    sparse_document = sum(line.semantic_type is None for prepared in prepared_pages for line in prepared.remaining_lines) <= 6
    for page_index, prepared in enumerate(prepared_pages):
        # 表格几乎覆盖全页时，完整认领不应删除正文样式的统计证据；缓存不含原生字符字典。
        profile_lines = [*prepared.remaining_lines, *(prepared.table_body_profile_lines if sparse_document else [])]
        for line in profile_lines:
            if line.semantic_type is not None:
                continue
            local_bbox = _rotate_bbox_to_upright(
                line.bbox,
                prepared.page_size,
                line.angle,
            )
            local_page_width = prepared.page_size[1] if line.angle in {90, 270} else prepared.page_size[0]
            height = (
                _line_canonical_style_scale(line, local_bbox)
                if use_canonical_scale
                else _line_effective_height(line, local_bbox)
            )
            normalized_width = max(0.0, local_bbox[2] - local_bbox[0]) / max(
                0.1,
                local_page_width,
            )
            if height <= 0 or normalized_width <= 0:
                continue
            samples.append(
                (
                    height,
                    page_index,
                    normalized_width,
                    line.font_signature if line.font_coverage >= 0.75 else None,
                    line.dominant_font_weight,
                )
            )
    if not samples:
        return None

    from .._ordered_statistics import ordered_clusters

    groups = ordered_clusters([sample[0] for sample in samples], 0.0, relative=0.1)
    height_clusters = [[samples[index][:3] for index in group] for group in groups]
    cross_page_clusters = [cluster for cluster in height_clusters if len({item[1] for item in cluster}) >= 2]
    eligible_clusters = cross_page_clusters or height_clusters
    # 单页演示稿中两条跨页宽大标题会压过多栏正文；至少三条独立正文行才竞争画像。
    if len(prepared_pages) == 1 and len(samples) >= 6:
        repeated_clusters = [cluster for cluster in eligible_clusters if len(cluster) >= 3]
        if repeated_clusters:
            eligible_clusters = repeated_clusters
    body_cluster = max(
        eligible_clusters,
        key=lambda cluster: (
            sum(item[2] for item in cluster),
            len({item[1] for item in cluster}),
            len(cluster),
        ),
    )
    body_height = statistics.median(item[0] for item in body_cluster)

    body_weights = [
        weight
        for height, _page_index, _width, _font, weight in samples
        if weight is not None and 0.9 <= height / body_height <= 1.1
    ]
    body_weight = statistics.median(body_weights) if body_weights else None

    font_pages: dict[tuple[str, int], set[int]] = {}
    font_widths: dict[tuple[str, int], float] = {}
    font_weights: dict[tuple[str, int], list[float]] = {}
    for height, page_index, width, font, weight in samples:
        if font is None or not 0.9 <= height / body_height <= 1.1:
            continue
        # 常规字体支持必须来自正文高度带；跨页重复的大标题不能反向污染正文画像。
        font_pages.setdefault(font, set()).add(page_index)
        font_widths[font] = font_widths.get(font, 0.0) + width
        if weight is not None:
            font_weights.setdefault(font, []).append(weight)

    regular_fonts = frozenset(
        font
        for font, pages in font_pages.items()
        if len(pages) >= 3
        and font_widths.get(font, 0.0) >= 2.0
        and font_widths.get(font, 0.0) >= 0.75 * len(pages)
        and _document_font_is_regular(
            font,
            font_weights.get(font, []),
            body_weight,
        )
    )
    return _DocumentBodyProfile(
        body_height=max(0.1, body_height),
        body_weight=body_weight,
        regular_fonts=regular_fonts,
        has_style_scale_repairs=any(
            line.style_scale_repaired for prepared in prepared_pages for line in prepared.remaining_lines
        ),
    )


def _document_font_is_regular(
    font: tuple[str, int],
    weights: list[float],
    body_weight: float | None,
) -> bool:
    """用字体样式位和全文正文基准过滤斜体、粗体等强调字体。"""

    if font[1] & (PDF_FONT_ITALIC_FLAG | PDF_FONT_FORCE_BOLD_FLAG):
        return False
    if not weights or body_weight is None:
        return True
    median_weight = statistics.median(weights)
    return median_weight < max(body_weight + 100.0, 1.15 * body_weight)


def _infer_lane_body_profile(lane: _TextLane) -> _LaneBodyProfile:
    """从栏带的长行主体估计正文行高、主字体、字重、常规行距和样式占比。"""

    # 快照只覆盖本次分析，后续行成员或几何变化时由调用方重新进入。
    available = [(line, bbox, _line_effective_height(line, bbox)) for line, bbox in lane.lines if line.semantic_type is None]
    if not available:
        return _LaneBodyProfile(1.0, None, None, 0.35, {})
    lane_width = max(0.1, lane.right - lane.left)
    long_lines = [item for item in available if item[1][2] - item[1][0] >= 0.45 * lane_width]
    body_rows = long_lines or available
    body_line_ids = {id(line) for line, _bbox, _height in body_rows}
    body_height = statistics.median(height for _line, _bbox, height in body_rows)

    font_support: dict[tuple[str, int], float] = {}
    style_support: dict[tuple[str, int], float] = {}
    total_style_width = 0.0
    for line, bbox, height in available:
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        line_width = max(0.1, bbox[2] - bbox[0])
        style_support[line.font_signature] = style_support.get(line.font_signature, 0.0) + line_width
        total_style_width += line_width
        if id(line) in body_line_ids and 0.75 <= height / body_height <= 1.35:
            font_support[line.font_signature] = font_support.get(line.font_signature, 0.0) + line_width
    body_font = max(font_support, key=font_support.get) if font_support else None
    if total_style_width > 0:
        style_support = {signature: width / total_style_width for signature, width in style_support.items()}

    body_weights = [
        line.dominant_font_weight
        for line, bbox, height in body_rows
        if line.dominant_font_weight is not None
        and (body_font is None or line.font_signature == body_font)
        and 0.75 <= height / body_height <= 1.35
    ]
    regular_gap, _gap_mad = _estimate_lane_gap(lane)
    return _LaneBodyProfile(
        body_height=max(0.1, body_height),
        body_font=body_font,
        body_weight=statistics.median(body_weights) if body_weights else None,
        regular_gap=regular_gap,
        style_support=style_support,
        body_row_count=len(body_rows),
    )


def _line_uses_document_regular_font(
    line: _LineItem,
    document_body_profile: _DocumentBodyProfile | None,
) -> bool:
    """判断当前行是否使用跨页反复出现且未加粗的常规字体。"""

    return (
        document_body_profile is not None
        and line.font_signature is not None
        and line.font_coverage >= 0.5
        and any(
            _font_signatures_share_family(line.font_signature, regular_font)
            for regular_font in document_body_profile.regular_fonts
        )
    )


__all__ = [
    "_infer_document_body_profile",
    "_document_font_is_regular",
    "_infer_lane_body_profile",
    "_line_uses_document_regular_font",
]
