"""以独立留白和相对字重补充小字号标题，不放宽普通正文的标题评分。"""

from __future__ import annotations

from ..geometry import _bbox_center_x, _bbox_union_many
from ..layout_evidence import build_layout_evidence
from ..line_layout import _line_effective_height
from ..models import _PreparedPage
from .common import _line_inside_visual_container, _line_near_visual_container
from .structural import _following_stable_body_bounds


def _classify_small_emphasized_titles(page: _PreparedPage, page_index: int) -> None:
    """小字号粗体只有在独立标题带、真实正文转折和容器排除证据共同成立时提升。"""
    if page_index == 0:
        return
    containers = [block["bbox"] for block in page.fixed_blocks]
    layout = build_layout_evidence(page.remaining_lines, page.page_size, barriers=containers)
    processed = set()
    for lane in layout.lanes:
        left, right = layout.corridor((lane.left, 0, lane.right, 1))
        rows = sorted(
            [line for line in page.remaining_lines if line.angle == 0 and left <= _bbox_center_x(line.bbox) < right],
            key=lambda line: (line.bbox[1], line.bbox[0]),
        )
        for i, first in enumerate(rows):
            if (
                first.source_index in processed
                or first.semantic_type is not None
                or first.title_suppressed
                or first.paragraph_group is not None
                or first.dominant_font_weight is None
                or first.font_coverage < 0.9
                or first.caption_start
                or first.paragraph_terminal
            ):
                continue
            scale = _line_effective_height(first, first.bbox)
            j = i + 1
            while j < len(rows) and j - i < 3:
                line = rows[j]
                if (
                    line.semantic_type is not None
                    or line.title_suppressed
                    or line.paragraph_group is not None
                    or line.font_signature != first.font_signature
                    or line.paragraph_terminal
                    or abs(line.bbox[0] - first.bbox[0]) > 0.5 * scale
                    or not -0.1 * scale <= line.bbox[1] - rows[j - 1].bbox[3] <= 0.8 * scale
                ):
                    break
                j += 1
            group = rows[i:j]
            if j >= len(rows):
                continue
            body = rows[j]
            em = _line_effective_height(body, body.bbox)
            bounds = _following_stable_body_bounds(rows, j, em)
            if bounds is None or body.dominant_font_weight is None or body.font_coverage < 0.9:
                continue
            band = _bbox_union_many([line.bbox for line in group])
            preceding = [line for line in rows[:i] if line.semantic_type not in {"header", "footer", "page_number"}]
            gap_above = band[1] - preceding[-1].bbox[3] if preceding else None
            gap_below = body.bbox[1] - band[3]
            # 栏首没有上一段时，以全页正文区顶部约束位置，不能用页眉或无限留白支持提升。
            body_tops = [
                line.bbox[1]
                for line in page.remaining_lines
                if line.angle == 0
                and line.semantic_type is None
                and line.paragraph_group is None
                and line.font_signature == body.font_signature
                and 0.9 <= _line_effective_height(line, line.bbox) / em <= 1.1
                and line.bbox[2] - line.bbox[0] >= 0.6 * (bounds[2] - bounds[0])
            ]
            at_body_top = bool(body_tops) and -3.5 * em <= band[1] - min(body_tops) <= 0.5 * em
            if (
                not 0.65 <= scale / em <= 0.85
                or first.dominant_font_weight < max(body.dominant_font_weight + 100, 1.15 * body.dominant_font_weight)
                or band[2] > bounds[2] + 0.5 * em
                or not -0.5 * em <= band[0] - bounds[0] <= 1.5 * em
                or (not at_body_top if gap_above is None else not 0.65 * em <= gap_above <= 2.5 * em)
                or not 0.35 * em <= gap_below <= 2.5 * em
                or not 0.07 * page.page_size[1] < band[1] < band[3] < 0.93 * page.page_size[1]
                or _line_inside_visual_container(band, containers)
                or _line_near_visual_container(band, containers, em)
            ):
                continue
            for line in group:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                line.title_band_id = first.source_index
                processed.add(line.source_index)
