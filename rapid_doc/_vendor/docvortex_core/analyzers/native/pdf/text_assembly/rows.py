"""依据栏带、缩进和排版重置寻找正文行分组边界。"""

from __future__ import annotations

import re
import statistics
from typing import Sequence

from .....foundation._text import is_hyphen_at_line_end
from .....schema import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _rotate_bbox_to_upright
from ..line_layout import (
    _connection_crosses_table,
    _effective_body_text_row_gap,
    _effective_text_row_gap,
    _horizontal_rule_separates_rows,
    _line_effective_height,
    _line_tight_output_bbox,
    _title_fonts_compatible,
)
from ..models import _LineItem, _LocalAxisLine, _TextLane
from ..inline.types import PDF_FONT_ITALIC_FLAG
from .common import (
    _ABSTRACT_METADATA_RE,
    _BULLET_ITEM_RE,
    _EMAIL_METADATA_RE,
    _FRONT_MATTER_FIELD_RE,
    _LABELLED_METADATA_RE,
    _LIST_ITEM_RE,
    _REFERENCE_ENTRY_RE,
    _URL_LINE_RE,
)


def _top_marginal_text_break_sources(lane: _TextLane, page_height: float) -> set[int]:
    """顶边独立编号或较小文字与稳定正文之间保留净空边界，不依赖跨页信息强制分类页眉。"""
    rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
    output: set[int] = set()
    for index in range(1, len(rows) - 2):
        band, body = rows[:index], rows[index : index + 3]
        if any(bounds[3] > 0.08 * page_height for _, bounds in band):
            break
        if any(line.semantic_type is not None or line.caption_start for line, _ in [*band, *body]):
            continue
        height = statistics.median(_line_effective_height(line, bounds) for line, bounds in body)
        if height <= 0 or body[0][1][1] > 0.16 * page_height:
            continue
        if any(not 0.85 * height <= _line_effective_height(line, bounds) <= 1.15 * height for line, bounds in body):
            continue
        left = body[0][1][0]
        if any(abs(bounds[0] - left) > height for _, bounds in body):
            continue
        if any(len(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", line.text)) < 4 for line, _ in body[:2]):
            continue
        if any(not -0.4 * height <= after[1][1] - before[1][3] <= height for before, after in zip(body, body[1:])):
            continue
        if body[0][1][1] - max(bounds[3] for _, bounds in band) < 0.5 * height:
            continue
        if not all(
            re.fullmatch(r"\d{1,4}", line.text.strip())
            and bounds[2] - bounds[0] <= 4 * height
            or _line_effective_height(line, bounds) <= 0.95 * height
            for line, bounds in band
        ):
            continue
        output.add(body[0][0].source_index)
        break
    return output


def _repeated_bullet_break_sources(line_geometry: list[tuple[_LineItem, BBox]]) -> set[int]:
    """同左缘重复圆点形成独立项目起点；分离的圆点同时保护其唯一同排正文，条目长短不影响边界。"""
    candidates: list[tuple[_LineItem, BBox, _LineItem | None]] = []
    for marker, bounds in line_geometry:
        if marker.semantic_type is not None or marker.caption_start or marker.paragraph_group is not None:
            continue
        if _BULLET_ITEM_RE.match(marker.text) is None:
            continue
        body = None
        height = _line_effective_height(marker, bounds)
        if marker.text.strip() in {"•", "●", "▪"}:
            if bounds[2] - bounds[0] > height:
                continue
            hosts = [
                line
                for line, bbox in line_geometry
                if line is not marker
                and line.semantic_type is None
                and not line.caption_start
                and line.paragraph_group is None
                and not line.formula_candidate_only
                and 0 <= bbox[0] - bounds[2] <= 3.5 * height
                and _bbox_axis_overlap_ratio(bounds, bbox, axis="y") >= 0.5
                and 0.7 * height <= _line_effective_height(line, bbox) <= 1.5 * height
                and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", line.text)) >= 2
            ]
            if len(hosts) != 1:
                continue
            body = hosts[0]
        elif len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", marker.text)) < 2:
            continue
        candidates.append((marker, bounds, body))
    output: set[int] = set()
    for marker, bounds, body in candidates:
        height = _line_effective_height(marker, bounds)
        for other, other_bounds, _ in candidates:
            if other is marker:
                continue
            other_height = _line_effective_height(other, other_bounds)
            if (
                max(height, other_height) > 1.35 * min(height, other_height)
                or abs(bounds[0] - other_bounds[0]) > 0.5 * max(height, other_height)
                or not 0.7 * max(height, other_height)
                <= abs(_bbox_center_y(bounds) - _bbox_center_y(other_bounds))
                <= 8 * max(height, other_height)
            ):
                continue
            output.add(marker.source_index)
            if body is not None:
                output.add(body.source_index)
            break
        if marker.source_index not in output:
            # 左栏首条可接右栏重复项目；必须同时有左栏冒号引导句与右栏至少两个同式圆点。
            peers = [
                (other, box)
                for other, box, _ in candidates
                if other is not marker
                and other.font_signature == marker.font_signature
                and 0.85 <= _line_effective_height(other, box) / height <= 1.15
                and box[0] >= bounds[2] + height
            ]
            repeated_other_column = any(
                abs(first_box[0] - second_box[0]) <= 0.5 * height
                and 0.7 * height <= abs(_bbox_center_y(first_box) - _bbox_center_y(second_box)) <= 8 * height
                for first, first_box in peers
                for second, second_box in peers
                if first is not second
            )
            colon_intro = any(
                line.semantic_type is None
                and line.paragraph_group is None
                and line.text.rstrip().endswith((":", "："))
                and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", line.text)) >= 2
                and 0 <= bounds[1] - box[3] <= 3 * height
                and 0 <= bounds[0] - box[0] <= 2 * height
                for line, box in line_geometry
            )
            if repeated_other_column and colon_intro:
                output.add(marker.source_index)
                if body is not None:
                    output.add(body.source_index)
    return output


def _italic_quote_to_body_break_sources(lane: _TextLane) -> set[int]:
    """两行缩进斜体引文收句后，向左重置的常规正文形成永久段界，防止标题回退后反向粘连。"""
    rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
    output: set[int] = set()
    for index in range(2, len(rows) - 1):
        before, previous, current, following = rows[index - 2 : index + 2]
        quote_lines = (before, previous)
        body_lines = (current, following)
        all_lines = (*quote_lines, *body_lines)
        if any(
            line.semantic_type is not None
            or line.caption_start
            or line.paragraph_group is not None
            or line.font_signature is None
            or line.formula_candidate_only
            for line, _ in all_lines
        ):
            continue
        if not all(line.font_signature[1] & PDF_FONT_ITALIC_FLAG for line, _ in quote_lines):
            continue
        if any(line.font_signature[1] & PDF_FONT_ITALIC_FLAG for line, _ in body_lines):
            continue
        height = statistics.median(_line_effective_height(line, bounds) for line, bounds in all_lines)
        if height <= 0 or any(
            not 0.85 * height <= _line_effective_height(line, bounds) <= 1.15 * height for line, bounds in all_lines
        ):
            continue
        if (
            not _title_fonts_compatible(before[0], previous[0])
            or not _title_fonts_compatible(current[0], following[0])
            or not (previous[0].paragraph_terminal or re.search(r"[.!?。！？][\])’\"']*$", previous[0].text.rstrip()))
            or abs(before[1][0] - previous[1][0]) > 0.25 * height
            or not 0.65 * height <= previous[1][0] - current[1][0] <= 3 * height
            or current[1][2] - current[1][0] < 0.75 * (lane.right - lane.left)
            or len(re.findall(r"\b[A-Za-z]{2,}\b", current[0].text)) < 6
            or not all(-0.1 * height <= _effective_text_row_gap(a, b) <= 0.8 * height for a, b in zip(all_lines, all_lines[1:]))
        ):
            continue
        output.add(current[0].source_index)
    return output


def _caption_to_body_break_sources(lane: _TextLane) -> set[int]:
    """图注后字号增大且留白重置的正文形成永久边界，避免相同字体家族导致正文被吞入图注。"""
    output: set[int] = set()
    active = False
    caption_left = 0.0
    caption_center = 0.0
    rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
    for row_index, (previous, current) in enumerate(zip(rows, rows[1:])):
        line, bounds = previous
        following, next_bounds = current
        body_continuation = (
            bounds[2] - bounds[0] >= 0.75 * (lane.right - lane.left)
            and not line.caption_start
            and not line.text.rstrip().endswith((".", ":", "。", "：", "!", "?", "！", "？"))
            and 0.85 <= _line_effective_height(line, bounds) / max(0.1, _line_effective_height(following, next_bounds)) <= 1.15
        )
        if (
            following.caption_start
            and not body_continuation
            and _effective_text_row_gap(previous, current)
            >= 0.5 * max(_line_effective_height(line, bounds), _line_effective_height(following, next_bounds))
        ):
            # 显式图表编号启动新的注释，不能续接前一图的Source或邻近正文。
            # 正文续行即使恰从图号开头，也不能仅凭标记制造段界。
            output.add(following.source_index)
        if line.caption_start:
            active = True
            caption_left = bounds[0]
            caption_center = (bounds[0] + bounds[2]) / 2
        if line.semantic_type is not None or following.semantic_type is not None or following.caption_start:
            active = False
            continue
        height = _line_effective_height(line, bounds)
        next_height = _line_effective_height(following, next_bounds)
        if (
            active
            and abs((next_bounds[0] + next_bounds[2]) / 2 - caption_center) <= 0.5 * height
            and next_bounds[1] - bounds[3] <= 0.75 * height
            and 0.85 <= next_height / height <= 1.15
        ):
            # 居中图题各行宽度不同，左缘稍向外伸不应提前丢失图题状态。
            caption_left = min(caption_left, next_bounds[0])
        # 居中署名的短括号尾行结束后，首行缩进加连续满栏行可确认正文重置；单个长图题续行不够。
        width = lane.right - lane.left
        body_rows = rows[row_index + 2 : row_index + 4]
        centered_credit_to_body = (
            active
            and line.text.rstrip().endswith((")", "）"))
            and bounds[2] - bounds[0] <= 0.6 * width
            and abs((bounds[0] + bounds[2]) / 2 - (lane.left + lane.right) / 2) <= 0.08 * width
            and 0.4 * height <= next_bounds[0] - lane.left <= 2 * height
            and next_bounds[2] - next_bounds[0] >= 0.85 * width
            and abs(next_bounds[2] - lane.right) <= 0.3 * height
            and 0.85 <= next_height / height <= 1.15
            and len(body_rows) == 2
            and all(
                abs(body_bbox[0] - lane.left) <= 0.3 * height
                and abs(body_bbox[2] - lane.right) <= 0.3 * height
                and 0.85 <= _line_effective_height(body_line, body_bbox) / height <= 1.15
                for body_line, body_bbox in body_rows
            )
        )
        if centered_credit_to_body:
            output.add(following.source_index)
            active = False
            continue
        if next_bounds[0] < caption_left - height or next_bounds[1] - bounds[3] > 3 * height:
            active = False
            continue
        if active and next_height >= 1.15 * height and next_bounds[1] - bounds[3] >= 0.5 * height:
            output.add(following.source_index)
            active = False
    return output


def _local_tight_output_line_bboxes(
    lines: Sequence[_LineItem],
    page_size: tuple[float, float],
    angle: int,
) -> tuple[list[BBox], bool]:
    """返回与原行顺序一致的 tight+1pt 局部框及是否存在可靠候选。"""

    output = []
    changed = False
    for line in lines:
        candidate = _line_tight_output_bbox(line, page_size)
        output.append(
            _rotate_bbox_to_upright(
                candidate or line.bbox,
                page_size,
                angle,
            )
        )
        changed = changed or candidate is not None
    return output, changed


def _starts_structural_reference_entry(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> bool:
    """仅在编号行相对续行明显左突时确认新的参考文献条目。"""

    if _REFERENCE_ENTRY_RE.match(current[0].text.strip()) is None:
        return False
    previous_height = _line_effective_height(*previous)
    current_height = _line_effective_height(*current)
    pair_height = max(previous_height, current_height)
    return (
        current[1][0] <= previous[1][0] - max(5.0, 0.6 * min(previous_height, current_height))
        and -0.75 * pair_height <= _effective_text_row_gap(previous, current) <= 1.5 * pair_height
    )


def _build_hanging_indent_group_map(
    lane: _TextLane,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> dict[int, int]:
    """仅按重复的左突首行和稳定续行缩进识别悬挂缩进条目。"""

    if len(lane.lines) < 4:
        return {}
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if len(rows) < 4:
        return {}
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in rows)
    start_tolerance = max(5.0, 0.65 * median_height)
    minimum_indent = max(7.0, 0.8 * median_height)
    continuation_tolerance = max(4.0, 0.55 * median_height)

    def rows_are_adjacent(
        previous: tuple[_LineItem, BBox],
        current: tuple[_LineItem, BBox],
    ) -> bool:
        """检查相邻行的净空和几何障碍是否允许组成同一缩进序列。"""

        effective_gap = _effective_text_row_gap(previous, current)
        top_pitch = current[1][1] - previous[1][1]
        robust_pitch_fallback = 0.5 * median_height <= top_pitch <= 1.8 * median_height
        if not -0.6 * median_height <= effective_gap <= 1.3 * median_height and not robust_pitch_fallback:
            return False
        if _connection_crosses_table(
            previous[0].bbox,
            current[0].bbox,
            table_bboxes,
        ):
            return False
        return not _horizontal_rule_separates_rows(
            previous[1],
            current[1],
            lane,
            axis_lines,
        )

    def consume_entry(
        start_index: int,
        start_left: float,
        expected_continuation_left: float | None,
        *,
        require_next_start: bool,
    ) -> tuple[int, float] | None:
        """消费一个左突首行及其续行，并返回下一条首行位置。"""

        lane_width = max(0.1, lane.right - lane.left)
        full_width_midparagraph_entry = (
            rows[start_index][1][2] - rows[start_index][1][0] >= 0.8 * lane_width
            and start_index + 1 < len(rows)
            and rows[start_index + 1][1][2] - rows[start_index + 1][1][0] <= 0.75 * lane_width
        )
        if (
            start_index > 0
            and rows_are_adjacent(
                rows[start_index - 1],
                rows[start_index],
            )
            and abs(rows[start_index - 1][1][0] - start_left) <= start_tolerance
            and not full_width_midparagraph_entry
        ):
            # 同左缘正文仍在连续时不能从段落中部启动悬挂条目序列。
            return None
        if (
            start_index > 0
            and is_hyphen_at_line_end(rows[start_index - 1][0].text)
            and rows_are_adjacent(rows[start_index - 1], rows[start_index])
        ):
            # 排版断词后的下一物理行属于前文，不能被缩进几何误当成新条目首行。
            return None
        continuation_index = start_index + 1
        if continuation_index >= len(rows):
            return None
        first_continuation = rows[continuation_index]
        if not rows_are_adjacent(rows[start_index], first_continuation):
            return None
        continuation_left = first_continuation[1][0]
        if continuation_left < start_left + minimum_indent:
            return None
        if (
            expected_continuation_left is not None
            and abs(continuation_left - expected_continuation_left) > continuation_tolerance
        ):
            return None

        continuation_index += 1
        while continuation_index < len(rows):
            previous = rows[continuation_index - 1]
            current = rows[continuation_index]
            current_left = current[1][0]
            if not rows_are_adjacent(previous, current):
                break
            if current_left < start_left + minimum_indent:
                break
            if abs(current_left - continuation_left) > continuation_tolerance:
                break
            continuation_index += 1

        if not require_next_start:
            return continuation_index, continuation_left
        if continuation_index >= len(rows):
            return None
        if not rows_are_adjacent(rows[continuation_index - 1], rows[continuation_index]):
            return None
        if abs(rows[continuation_index][1][0] - start_left) > start_tolerance:
            return None
        return continuation_index, continuation_left

    group_map: dict[int, int] = {}
    group_index = 0
    row_index = 0
    while row_index < len(rows) - 3:
        start_left = rows[row_index][1][0]
        first_entry = consume_entry(
            row_index,
            start_left,
            None,
            require_next_start=True,
        )
        if first_entry is None:
            row_index += 1
            continue

        _next_start_index, continuation_left = first_entry
        start_indices = [row_index]
        current_start_index = row_index
        end_index: int | None = None
        while True:
            next_entry = consume_entry(
                current_start_index,
                start_left,
                continuation_left,
                require_next_start=True,
            )
            if next_entry is None:
                final_entry = consume_entry(
                    current_start_index,
                    start_left,
                    continuation_left,
                    require_next_start=False,
                )
                if final_entry is not None:
                    end_index = final_entry[0]
                break
            next_start_index, _continuation_left = next_entry
            prospective_entry = consume_entry(
                next_start_index,
                start_left,
                continuation_left,
                require_next_start=False,
            )
            if prospective_entry is None:
                # 当前条目已经完整确认；后面的普通左对齐段落只作为终止边界，
                # 不能让它反向使此前所有悬挂缩进条目失效。
                end_index = next_start_index
                break
            start_indices.append(next_start_index)
            current_start_index = next_start_index
        if len(start_indices) < 2 or end_index is None:
            row_index += 1
            continue

        entry_ranges = [
            (start, end)
            for start, end in zip(
                start_indices,
                [*start_indices[1:], end_index],
                strict=True,
            )
        ]
        for start, end in entry_ranges:
            for line, _bbox in rows[start:end]:
                group_map[line.source_index] = group_index
            group_index += 1
        row_index = end_index

    return group_map


def _infer_local_text_lane_map(lane: _TextLane) -> dict[int, _TextLane]:
    """从连续同左缘正文推导局部栏宽，修正跨栏上文污染的全宽栏带。"""

    if lane.is_span or len(lane.lines) < 3:
        return {}
    rows = sorted(
        lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in rows)
    left_tolerance = max(3.0, 0.75 * median_height)
    height_ratio_limit = 1.25
    runs: list[list[tuple[_LineItem, BBox]]] = []
    current_run: list[tuple[_LineItem, BBox]] = []

    def submit_run() -> None:
        """提交当前连续正文行，语义行和明显左缘变化都会结束该局部区段。"""

        nonlocal current_run
        if current_run:
            runs.append(current_run)
            current_run = []

    for item in rows:
        line, bbox = item
        if line.semantic_type is not None:
            submit_run()
            continue
        if not current_run:
            current_run = [item]
            continue
        run_left = statistics.median(member[1][0] for member in current_run)
        run_heights = [_line_effective_height(member, member_bbox) for member, member_bbox in current_run]
        current_height = _line_effective_height(line, bbox)
        if (
            abs(bbox[0] - run_left) <= left_tolerance
            and max([*run_heights, current_height]) / max(0.1, min([*run_heights, current_height])) <= height_ratio_limit
        ):
            current_run.append(item)
        else:
            submit_run()
            current_run = [item]
    submit_run()

    global_width = max(0.1, lane.right - lane.left)
    local_by_source: dict[int, _TextLane] = {}
    for run in runs:
        if len(run) < 3:
            continue
        local_left = statistics.median(bbox[0] for _line, bbox in run)
        local_right = max(bbox[2] for _line, bbox in run)
        local_width = max(0.1, local_right - local_left)
        wide_support = sum(bbox[2] - bbox[0] >= 0.7 * local_width for _line, bbox in run)
        if global_width < 1.4 * local_width or wide_support < 3:
            continue
        local_lane = _TextLane(
            left=local_left,
            right=local_right,
            lines=run,
            is_span=False,
        )
        for line, _bbox in run:
            local_by_source[line.source_index] = local_lane
    return local_by_source


def _structured_text_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """用重复强调首行和前行右侧留白确认结构化正文的新段起点。"""

    rows = sorted(
        lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    break_sources: set[int] = set()
    regions: list[list[tuple[_LineItem, BBox]]] = []
    for row in rows:
        if row[0].semantic_type is not None:
            if regions and regions[-1]:
                regions.append([])
            continue
        if not regions:
            regions.append([])
        regions[-1].append(row)

    for region in regions:
        candidates: list[int] = []
        for index, (line, bbox) in enumerate(region):
            height = _line_effective_height(line, bbox)
            line_width = bbox[2] - bbox[0]
            if (
                line.leading_emphasis_width is not None
                and line.leading_emphasis_width <= 0.2 * lane_width
                and line_width >= 0.95 * lane_width
                and abs(bbox[0] - lane.left) <= 0.75 * height
            ):
                candidates.append(index)
        if len(candidates) < 3:
            continue
        for index in candidates:
            if index == 0:
                continue
            previous = region[index - 1]
            current = region[index]
            pair_height = max(
                _line_effective_height(*previous),
                _line_effective_height(*current),
            )
            previous_fill = (previous[1][2] - lane.left) / lane_width
            vertical_gap = _effective_text_row_gap(previous, current)
            if previous_fill <= 0.8 and -0.25 * pair_height <= vertical_gap <= regular_gap + max(
                0.75 * pair_height, 3.0 * gap_mad
            ):
                break_sources.add(current[0].source_index)
    return break_sources


def _prose_paragraph_break_sources(lane: _TextLane, regular_gap: float, gap_mad: float) -> set[int]:
    """用句末、可容纳下一首词的短尾及重复首行缩进确认自然段，保护公式和已有条目归组。"""
    rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
    width = max(0.1, lane.right - lane.left)
    body = [
        (line, bbox)
        for line, bbox in rows
        if line.semantic_type is None and line.paragraph_group is None and bbox[2] - bbox[0] >= 0.7 * width
    ]
    if len(body) < 3:
        return set()
    em = statistics.median(_line_effective_height(line, bbox) for line, bbox in body)
    indent_starts = [bbox[0] for _, bbox in body if 0.65 * em <= bbox[0] - lane.left <= 3 * em]
    output = set()
    inside_item = False
    caption_left = None
    for index, (previous, current) in enumerate(zip(rows, rows[1:])):
        first, fb = previous
        line, bbox = current
        if first.semantic_type is not None or bbox[1] - fb[3] > 3 * em:
            inside_item = False
            caption_left = None
        if first.caption_start:
            caption_left = fb[0]
        if caption_left is not None and abs(bbox[0] - caption_left) > em:
            caption_left = None
        if caption_left is not None and index + 2 < len(rows):
            next_line, next_bbox = rows[index + 2]
            # 完整小字号图注收句后，留白及两行较大正文共同结束图注保护，恢复后续自然段判定。
            first_height = _line_effective_height(first, fb)
            body_height = _line_effective_height(line, bbox)
            if (
                first.paragraph_terminal
                and first.semantic_type is None
                and line.semantic_type is None
                and next_line.semantic_type is None
                and not line.caption_start
                and not next_line.caption_start
                and 0.7 * first_height <= _effective_body_text_row_gap(previous, current) <= 3 * first_height
                and 1.05 * first_height <= body_height <= 1.25 * first_height
                and 0.95 * body_height <= _line_effective_height(next_line, next_bbox) <= 1.05 * body_height
                and bbox[2] - bbox[0] >= 0.7 * width
                and next_bbox[2] - next_bbox[0] >= 0.7 * width
                and _title_fonts_compatible(line, next_line)
            ):
                caption_left = None
        if re.match(r"^\s*(?:[•●▪]|\d+[.)](?:\s|$))", first.text) or _REFERENCE_ENTRY_RE.match(first.text):
            inside_item = True
        if (
            first.angle != 0
            or line.angle != 0
            or first.semantic_type is not None
            or line.semantic_type is not None
            or first.paragraph_group is not None
            or line.paragraph_group is not None
            or first.formula_candidate_only
            or line.formula_candidate_only
            or first.compact_formula_cluster
            or line.compact_formula_cluster
            or inside_item
            or caption_left is not None
            or not (first.paragraph_terminal or re.search(r"[.!?。！？][\])’\"']*$", first.text.rstrip()))
            or re.match(r"^[A-Z][A-Za-z]+\b", line.text.strip()) is None
            or bbox[2] - bbox[0] < 0.7 * width
        ):
            continue
        gap = _effective_body_text_row_gap(previous, current)
        first_ink, current_ink = first.ink_bbox or fb, line.ink_bbox or bbox
        if current_ink[1] < first_ink[3] - 0.15 * em:
            continue
        # 原生段尾证据包含引用上标之前的句点；明显空行必须成为永久段界，不能被块级续行重新合并。
        blank_paragraph = (
            max(1.15 * em, regular_gap + 0.85 * em + 3 * gap_mad) <= gap <= 3 * em
            and abs(bbox[0] - lane.left) <= 0.35 * em
            and abs(fb[0] - lane.left) <= 0.35 * em
            and _title_fonts_compatible(first, line)
            and max(_line_effective_height(*previous), _line_effective_height(*current))
            <= 1.2 * min(_line_effective_height(*previous), _line_effective_height(*current))
        )
        if blank_paragraph:
            output.add(line.source_index)
            continue
        if not -0.25 * em <= gap <= regular_gap + max(0.75 * em, 3 * gap_mad):
            continue
        following = rows[index + 2] if index + 2 < len(rows) else None
        repeated_indent = sum(abs(left - bbox[0]) <= 0.25 * em for left in indent_starts) >= 2
        returns_to_left = (
            following is not None
            and following[0].semantic_type is None
            and abs(following[1][0] - lane.left) <= 0.35 * em
            and following[1][2] - following[1][0] >= 0.65 * width
            and _title_fonts_compatible(line, following[0])
        )
        indented = 0.65 * em <= bbox[0] - lane.left <= 3 * em and (returns_to_left or repeated_indent)
        word = re.match(r"\S+", line.text.strip()).group()
        word_width = len(word) * (line.median_glyph_width or 0.5 * em)
        short_tail = (
            abs(bbox[0] - lane.left) <= 0.35 * em
            and abs(fb[0] - lane.left) <= 0.35 * em
            and fb[2] - fb[0] <= 0.8 * width
            and lane.right - fb[2] >= word_width + 2 * em
            and _title_fonts_compatible(first, line)
        )
        if indented or short_tail:
            output.add(line.source_index)
    return output


def _cjk_prose_break_sources(lane: _TextLane, regular_gap: float, gap_mad: float) -> set[int]:
    """中文句末后的额外净空证明段界，短冒号标签也保留边界；普通同距续行不拆分。"""
    rows = sorted(lane.lines, key=lambda item: (item[1][1], item[1][0]))
    width = max(0.1, lane.right - lane.left)
    output = set()
    for (previous, before), (current, bounds) in zip(rows, rows[1:]):
        if any(
            line.semantic_type is not None
            or line.paragraph_group is not None
            or line.caption_start
            or line.formula_candidate_only
            or line.compact_formula_cluster
            or line.angle != 0
            for line in (previous, current)
        ):
            continue
        em = max(_line_effective_height(previous, before), _line_effective_height(current, bounds))
        if (
            em <= 0
            or previous.font_signature is None
            or not _title_fonts_compatible(previous, current)
            or min(_line_effective_height(previous, before), _line_effective_height(current, bounds)) < 0.85 * em
            or abs(bounds[0] - before[0]) > 0.25 * em
            or not re.search(r"[。！？][”’）]*$", previous.text.rstrip())
            or len(re.findall(r"[\u3400-\u9fff]", previous.text)) < 4
            or len(re.findall(r"[\u3400-\u9fff]", current.text)) < 8
        ):
            continue
        labelled = re.match(r"^[\u3400-\u9fff]{1,8}[：:]", current.text.strip()) is not None
        if re.match(r"^[\u3400-\u9fff]", current.text.strip()) is None or bounds[2] - bounds[0] < 0.65 * width and not labelled:
            continue
        gap = _effective_body_text_row_gap((previous, before), (current, bounds))
        threshold = max((0.5 if labelled else 0.65) * em, regular_gap + (0.2 if labelled else 0.3) * em + 3 * gap_mad)
        if threshold <= gap <= 3 * em:
            output.add(current.source_index)
    return output


def _cjk_entry_break_sources(lane: _TextLane) -> set[int]:
    """同左缘连续顿号编号及带日期的重复书名号报告各自起段，不拆叙述中的编号或引用。"""
    rows = sorted(
        (
            item
            for item in lane.lines
            if item[0].semantic_type is None and item[0].paragraph_group is None and not item[0].caption_start
        ),
        key=lambda item: (item[1][1], item[1][0]),
    )
    numbered = [
        (index, line, bounds, re.match(r"^\s*(\d{1,2})、\s*[\u3400-\u9fff]", line.text))
        for index, (line, bounds) in enumerate(rows)
    ]
    numbered = [(index, line, bounds, int(match.group(1))) for index, line, bounds, match in numbered if match]
    output = set()
    for first, second, third in zip(numbered, numbered[1:], numbered[2:]):
        em = max(_line_effective_height(line, bounds) for _, line, bounds, _ in (first, second, third))
        if (
            second[3] == first[3] + 1
            and third[3] == second[3] + 1
            and first[1].font_signature is not None
            and all(
                _title_fonts_compatible(first[1], item[1])
                and abs(item[2][0] - first[2][0]) <= 0.3 * em
                and 0.85 * em <= _line_effective_height(item[1], item[2])
                for item in (second, third)
            )
            and all(
                1 <= after[0] - before[0] <= 8 and 0 < after[2][1] - before[2][3] <= 10 * em
                for before, after in ((first, second), (second, third))
            )
        ):
            output.update(item[1].source_index for item in (first, second, third))
    references = [
        (index, line, bounds) for index, (line, bounds) in enumerate(rows) if re.match(r"^《[\u3400-\u9fff]", line.text.strip())
    ]
    if len(references) >= 3:
        for first, second in zip(references, references[1:]):
            tail, tail_bounds = rows[second[0] - 1]
            em = max(_line_effective_height(first[1], first[2]), _line_effective_height(second[1], second[2]))
            if (
                1 <= second[0] - first[0] <= 4
                and first[1].font_signature is not None
                and _title_fonts_compatible(first[1], second[1])
                and abs(first[2][0] - second[2][0]) <= 0.3 * em
                and 0.85 * em <= _line_effective_height(second[1], second[2])
                and re.search(r"[—–-]+\s*\d{4}[-./]\d{1,2}[-./]\d{1,2}\s*$", tail.text)
                and -0.1 * em <= second[2][1] - tail_bounds[3] <= em
            ):
                output.update((first[1].source_index, second[1].source_index))
    return output


def _isolated_indented_paragraph_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """识别短终止尾行之后的缩进首行，并要求下一行回到稳定栏左缘。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    output: set[int] = set()
    terminal_re = re.compile(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$")
    for previous, current, following in zip(
        rows,
        rows[1:],
        rows[2:],
    ):
        previous_height = _line_effective_height(*previous)
        current_height = _line_effective_height(*current)
        following_height = _line_effective_height(*following)
        pair_height = max(
            previous_height,
            current_height,
            following_height,
        )
        current_indent = current[1][0] - lane.left
        if (
            previous[1][2] - previous[1][0] > 0.3 * lane_width
            or terminal_re.search(previous[0].text.rstrip()) is None
            or not max(5.0, 0.65 * pair_height) <= current_indent <= 3.0 * pair_height
            or current[1][2] - current[1][0] < 0.75 * lane_width
            or abs(following[1][0] - lane.left) > 0.75 * pair_height
            or following[1][2] - following[1][0] < 0.65 * lane_width
            or not _title_fonts_compatible(current[0], following[0])
        ):
            continue
        first_gap = _effective_body_text_row_gap(previous, current)
        second_gap = _effective_body_text_row_gap(current, following)
        gap_limit = regular_gap + max(
            0.75 * pair_height,
            3.0 * gap_mad,
        )
        if -0.25 * pair_height <= first_gap <= gap_limit and -0.25 * pair_height <= second_gap <= gap_limit:
            output.add(current[0].source_index)
    return output


def _centered_visual_reset_break_sources(
    lane: _TextLane,
    visual_bboxes: Sequence[BBox],
    local_page_height: float,
) -> set[int]:
    """识别视觉主体下方短居中行到更宽居中行的独立注释重启。"""

    if not visual_bboxes:
        return set()
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    output: set[int] = set()
    for previous, current in zip(rows, rows[1:]):
        previous_bbox = previous[1]
        current_bbox = current[1]
        previous_width = previous_bbox[2] - previous_bbox[0]
        current_width = current_bbox[2] - current_bbox[0]
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        if (
            previous_width > 0.7 * current_width
            or current_bbox[0] > previous_bbox[0] - 0.25 * pair_height
            or current_bbox[2] < previous_bbox[2] + 0.25 * pair_height
            or abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) > 0.1 * current_width
        ):
            continue
        vertical_gap = _effective_text_row_gap(previous, current)
        if not -0.25 * pair_height <= vertical_gap <= 0.75 * pair_height:
            continue
        if any(
            -0.25 * pair_height <= previous_bbox[1] - visual_bbox[3] <= max(2.0 * pair_height, 0.03 * local_page_height)
            and _bbox_axis_overlap_ratio(current_bbox, visual_bbox, axis="x") >= 0.8
            and abs(_bbox_center_x(current_bbox) - _bbox_center_x(visual_bbox))
            <= 0.12 * max(current_width, visual_bbox[2] - visual_bbox[0])
            for visual_bbox in visual_bboxes
        ):
            output.add(current[0].source_index)
    return output


def _leading_typography_reset_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """识别短尾之后以独立行首字体 run 开启的宽行结构段。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    repeated_emphasis = (
        sum(
            line.leading_emphasis_width is not None and 0.03 * lane_width <= line.leading_emphasis_width <= 0.65 * lane_width
            for line, _bbox in rows
        )
        >= 3
    )
    repeated_typographic_starts = (
        sum(
            line.leading_typography_width is not None and 0.02 * lane_width <= line.leading_typography_width <= 0.2 * lane_width
            for line, _bbox in rows
        )
        >= 3
    )
    output: set[int] = set()
    for row_index, (previous, current) in enumerate(zip(rows, rows[1:])):
        previous_width = previous[1][2] - previous[1][0]
        current_width = current[1][2] - current[1][0]
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        # 完整收句后的独立居中句有双侧缩进和明显净空；星号等外围装饰不改变句末证据。
        if (
            previous_width <= 0.65 * lane_width
            and 0.4 * lane_width <= current_width <= 0.85 * lane_width
            and abs(previous[1][0] - lane.left) <= 0.35 * pair_height
            and min(current[1][0] - lane.left, lane.right - current[1][2]) >= 2 * pair_height
            and abs(_bbox_center_x(current[1]) - 0.5 * (lane.left + lane.right)) <= 0.5 * pair_height
            and min(_line_effective_height(*previous), _line_effective_height(*current)) >= 0.85 * pair_height
            and re.search(r"[.!?。！？][\])’\"'*]*$", previous[0].text.rstrip())
            and re.search(r"[.!?。！？][\])’\"'*]*$", current[0].text.rstrip())
            and max(0.75 * pair_height, regular_gap + 0.5 * pair_height + 3 * gap_mad)
            <= _effective_body_text_row_gap(previous, current)
            <= 2 * pair_height
            and previous[0].paragraph_group is None
            and current[0].paragraph_group is None
            and not previous[0].caption_start
            and not current[0].caption_start
            and not previous[0].formula_candidate_only
            and not current[0].formula_candidate_only
            and not previous[0].compact_formula_cluster
            and not current[0].compact_formula_cluster
            and not current[0].inline_math_regions
        ):
            output.add(current[0].source_index)
            continue
        # 重复行首字体标签与明显空行共同确认结构段，无需段尾以句点结束；保留普通字体续行和数学屏障。
        if (
            repeated_typographic_starts
            and current[0].leading_typography_width is not None
            and 0.02 * lane_width <= current[0].leading_typography_width <= 0.2 * lane_width
            and current_width >= 0.75 * lane_width
            and abs(previous[1][0] - lane.left) <= 0.35 * pair_height
            and abs(current[1][0] - lane.left) <= 0.35 * pair_height
            and max(0.9 * pair_height, regular_gap + 0.55 * pair_height + 3 * gap_mad)
            <= _effective_body_text_row_gap(previous, current)
            <= 3 * pair_height
            and _title_fonts_compatible(previous[0], current[0])
            and min(_line_effective_height(*previous), _line_effective_height(*current)) >= 0.85 * pair_height
            and not previous[0].caption_start
            and not current[0].caption_start
            and previous[0].paragraph_group is None
            and current[0].paragraph_group is None
            and not previous[0].formula_candidate_only
            and not current[0].formula_candidate_only
            and not previous[0].compact_formula_cluster
            and not current[0].compact_formula_cluster
            and not current[0].inline_math_regions
        ):
            output.add(current[0].source_index)
            continue
        following = rows[row_index + 2] if row_index + 2 < len(rows) else None
        local_emphasis = (
            previous_width <= 0.45 * lane_width
            and following is not None
            and (current[0].dominant_font_weight or 0) >= 550
            and following[0].dominant_font_weight is not None
            and current[0].dominant_font_weight >= following[0].dominant_font_weight + 100
            and following[0].leading_emphasis_width is None
            and following[1][2] - following[1][0] >= 0.75 * lane_width
            and abs(following[1][0] - lane.left) <= 0.5 * pair_height
            and 0.85 * pair_height <= _line_effective_height(*following) <= 1.15 * pair_height
        )
        explicit_emphasis = (
            (repeated_emphasis or local_emphasis)
            and current[0].leading_emphasis_width is not None
            and current[0].leading_emphasis_width <= 0.65 * lane_width
            and re.search(r"[.!?。！？][\])’\"']*$", previous[0].text.rstrip()) is not None
        )
        if (
            (current[0].leading_typography_width is None and not explicit_emphasis)
            or (
                current[0].leading_typography_width is not None
                and current[0].leading_typography_width > 0.2 * lane_width
                and not explicit_emphasis
            )
            or (previous_width > 0.45 * lane_width and not explicit_emphasis)
            or current_width < 0.75 * lane_width
            or abs(previous[1][0] - lane.left) > 0.75 * pair_height
            or abs(current[1][0] - lane.left) > 0.75 * pair_height
            or current[0].formula_candidate_only
            or current[0].compact_formula_cluster
            or current[0].inline_math_regions
        ):
            continue
        vertical_gap = _effective_body_text_row_gap(previous, current)
        if (
            -0.25 * pair_height
            <= vertical_gap
            <= regular_gap
            + max(
                0.75 * pair_height,
                3.0 * gap_mad,
            )
        ):
            output.add(current[0].source_index)
    return output


def _formula_style_text_row_break_sources(
    lane: _TextLane,
) -> set[int]:
    """按相邻显示行几何拆分被公式检测回退为正文的独立文本行。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    matching_edges: set[int] = set()
    for index, (previous, current) in enumerate(zip(rows, rows[1:])):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if not (previous_line.paragraph_formula_context and current_line.paragraph_formula_context):
            continue
        previous_height = _line_effective_height(*previous)
        current_height = _line_effective_height(*current)
        minimum_height = min(previous_height, current_height)
        maximum_height = max(previous_height, current_height)
        if minimum_height < 0.75 * maximum_height:
            continue
        previous_width = previous_bbox[2] - previous_bbox[0]
        current_width = current_bbox[2] - current_bbox[0]
        if min(previous_width, current_width) < 0.45 * lane_width or max(previous_width, current_width) > 0.95 * lane_width:
            continue
        lane_center = 0.5 * (lane.left + lane.right)
        if (
            abs(_bbox_center_x(previous_bbox) - lane_center) > 0.15 * lane_width
            or abs(_bbox_center_x(current_bbox) - lane_center) > 0.15 * lane_width
        ):
            continue
        vertical_overlap = max(
            0.0,
            min(previous_bbox[3], current_bbox[3]) - max(previous_bbox[1], current_bbox[1]),
        )
        top_pitch = current_bbox[1] - previous_bbox[1]
        pair_height = statistics.median((previous_height, current_height))
        if vertical_overlap <= 0.2 * minimum_height and 0.9 * pair_height <= top_pitch <= 2.0 * pair_height:
            matching_edges.add(index)

    output: set[int] = set()
    for index in matching_edges:
        output.add(rows[index][0].source_index)
        output.add(rows[index + 1][0].source_index)
        if index + 2 < len(rows):
            # 同时保护显示行组后的正文起点，避免上下文恢复阶段重新跨界合并。
            output.add(rows[index + 2][0].source_index)
    return output


def _front_matter_keyword_break_sources(
    lane: _TextLane,
    local_page_height: float,
    page_index: int | None,
) -> set[int]:
    """把首页关键词和文献元数据行固定为独立文本块起点。"""

    if page_index != 0:
        return set()
    return {
        line.source_index
        for line, bbox in lane.lines
        if line.semantic_type is None
        and bbox[1] <= 0.65 * local_page_height
        and _FRONT_MATTER_FIELD_RE.match(line.text) is not None
    }


def _component_starts_with_emphasized_row(
    lines: list[_LineItem],
) -> bool:
    """识别行内强调或首行字重显著高于后续正文的组件起点。"""

    if not lines:
        return False
    if lines[0].leading_emphasis_width is not None:
        return True
    first_weight = lines[0].dominant_font_weight
    following_weights = [line.dominant_font_weight for line in lines[1:] if line.dominant_font_weight is not None]
    if first_weight is None or not following_weights:
        return False
    body_weight = statistics.median(following_weights)
    return first_weight - body_weight >= 100.0 and first_weight >= 1.15 * max(1.0, body_weight)


def _explicit_text_break_sources(
    lane: _TextLane,
) -> set[int]:
    """用通用列表标记和 E-mail 元数据确认正文中的显式硬分段。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    output = {line.source_index for line, _bbox in rows if _ABSTRACT_METADATA_RE.match(line.text) is not None}
    numbered = [(line, bbox, re.match(r"^(\d{1,2})[.]\s+\D", line.text)) for line, bbox in rows]
    numbered = [(line, bbox, int(match.group(1))) for line, bbox, match in numbered if match is not None]
    for first, second in zip(numbered, numbered[1:]):
        em = max(_line_effective_height(first[0], first[1]), _line_effective_height(second[0], second[1]))
        if second[2] == first[2] + 1 and abs(first[1][0] - second[1][0]) <= em:
            output.update((first[0].source_index, second[0].source_index))
    lane_width = max(0.1, lane.right - lane.left)
    output.update(
        line.source_index
        for line, bbox in rows
        if _BULLET_ITEM_RE.match(line.text) is not None and bbox[2] - bbox[0] >= 0.8 * lane_width
    )
    for index, (line, bbox) in enumerate(rows):
        if _EMAIL_METADATA_RE.match(line.text) is None or index == 0:
            continue
        previous_line, previous_bbox = rows[index - 1]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(line, bbox),
        )
        if abs(bbox[0] - previous_bbox[0]) <= 0.75 * pair_height:
            output.add(line.source_index)
    for row_index, (previous, current) in enumerate(
        zip(rows, rows[1:]),
    ):
        previous_is_label = _LABELLED_METADATA_RE.match(
            previous[0].text,
        )
        current_is_label = _LABELLED_METADATA_RE.match(
            current[0].text,
        )
        label_pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        if (
            previous_is_label is not None
            and current_is_label is not None
            and _URL_LINE_RE.match(current[0].text) is None
            and len(previous_is_label.group("label")) >= 4
            and len(current_is_label.group("label")) >= 4
            and any("\u3400" <= char <= "\u9fff" for char in previous_is_label.group("label"))
            and any("\u3400" <= char <= "\u9fff" for char in current_is_label.group("label"))
            and previous_is_label.group("label").casefold() != current_is_label.group("label").casefold()
            and current[1][1] - previous[1][1] <= 2.0 * label_pair_height
            and previous[1][2] - previous[1][0] <= 0.75 * lane_width
            and current[1][2] - current[1][0] <= 0.75 * lane_width
        ):
            output.add(current[0].source_index)
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        next_row = rows[row_index + 2] if row_index + 2 < len(rows) else None
        indented_item_continuation = (
            current[1][0] - lane.left >= max(5.0, 0.65 * pair_height)
            and next_row is not None
            and next_row[1][0] - lane.left <= 0.5 * pair_height
            and 0.5 * pair_height <= next_row[1][1] - current[1][1] <= 2.25 * pair_height
        )
        if (
            _LIST_ITEM_RE.match(current[0].text) is not None
            and previous[0].text.rstrip().endswith((":", "："))
            and previous[1][2] - previous[1][0] <= 0.8 * lane_width
            and indented_item_continuation
        ):
            output.add(current[0].source_index)
    return output


__all__ = [
    "_local_tight_output_line_bboxes",
    "_starts_structural_reference_entry",
    "_build_hanging_indent_group_map",
    "_infer_local_text_lane_map",
    "_structured_text_break_sources",
    "_isolated_indented_paragraph_break_sources",
    "_centered_visual_reset_break_sources",
    "_leading_typography_reset_break_sources",
    "_formula_style_text_row_break_sources",
    "_front_matter_keyword_break_sources",
    "_component_starts_with_emphasized_row",
    "_explicit_text_break_sources",
]
