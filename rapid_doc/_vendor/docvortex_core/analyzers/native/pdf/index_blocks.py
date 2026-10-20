"""识别并构造 Flash 原生 PDF 的目录正文块。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics


from ....schema import BBox, BlockType

from .models import _LineItem
from .geometry import _bbox_center_x, _bbox_center_y, _bbox_overlap_in_smaller, _bbox_union_many, _rotate_bbox_to_upright
from .line_layout import _line_effective_height, _lines_tight_output_bbox


_INDEX_PAGE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9０-９])(?:[0-9０-９]+|[ivxlcdmIVXLCDM]+)"
    r"[\s.。．、,，;；:：)）\]】]*$"
)
_INDEX_MIN_ROWS = 5
_INDEX_MIN_PAGE_NUMBER_RATIO = 0.7


@dataclass(slots=True)
class _IndexRow:
    """保存目录候选视觉行的成员、局部几何和合并文本。"""

    members: list[_LineItem]
    local_member_bboxes: list[BBox]
    local_bbox: BBox
    content: str
    ends_in_page_number: bool


def _extract_index_blocks(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    container_bboxes: list[BBox],
    *,
    require_heading: bool = False,
    require_explicit_heading: bool = False,
) -> tuple[list[dict[str, object]], list[_LineItem]]:
    """用页码行尾和稳定版式识别目录；预判阶段可强制要求几何目录标题。"""

    claimed_line_ids: set[int] = set()
    blocks: list[dict[str, object]] = []
    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        rows = _build_index_rows(lines, page_size, angle)
        if len(rows) < _INDEX_MIN_ROWS:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_containers = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        eligible_rows = [
            row
            for row in rows
            if not any(_bbox_overlap_in_smaller(row.local_bbox, container_bbox) >= 0.35 for container_bbox in local_containers)
        ]
        # Confirm leader rows collectively, including short first/last sections.
        # This evidence does not depend on the language of the TOC heading.
        leader_rows = [row for row in eligible_rows if row.ends_in_page_number
                       and re.search(r"(?:[.．·]\s*){4,}", row.content)
                       and row.local_bbox[2] - row.local_bbox[0] >= .5 * local_page_width]
        if len(leader_rows) >= 5 and len(leader_rows) >= .6 * len(eligible_rows):
            right = statistics.median(row.local_bbox[2] for row in leader_rows)
            aligned = [row for row in leader_rows if abs(row.local_bbox[2] - right) <= .02 * local_page_width]
            if right >= .75 * local_page_width and len(aligned) >= .8 * len(leader_rows):
                for row in aligned:
                    for line in row.members:
                        line.semantic_type = BlockType.INDEX
                        claimed_line_ids.add(id(line))
                blocks.append(_index_rows_to_block(_join_index_label_continuations(aligned), angle, page_size))
                eligible_rows = [row for row in eligible_rows if not any(id(line) in claimed_line_ids for line in row.members)]
        for heading, candidate in _unpaged_numbered_index_bands(eligible_rows):
            for line in heading.members:
                line.semantic_type = "paragraph_title"
            for row in candidate:
                for line in row.members:
                    line.semantic_type = BlockType.INDEX
                    claimed_line_ids.add(id(line))
            blocks.append(_index_rows_to_block(candidate, angle, page_size))
        eligible_rows = [row for row in eligible_rows if not any(id(line) in claimed_line_ids for line in row.members)]
        for band in _split_index_bands(eligible_rows):
            candidate = _trim_index_band_edges(band)
            heading_row = _find_index_heading_row(
                eligible_rows,
                candidate,
                local_page_width,
            )
            explicit_heading = heading_row is not None and _is_index_heading(heading_row.content)
            if require_explicit_heading and not explicit_heading:
                continue
            if not _index_band_has_stable_layout(candidate, local_page_width, explicit_heading=explicit_heading):
                continue
            if require_heading and heading_row is None:
                continue
            candidate = _include_repeated_leading_index_section(eligible_rows, candidate)
            candidate = _join_index_label_continuations(candidate)
            if heading_row is not None:
                for line in heading_row.members:
                    line.semantic_type = "paragraph_title"
            for row in candidate:
                for line in row.members:
                    line.semantic_type = BlockType.INDEX
                    claimed_line_ids.add(id(line))
            blocks.append(_index_rows_to_block(candidate, angle, page_size))

    remaining_lines = [line for line in lines if id(line) not in claimed_line_ids]
    return blocks, remaining_lines


def _index_rows_to_block(rows: list[_IndexRow], angle: int, page_size: tuple[float, float]) -> dict[str, object]:
    """统一保存已确认目录的逻辑行顺序和原生成员范围，页码随完整条目输出。"""
    members = [line for row in rows for line in row.members]
    block: dict[str, object] = {
        "type": BlockType.INDEX,
        "bbox": _bbox_union_many([line.bbox for line in members]),
        "angle": angle,
        "content": "\n".join(row.content for row in rows),
    }
    tight = _lines_tight_output_bbox(members, page_size)
    if tight is not None:
        block["_tight_output_bbox"] = tight
    return block


def _index_row_em(row: _IndexRow) -> float:
    """使用原生字形尺度比较行距，避免多行条目的合并外框放大字号。"""
    return statistics.median(_index_line_height(line, box) for line, box in zip(row.members, row.local_member_bboxes))


def _index_line_height(line: _LineItem, box: BBox) -> float:
    """点线引导符占多数时，以字母数字的原生字框尺度验证目录行，避免点的墨迹高度污染。"""
    height = _line_effective_height(line, box)
    if re.search(r"(?:[.．·]\s*){3,}", line.text) is None:
        return height
    heights = []
    for char in line.chars:
        bounds = char.get("bbox")
        if str(char.get("char", "")).isalnum() and bounds is not None:
            local = _rotate_bbox_to_upright(bounds, (1.0, 1.0), line.angle)
            if local[3] > local[1]:
                heights.append(local[3] - local[1])
    return max(height, statistics.median(heights)) if heights else height


def _include_repeated_leading_index_section(all_rows: list[_IndexRow], candidate: list[_IndexRow]) -> list[_IndexRow]:
    """已确认页码目录内反复出现同式章节时，把紧邻首条的同式章节也归入目录。"""
    if not candidate:
        return candidate
    position = next((index for index, row in enumerate(all_rows) if row is candidate[0]), 0)
    if position == 0:
        return candidate
    previous = all_rows[position - 1]
    marker = re.compile(r"^(?:part|chapter|section|book|volume|vol\.)\s+(?:[IVXLCDM]+|\d+)[.：:\s]", re.I)
    if previous.ends_in_page_number or not marker.match(previous.content):
        return candidate
    font = previous.members[0].font_signature
    em = _index_row_em(previous)
    if font is None or em <= 0 or not 0.25 * em <= candidate[0].local_bbox[1] - previous.local_bbox[3] <= 3 * em:
        return candidate
    peers = [
        row
        for row in candidate
        if not row.ends_in_page_number
        and marker.match(row.content)
        and row.members[0].font_signature == font
        and 0.9 <= _index_row_em(row) / em <= 1.1
        and abs(row.local_bbox[0] - previous.local_bbox[0]) <= 0.5 * em
    ]
    return [previous, *candidate] if len(peers) >= 2 else candidate


def _join_index_label_continuations(rows: list[_IndexRow]) -> list[_IndexRow]:
    """只在已确认目录内合并紧邻的同式标签续行，把右侧页码移到完整标签之后。"""
    result: list[_IndexRow] = []
    for row in rows:
        if result:
            previous = result[-1]
            em = _index_row_em(previous)
            sidecar = previous.members[-1]
            label = previous.members[0]
            if (
                previous.ends_in_page_number
                and not row.ends_in_page_number
                and len(previous.members) >= 2
                and re.fullmatch(r"[\d０-９ivxlcdmIVXLCDM]+", sidecar.text.strip())
                and previous.local_member_bboxes[-1][0] > previous.local_bbox[0] + 8 * em
                and not re.match(r"^(?:part|chapter|section|\d+[.)])\b", row.content, re.I)
                and label.font_signature is not None
                and all(line.font_signature == label.font_signature for line in row.members)
                and 0.9 <= _index_row_em(row) / em <= 1.1
                and abs(row.local_bbox[0] - previous.local_bbox[0]) <= 0.5 * em
                and -0.25 * em <= row.local_bbox[1] - previous.local_bbox[3] <= 0.75 * em
                and row.local_bbox[2] < previous.local_member_bboxes[-1][0] - em
            ):
                members = [*previous.members[:-1], *row.members, sidecar]
                boxes = [*previous.local_member_bboxes[:-1], *row.local_member_bboxes, previous.local_member_bboxes[-1]]
                result[-1] = _IndexRow(
                    members, boxes, _bbox_union_many(boxes), " ".join(line.text.strip() for line in members), True
                )
                continue
        result.append(row)
    return result


def _unpaged_numbered_index_bands(rows: list[_IndexRow]) -> list[tuple[_IndexRow, list[_IndexRow]]]:
    """明确目录题名下连续五项以上的同式编号短条目可无页码，普通步骤正文不能触发。"""
    result = []
    numbered = re.compile(r"^(\d+)[.)]\s*(\S.+)$")
    for position, heading in enumerate(rows[:-1]):
        if not _is_index_heading(heading.content):
            continue
        first = rows[position + 1]
        match = numbered.match(first.content)
        em = _index_row_em(first)
        font = first.members[0].font_signature
        if (
            match is None
            or int(match[1]) != 1
            or em <= 0
            or font is None
            or not 0 <= first.local_bbox[1] - heading.local_bbox[3] <= 6 * em
            or not 1.1 <= _index_row_em(heading) / em <= 3
        ):
            continue
        candidate = []
        for row in rows[position + 1 :]:
            # 只有小于半个正文尺度的孤立纯数字可忽略，不能跨越其他正文或新的章节。
            if re.fullmatch(r"\d+", row.content) and _index_row_em(row) < 0.5 * em:
                continue
            match = numbered.match(row.content)
            if (
                match is None
                or int(match[1]) != len(candidate) + 1
                or row.ends_in_page_number
                or len(re.findall(r"[A-Za-z]+", match[2])) > 14
                or len(re.findall(r"[\u4e00-\u9fff]", match[2])) > 50
                or re.search(r"[.!?。！？]\s*$", match[2])
                or not 0.9 <= _index_row_em(row) / em <= 1.1
                or any(line.font_signature != font for line in row.members)
                or abs(row.local_bbox[0] - first.local_bbox[0]) > 0.5 * em
                or candidate
                and not 0.8 * em <= _bbox_center_y(row.local_bbox) - _bbox_center_y(candidate[-1].local_bbox) <= 3 * em
            ):
                break
            candidate.append(row)
        if len(candidate) >= _INDEX_MIN_ROWS:
            result.append((heading, candidate))
    return result


def _build_index_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
) -> list[_IndexRow]:
    """按 visual_row_id 复原当前方向的完整视觉行，保留左右分裂成员。"""

    row_groups: dict[tuple[str, int], list[tuple[_LineItem, BBox]]] = {}
    for line in lines:
        if line.angle != angle or line.semantic_type is not None:
            continue
        local = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        # 斜置水印可能被归入最近的正交方向，其外框远高于字形尺度，不能认领为目录成员。
        if local[3] - local[1] > 1.7 * _index_line_height(line, local):
            continue
        key = ("visual", line.visual_row_id) if line.visual_row_id is not None else ("source", line.source_index)
        row_groups.setdefault(key, []).append((line, local))

    rows: list[_IndexRow] = []
    for members in row_groups.values():
        ordered_pairs = sorted(
            members,
            key=lambda pair: (
                pair[1][0],
                pair[0].run_index,
                pair[0].source_index,
            ),
        )
        # 同一次目录分组内没有成员几何写入，复用当前输入的旋转框，避免排序与物化时重复变换。
        ordered = [line for line, _ in ordered_pairs]
        local_member_bboxes = [local for _, local in ordered_pairs]
        local_bbox = _bbox_union_many(local_member_bboxes)
        content = " ".join(part for line in ordered if (part := line.text.strip()))
        if not content:
            continue
        rows.append(
            _IndexRow(
                members=ordered,
                local_member_bboxes=local_member_bboxes,
                local_bbox=local_bbox,
                content=content,
                ends_in_page_number=_index_row_ends_in_page_number(content),
            )
        )
    rows.sort(
        key=lambda row: (
            row.local_bbox[1],
            row.local_bbox[0],
            min(line.source_index for line in row.members),
        )
    )
    combined = []
    for row in rows:
        if combined:
            previous = combined[-1]
            em = max(previous.local_bbox[3] - previous.local_bbox[1], row.local_bbox[3] - row.local_bbox[1])
            if abs(_bbox_center_y(row.local_bbox) - _bbox_center_y(previous.local_bbox)) <= 0.35 * em and (
                row.local_bbox[0] >= previous.local_bbox[2] or previous.local_bbox[0] >= row.local_bbox[2]
            ):
                ordered = sorted(
                    zip(previous.members + row.members, previous.local_member_bboxes + row.local_member_bboxes),
                    key=lambda pair: pair[1][0],
                )
                content = " ".join(line.text.strip() for line, _ in ordered)
                combined[-1] = _IndexRow(
                    [line for line, _ in ordered],
                    [bbox for _, bbox in ordered],
                    _bbox_union_many([previous.local_bbox, row.local_bbox]),
                    content,
                    _index_row_ends_in_page_number(content),
                )
                continue
        combined.append(row)
    return combined


def _index_row_ends_in_page_number(content: str) -> bool:
    """识别行尾的半角、全角阿拉伯页码或罗马页码。"""

    return _INDEX_PAGE_NUMBER_RE.search(content.rstrip()) is not None


def _is_index_heading(text: str) -> bool:
    """标准目录题名只提供语义先验，仍须重复条目和页码排列才能确认目录。"""
    return re.fullmatch(r"(?:table\s+of\s+)?contents|目\s*录", text.strip(), re.I) is not None


def _split_index_bands(rows: list[_IndexRow]) -> list[list[_IndexRow]]:
    """按显著纵向断层拆分候选带，同时容纳目录章节之间的加大行距。"""

    if not rows:
        return []
    median_height = statistics.median(max(_index_line_height(line, row.local_bbox) for line in row.members) for row in rows)
    maximum_pitch = 3.25 * max(0.1, median_height)
    bands: list[list[_IndexRow]] = [[rows[0]]]
    for row in rows[1:]:
        previous = bands[-1][-1]
        if _bbox_center_y(row.local_bbox) - _bbox_center_y(previous.local_bbox) > maximum_pitch:
            bands.append([row])
        else:
            bands[-1].append(row)
    return bands


def _trim_index_band_edges(rows: list[_IndexRow]) -> list[_IndexRow]:
    """移除候选带两端不带页码的标题或邻接正文，内部少量续行继续保留。"""

    start = 0
    end = len(rows)
    while start < end and not rows[start].ends_in_page_number:
        start += 1
    while end > start and (
        not rows[end - 1].ends_in_page_number or re.fullmatch(r"[\d\sivxlcdmIVXLCDM]+", rows[end - 1].content)
    ):
        end -= 1
    return rows[start:end]


def _index_band_has_stable_layout(
    rows: list[_IndexRow],
    local_page_width: float,
    *,
    explicit_heading: bool = False,
) -> bool:
    """联合页码比例、右边界、行宽、缩进和行距确认目录候选。"""

    if len(rows) < _INDEX_MIN_ROWS or local_page_width <= 0:
        return False
    if not explicit_heading:
        # 无目录题名时，重复金额槽是二维数据证据；条目编号加一个页码仍保留目录资格。
        numeric_rows = sum(
            len(re.findall(r"(?<!\S)[(（]?[+-]?\d[\d,]*(?:\.\d+)?[)）]?(?!\S)", row.content)) >= 3 for row in rows
        )
        if numeric_rows >= max(3, 0.5 * len(rows)):
            return False
    page_number_rows = [row for row in rows if row.ends_in_page_number]
    required_page_number_rows = max(
        4,
        math.ceil(_INDEX_MIN_PAGE_NUMBER_RATIO * len(rows)),
    )
    if len(page_number_rows) < required_page_number_rows:
        return False

    row_heights = [max(0.1, row.local_bbox[3] - row.local_bbox[1]) for row in rows]
    median_height = statistics.median(row_heights)
    median_right = statistics.median(row.local_bbox[2] for row in page_number_rows)
    right_tolerance = max(1.5 * median_height, 0.02 * local_page_width)
    aligned_right_ratio = sum(abs(row.local_bbox[2] - median_right) <= right_tolerance for row in page_number_rows) / len(
        page_number_rows
    )
    if median_right < (0.5 if explicit_heading else 0.75) * local_page_width or aligned_right_ratio < 0.7:
        return False

    wide_row_ratio = sum(
        row.local_bbox[2] - row.local_bbox[0] >= (0.4 if explicit_heading else 0.55) * local_page_width for row in rows
    ) / len(rows)
    right_sidecar_count = sum(_index_row_has_right_sidecar(row, local_page_width, median_height) for row in rows)
    if wide_row_ratio < 0.7 and right_sidecar_count < 2:
        return False

    left_body_ratio = sum(row.local_bbox[0] <= 0.3 * local_page_width for row in rows) / len(rows)
    if left_body_ratio < 0.7:
        return False

    pitches = [
        _bbox_center_y(current.local_bbox) - _bbox_center_y(previous.local_bbox) for previous, current in zip(rows, rows[1:])
    ]
    if not pitches or min(pitches) <= 0:
        return False
    median_pitch = statistics.median(pitches)
    regular_pitch_ratio = sum(0.55 * median_pitch <= pitch <= 1.75 * median_pitch for pitch in pitches) / len(pitches)
    return regular_pitch_ratio >= 0.7


def _find_index_heading_row(
    all_rows: list[_IndexRow],
    candidate: list[_IndexRow],
    local_page_width: float,
) -> _IndexRow | None:
    """用候选带上方的居中、短行和垂直邻接关系保留目录标题。"""

    if not candidate:
        return None
    first_position = next(
        (index for index, row in enumerate(all_rows) if row is candidate[0]),
        None,
    )
    if first_position is None or first_position == 0:
        return None
    heading = all_rows[first_position - 1]
    if heading.ends_in_page_number:
        return None
    candidate_heights = [max(0.1, row.local_bbox[3] - row.local_bbox[1]) for row in candidate]
    median_height = statistics.median(candidate_heights)
    heading_width = heading.local_bbox[2] - heading.local_bbox[0]
    centered = abs(_bbox_center_x(heading.local_bbox) - 0.5 * local_page_width) <= 0.15 * local_page_width
    vertical_pitch = _bbox_center_y(candidate[0].local_bbox) - _bbox_center_y(heading.local_bbox)
    if (
        heading_width > 0.4 * local_page_width
        or not centered
        and not _is_index_heading(heading.content)
        or not 1.25 * median_height <= vertical_pitch <= (12.0 if _is_index_heading(heading.content) else 5.0) * median_height
    ):
        return None
    return heading


def _index_row_has_right_sidecar(
    row: _IndexRow,
    local_page_width: float,
    median_height: float,
) -> bool:
    """检查视觉行末尾是否存在靠近页面右侧的窄页码片段。"""

    if len(row.members) < 2:
        return False
    last_bbox = row.local_member_bboxes[-1]
    return last_bbox[0] >= 0.75 * local_page_width and last_bbox[2] - last_bbox[0] <= max(
        0.12 * local_page_width, 4.0 * median_height
    )
