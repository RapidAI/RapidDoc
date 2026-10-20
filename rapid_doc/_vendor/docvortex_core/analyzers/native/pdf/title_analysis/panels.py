"""用重复面板、正文尺度转折和图像邻接确认局部标题。"""

from __future__ import annotations

import re

from ..geometry import _bbox_union_many
from ..models import _LineItem


def _panel_title_rows(seed: _LineItem, lines: list[_LineItem]) -> list[_LineItem]:
    """收集同左缘同样式的紧邻短续行，不把后续小字号简介并入标题。"""
    result = [seed]
    for line in sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0])):
        previous = result[-1]
        em = seed.effective_height
        if line is seed or line.angle != 0 or line.bbox[1] <= previous.bbox[1]:
            continue
        if (
            line.semantic_type in {None, "paragraph_title"}
            and line.font_signature == seed.font_signature
            and abs(line.bbox[0] - seed.bbox[0]) <= 0.3 * em
            and 0.9 * em <= line.effective_height <= 1.1 * em
            and -0.2 * em <= line.bbox[1] - previous.bbox[3] <= 0.8 * em
            and len(" ".join(item.text for item in [*result, line]).split()) <= 22
            and re.search(r"[.!?。！？;；]$", line.text.strip()) is None
        ):
            result.append(line)
            if len(result) == 3:
                break
    return result


def classify_panel_titles(lines: list[_LineItem], images: list[tuple], page_width: float) -> None:
    """重复并列面板须有独立正文或下方图像；重复粗体侧栏还须有图像邻接。"""
    seeds = [
        line
        for line in lines
        if line.angle == 0
        and line.semantic_type in {None, "paragraph_title"}
        and not line.title_suppressed
        and line.font_signature
        and 2 <= len(line.text.split()) <= 12
        and re.search(r"[.!?。！？;；]$", line.text.strip()) is None
    ]
    runs = []
    used = set()
    for seed in sorted(seeds, key=lambda line: (line.bbox[1], line.bbox[0])):
        if id(seed) in used:
            continue
        rows = _panel_title_rows(seed, lines)
        used.update(map(id, rows))
        runs.append((seed, rows, _bbox_union_many([line.bbox for line in rows])))
    for seed, rows, bounds in runs:
        em = seed.effective_height
        # 紧贴较大标题的小字号简介不是另一个标题，即使它也在并列图像上方。
        if any(
            other is not seed
            and other.angle == 0
            and abs(other.bbox[0] - bounds[0]) <= 0.3 * em
            and 1.1 * em <= other.effective_height <= 1.5 * em
            and -0.1 * em <= bounds[1] - other.bbox[3] <= 0.8 * em
            for other in lines
        ):
            continue
        peers = [
            (other, run, box)
            for other, run, box in runs
            if other.font_signature == seed.font_signature
            and 0.9 * em <= other.effective_height <= 1.1 * em
            and abs(box[1] - bounds[1]) <= 0.5 * em
        ]
        peers.sort(key=lambda item: item[2][0])
        if len(peers) >= 2 and all(b[2][0] - a[2][2] >= 2 * em for a, b in zip(peers, peers[1:])):
            verified = True
            for index, (other, run, box) in enumerate(peers):
                right = peers[index + 1][2][0] - em if index + 1 < len(peers) else page_width
                follows = [
                    line
                    for line in lines
                    if id(line) not in {id(item) for item in run}
                    and line.semantic_type is None
                    and abs(line.bbox[0] - box[0]) <= 0.4 * em
                    and box[3] <= line.bbox[1] <= box[3] + 3 * em
                    and line.bbox[2] < right
                    and line.effective_height <= 0.9 * em
                    and len(line.text.split()) >= 4
                ]
                graphic = any(
                    abs(image[0] - box[0]) <= 1.5 * em
                    and box[3] <= image[1] <= box[3] + 4 * em
                    and image[2] <= right
                    and image[3] - image[1] >= 3 * em
                    for image in images
                )
                if not follows and not graphic:
                    verified = False
            if verified:
                for other, run, _ in peers:
                    for line in run:
                        line.semantic_type = "paragraph_title"
                        line.structural_title = line.explicit_section_title = True
                        line.title_band_id = other.source_index
        if (seed.dominant_font_weight or 0) < 600 or bounds[2] - bounds[0] > 0.25 * page_width:
            continue
        side_peers = [
            (other, run, box)
            for other, run, box in runs
            if other.font_signature == seed.font_signature
            and (other.dominant_font_weight or 0) >= 600
            and abs(box[0] - bounds[0]) <= 0.3 * em
            and 0.9 * em <= other.effective_height <= 1.1 * em
        ]
        if len(side_peers) < 2:
            continue
        if all(
            any(
                image[0] <= box[0] + em
                and image[2] >= box[2] - em
                and (0 <= box[1] - image[3] <= 3 * em or 0 <= image[1] - box[3] <= 3 * em)
                and image[3] - image[1] >= 3 * em
                for image in images
            )
            for _, _, box in side_peers
        ):
            for other, run, _ in side_peers:
                for line in run:
                    line.semantic_type = "paragraph_title"
                    line.structural_title = line.explicit_section_title = True
                    line.title_band_id = other.source_index
