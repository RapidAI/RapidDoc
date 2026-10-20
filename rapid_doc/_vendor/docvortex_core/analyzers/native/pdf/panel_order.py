"""为重复标题带下的独立正文栏建立PDF局部虚拟项，保留共享XYCut默认规则。"""

import statistics
import re

from .geometry import _bbox_union_many


def parallel_heading_panel_groups(blocks, excluded_indices):
    """同高同式标题、逐栏左缘和净空栏沟证明独立面板；跨栏正文及缺失主体拒绝分组。"""
    titles = [
        (index, block)
        for index, block in enumerate(blocks)
        if index not in excluded_indices and block.get("type") == "paragraph_title" and block.get("angle", 0) == 0
    ]
    bands = []
    for item in sorted(titles, key=lambda item: item[1]["bbox"][1]):
        box = item[1]["bbox"]
        if bands and abs(box[1] - bands[-1][0][1]["bbox"][1]) <= 0.5 * (box[3] - box[1]):
            bands[-1].append(item)
        else:
            bands.append([item])
    output = []
    for band in bands:
        if len(band) < 3:
            continue
        band.sort(key=lambda item: item[1]["bbox"][0])
        em = statistics.median(block["bbox"][3] - block["bbox"][1] for _, block in band)
        fonts = [block.get("_font_signatures") for _, block in band]
        if (
            em <= 0
            or any(not 0.9 * em <= block["bbox"][3] - block["bbox"][1] <= 1.1 * em for _, block in band)
            or not fonts[0]
            or any(font != fonts[0] for font in fonts)
            or any(b[1]["bbox"][0] - a[1]["bbox"][2] < 2 * em for a, b in zip(band, band[1:]))
        ):
            continue
        groups = []
        for column, (index, title) in enumerate(band):
            left = title["bbox"][0]
            right = band[column + 1][1]["bbox"][0] - em if column + 1 < len(band) else float("inf")
            members = [
                (i, block)
                for i, block in enumerate(blocks)
                if i not in excluded_indices
                and i != index
                and block.get("type") in {"text", "image"}
                and block.get("angle", 0) == 0
                and abs(block["bbox"][0] - left) <= 0.35 * em
                and block["bbox"][1] >= title["bbox"][3]
                and block["bbox"][2] <= right
            ]
            members.sort(key=lambda item: item[1]["bbox"][1])
            if not members or not 0.5 * em <= members[0][1]["bbox"][1] - title["bbox"][3] <= 3 * em:
                break
            bottom = members[0][1]["bbox"][3]
            accepted = [members[0]]
            for item in members[1:]:
                if item[1]["bbox"][1] - bottom > 3 * em:
                    break
                accepted.append(item)
                bottom = item[1]["bbox"][3]
            # 独立面板的上方指标标题与下方解释共属一栏，避免先遍历所有指标。
            preceding = [
                (i, block)
                for i, block in enumerate(blocks)
                if i not in excluded_indices
                and i != index
                and block.get("type") == "paragraph_title"
                and block.get("angle", 0) == 0
                and abs(block["bbox"][0] - left) <= 0.35 * em
                and block["bbox"][2] <= right
                and 0 <= title["bbox"][1] - block["bbox"][3] <= 2 * em
            ]
            groups.append([*preceding, (index, title), *accepted])
        if len(groups) != len(band):
            continue
        selected = {i for group in groups for i, _ in group}
        region = _bbox_union_many([block["bbox"] for group in groups for _, block in group])
        if any(
            i not in selected
            and block.get("type") not in {"header", "footer", "page_number"}
            and min(region[2], block["bbox"][2]) > max(region[0], block["bbox"][0])
            and min(region[3], block["bbox"][3]) > max(region[1], block["bbox"][1])
            for i, block in enumerate(blocks)
        ):
            continue
        output.extend(groups)
    return output


def numbered_step_member_groups(blocks, excluded_indices):
    """连续字母子步骤与前后同栏编号共同证明归属，排序时将父步骤和子项作为整体。"""
    available = [
        (i, block)
        for i, block in enumerate(blocks)
        if i not in excluded_indices
        and block.get("type") == "text"
        and block.get("angle", 0) == 0
        and isinstance(block.get("content"), str)
    ]
    children = [(i, block) for i, block in available if re.match(r"^[a-z][.)]\s", block["content"].strip())]
    children.sort(key=lambda item: (item[1]["bbox"][1], item[1]["bbox"][0]))
    output = []
    for i, first in children:
        if first["content"].lstrip()[0] != "a":
            continue
        em = first["bbox"][3] - first["bbox"][1]
        run = [(i, first)]
        for j, child in children:
            previous = run[-1][1]
            if child["bbox"][1] <= previous["bbox"][1]:
                continue
            if (
                ord(child["content"].lstrip()[0]) == ord(previous["content"].lstrip()[0]) + 1
                and abs(child["bbox"][0] - first["bbox"][0]) <= 0.3 * em
                and -0.15 * em <= child["bbox"][1] - previous["bbox"][3] <= 1.0 * em
            ):
                run.append((j, child))
            else:
                break
        if len(run) < 3:
            continue
        parents = [
            (j, block)
            for j, block in available
            if re.match(r"^\d+[.)]\s", block["content"].strip())
            and 0.5 * em <= first["bbox"][0] - block["bbox"][0] <= 4 * em
            and -0.2 * em <= first["bbox"][1] - block["bbox"][3] <= 1.5 * em
        ]
        if len(parents) != 1:
            continue
        parent = parents[0]
        number = int(re.match(r"^\d+", parent[1]["content"].strip())[0])
        following = [
            block
            for _, block in available
            if re.match(rf"^{number + 1}[.)]\s", block["content"].strip())
            and abs(block["bbox"][0] - parent[1]["bbox"][0]) <= 0.3 * em
            and -0.2 * em <= block["bbox"][1] - run[-1][1]["bbox"][3] <= 2 * em
        ]
        if len(following) == 1:
            output.append([parent, *run])
    return output
