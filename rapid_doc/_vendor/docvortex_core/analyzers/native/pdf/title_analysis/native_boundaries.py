"""以原生条目成员和正文连续性补充标题边界证据。"""

import re

from ..geometry import _bbox_center_y
from .structural import _set_native_title_band


def classify_bilingual_term_bands(lines):
    """编号、独立双语名称和后续定义共同确认术语标题，保留符号及来源为正文。"""
    ordered = sorted((line for line in lines if line.angle == 0), key=lambda line: (line.bbox[1], line.bbox[0]))
    for number in ordered:
        if number.semantic_type is not None or re.fullmatch(r"\d+(?:\.\d+)+", number.text.strip()) is None:
            continue
        em = max(number.effective_height, 1)
        followers = [
            line
            for line in ordered
            if line is not number
            and line.semantic_type is None
            and 0 <= line.bbox[1] - number.bbox[1] <= 2.5 * em
            and 0.3 * em <= line.bbox[0] - number.bbox[0] <= 20 * em
        ]
        if not followers:
            continue
        first = min(followers, key=lambda line: line.bbox[1])
        title = [line for line in followers if abs(_bbox_center_y(line.bbox) - _bbox_center_y(first.bbox)) <= 0.35 * em]
        text = " ".join(line.text for line in sorted(title, key=lambda line: line.bbox[0]))
        if (
            not re.search(r"[\u3400-\u9fff]", text)
            or not re.search(r"[A-Za-z]{3,}", text)
            or re.search(r"[。；：;:]", text)
            or len(text) > 140
        ):
            continue
        bottom = max(line.bbox[3] for line in title)
        definitions = [
            line
            for line in ordered
            if line not in title
            and line is not number
            and 0 <= line.bbox[1] - bottom <= 2.5 * em
            and line.bbox[0] >= number.bbox[0] - 0.5 * em
            and re.search(r"[\u3400-\u9fff]{3,}", line.text)
        ]
        if not definitions:
            continue
        _set_native_title_band([number, *title], lines)
        # 相邻条目之前的尾行是确定段界；正文符号及来源不能因字号差异再次晋升。
        preceding = [
            line
            for line in ordered
            if line.bbox[1] < number.bbox[1] and line.semantic_type is None and line.bbox[0] >= number.bbox[0] - em
        ]
        if preceding:
            max(preceding, key=lambda line: line.bbox[1]).paragraph_terminal = True
        next_numbers = [
            line.bbox[1]
            for line in ordered
            if line.bbox[1] > number.bbox[1]
            and (re.fullmatch(r"\d+(?:\.\d+)+", line.text.strip()) or re.match(r"\d+(?:\.\d+)*\s+\S", line.text.strip()))
        ]
        end = min(next_numbers, default=float("inf"))
        for line in ordered:
            if bottom <= line.bbox[1] < end and line.semantic_type is None:
                line.title_suppressed = True


def suppress_continuous_sentence_titles(lines):
    """紧接缩进续行的长句保持正文，并让标题分类和非首页升格共享反证。"""
    for line in lines:
        if (
            line.semantic_type not in {"paragraph_title", "doc_title"}
            or line.angle != 0
            or line.structural_title
            or line.explicit_section_title
            or line.caption_start
        ):
            continue
        em = max(line.effective_height, 1)
        if (
            len(re.findall(r"[\u3400-\u9fff]", line.text)) < 6
            or not re.search(r"[，,、；;]", line.text)
            or re.search(r"[。！？:：]\s*$", line.text)
        ):
            continue
        followers = [
            other
            for other in lines
            if other is not line
            and other.angle == 0
            and other.semantic_type in {None, "text"}
            and -0.25 * em <= other.bbox[1] - line.bbox[3] <= 0.85 * em
            and -3 * em <= other.bbox[0] - line.bbox[0] <= 0.5 * em
            and re.search(r"[\u3400-\u9fff]", other.text)
        ]
        if not followers:
            continue
        following = min(followers, key=lambda other: other.bbox[1])
        if following.bbox[2] - following.bbox[0] >= 0.8 * (line.bbox[2] - line.bbox[0]):
            previous = [
                other
                for other in lines
                if other is not line
                and other.angle == 0
                and other.semantic_type is None
                and not other.structural_title
                and 0 <= line.bbox[1] - other.bbox[3] <= em
                and abs(other.bbox[0] - line.bbox[0]) <= 0.5 * em
                and other.bbox[2] - other.bbox[0] >= 0.85 * (line.bbox[2] - line.bbox[0])
                and re.search(r"[，,、；;]", other.text)
                and not re.search(r"[。！？]\s*$", other.text)
            ]
            if not previous:
                continue
        line.title_suppressed = True
        if line.semantic_type in {"paragraph_title", "doc_title"}:
            line.semantic_type = None
            line.title_band_id = None
            line.paragraph_group = None


def classify_appendix_title_bands(lines, page_size):
    """居中附录标记、性质行和名称组成局部标题带，字母章节号确认其结构身份。"""
    for line in lines:
        marker = re.fullmatch(r"附\s*录\s*([A-Z])", line.text.strip())
        if marker is None or line.semantic_type is not None or line.angle != 0:
            continue
        em = max(line.effective_height, 1)
        center = 0.5 * (line.bbox[0] + line.bbox[2])
        members = [
            other
            for other in lines
            if other.angle == 0
            and other.semantic_type is None
            and line.bbox[1] <= other.bbox[1] <= line.bbox[3] + 3 * em
            and abs(0.5 * (other.bbox[0] + other.bbox[2]) - center) <= em
            and other.bbox[2] - other.bbox[0] < 0.5 * page_size[0]
        ]
        sections = [
            other
            for other in lines
            if re.match(r"[A-Z]\.\d+\s+\S", other.text)
            and 0 <= other.bbox[1] - max(member.bbox[3] for member in members) <= 4 * em
        ]
        if len(members) >= 2 and sections:
            _set_native_title_band(members, lines)


def order_leading_title_bands(blocks, page_size):
    """给居中页首标题与其紧邻同栏首节建立先后约束，不改动整页多栏序列。"""
    result = list(blocks)
    for title in blocks:
        box = title["bbox"]
        if (
            title.get("type") not in {"doc_title", "paragraph_title"}
            or title.get("angle", 0) != 0
            or box[1] > 0.3 * page_size[1]
            or box[2] - box[0] > 0.65 * page_size[0]
            or abs(0.5 * (box[0] + box[2]) - 0.5 * page_size[0]) > 0.08 * page_size[0]
        ):
            continue
        # 页首标题带之前已有正文或视觉内容时，它属于局部栏流；不得越过另一栏的前文。
        if any(
            other is not title
            and other.get("type") not in {"header", "footer", "page_number", "page_footnote"}
            and other["bbox"][1] < box[1] - 0.25 * (box[3] - box[1])
            for other in blocks
        ):
            continue
        if any(
            other is not title
            and other.get("type") not in {"header", "footer", "page_number", "page_footnote"}
            and min(other["bbox"][3], box[3]) - max(other["bbox"][1], box[1])
            > 0.2 * min(other["bbox"][3] - other["bbox"][1], box[3] - box[1])
            for other in blocks
        ):
            continue
        lower = [
            other
            for other in blocks
            if other is not title
            and other.get("type") in {"paragraph_title", "text"}
            and other["bbox"][0] < 0.25 * page_size[0]
            and 0 <= other["bbox"][1] - box[3] <= 1.5 * (box[3] - box[1])
        ]
        if not lower:
            continue
        section = min(lower, key=lambda other: other["bbox"][1])
        if result.index(title) > result.index(section):
            result.remove(title)
            result.insert(result.index(section), title)
    return _order_leading_shared_flow(result, page_size)


def _order_leading_shared_flow(blocks, page_size):
    """Let a leading single-line band precede its section in a proved shared flow.

    Ordering does not promote ordinary text to a heading. A wide multiline body
    establishes the flow; competing columns or visual blocks veto the move.
    """
    result = list(blocks)
    marginals = {"header", "footer", "page_number", "page_footnote"}
    ordinary = [b for b in blocks if b.get("type") not in marginals]
    for upper in ordinary:
        if upper.get("type") not in {"text", "doc_title", "paragraph_title"} or upper.get("angle", 0) != 0:
            continue
        box = upper["bbox"]
        height = box[3] - box[1]
        if height <= 0 or len(upper.get("_text_lines", [])) > 1:
            continue
        if any(b is not upper and b["bbox"][1] < box[1] - .25 * height for b in ordinary):
            continue
        sections = [b for b in ordinary if b is not upper
                    and b.get("type") in {"paragraph_title", "doc_title"}
                    and b.get("angle", 0) == 0
                    and 0 <= b["bbox"][1] - box[3] <= 1.5 * height]
        if not sections:
            continue
        section = min(sections, key=lambda b: b["bbox"][1])
        sb = section["bbox"]
        bodies = [b for b in ordinary if b.get("type") == "text" and b is not upper
                  and b.get("angle", 0) == 0 and len(b.get("_text_lines", [])) >= 2
                  and 0 <= b["bbox"][1] - sb[3] <= 2 * height
                  and b["bbox"][2] - b["bbox"][0] >= .55 * page_size[0]]
        if not bodies:
            continue
        body = min(bodies, key=lambda b: b["bbox"][1])
        bb = body["bbox"]
        width = bb[2] - bb[0]
        line_heights = [v for v in body.get("_line_heights", []) if v > 0]
        em = sorted(line_heights)[len(line_heights)//2] if line_heights else height
        if not (
            .65 * em <= height <= 1.6 * em
            and .15 * width <= box[2] - box[0] <= .65 * width
            and abs((box[0] + box[2]) - (bb[0] + bb[2])) <= .16 * width
            and -.5 * em <= sb[0] - bb[0] <= 3 * em
            and sb[2] <= bb[2] + .5 * em
        ):
            continue
        # No unrelated content may share the leading band's vertical region,
        # nor form a parallel flow next to the supporting body paragraph.
        if any(
            b is not upper and b is not section and b is not body
            and ((b["bbox"][1] < sb[3] and b["bbox"][3] > box[1])
                 or (b["bbox"][1] < bb[3] and b["bbox"][3] > bb[1]
                     and (b["bbox"][2] <= bb[0] or b["bbox"][0] >= bb[2])))
            for b in ordinary
        ):
            continue
        first, last = result.index(section), result.index(upper)
        if first < last and all(b is section or b is body for b in result[first:last]):
            result.remove(upper)
            result.insert(first, upper)
    return result


def protect_parallel_table_end_fields(lines, table_bboxes):
    """表后重复两行短字段的平行矩阵保留为正文，避免移除表体后误判签署栏为标题或脚注。"""
    for table in table_bboxes:
        short = [
            line
            for line in lines
            if line.semantic_type is None
            and line.angle == 0
            and line.bbox[1] > table[3] + 2 * max(line.effective_height, 1)
            and 2 <= len(re.sub(r"\s+", "", line.text)) <= 14
            and re.search(r"[\u3400-\u9fff]", line.text)
            and not re.match(r"[\d*†‡]|[一二三四五六七八九十]+[、.]", line.text.strip())
        ]
        rows = []
        for line in sorted(short, key=lambda line: (_bbox_center_y(line.bbox), line.bbox[0])):
            if not rows or abs(_bbox_center_y(line.bbox) - _bbox_center_y(rows[-1][0].bbox)) > 0.35 * line.effective_height:
                rows.append([line])
            else:
                rows[-1].append(line)
        bands = []
        for first, second in zip(rows, rows[1:]):
            if len(first) < 2 or len(second) < 2:
                continue
            em = max(line.effective_height for line in first + second)
            if 0 < second[0].bbox[1] - first[0].bbox[3] < 1.5 * em and all(
                any(abs(a.bbox[0] - b.bbox[0]) < 0.5 * em for b in second) for a in first[:2]
            ):
                bands.append(first + second)
        if len(bands) >= 2:
            for band in bands:
                for line in band:
                    line.semantic_type = "text"
                    line.title_suppressed = True
