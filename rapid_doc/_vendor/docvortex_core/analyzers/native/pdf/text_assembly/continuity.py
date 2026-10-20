"""共享参考条目和重叠段落的成员证据，避免最后按外框盲目拼接文本。"""

from __future__ import annotations

import re
import statistics
import unicodedata

from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_y, _rotate_bbox_to_upright, _bbox_union_many
from ..line_layout import _line_effective_height
from ..layout_evidence import build_layout_evidence
from ..text_roles import text_role, metadata_field
from ..models import _LineItem, _PreparedPage
from .common import _merge_internal_text_block_group, _merge_text_line_content


_REFERENCE_NUMBER = re.compile(r"^\s*(?:[\[［](\d{1,4})[\]］]|(\d{1,3})[.])\s*(?=\D)")


def _is_reference_heading(text: str) -> bool:
    """参考标题允许汉字间排版空白；匹配时归一化而不改写原文内容。"""
    compact = re.sub(r"\s+", "", text).rstrip(":：").strip("[]［］【】()（）")
    return bool(re.fullmatch(r"(?:references|bibliography|参考文献(?:[（(]references[）)])?)", compact, re.I))


def group_native_inline_formula_prose(lines: list[_LineItem], regions: list[tuple]) -> None:
    """至少两处原生行内矢量公式证明同式正文连续带，公式槽位纳入行宽且字体重置阻断段组。"""
    if len(regions) < 2:
        return
    rows = []
    eligible = [
        line
        for line in lines
        if line.angle == 0
        and line.semantic_type is None
        and not line.caption_start
        and line.paragraph_group is None
        and line.font_signature is not None
    ]
    for line in sorted(eligible, key=lambda item: (item.bbox[1], item.bbox[0])):
        same = [
            row
            for row in rows
            if row[0].visual_row_id is not None
            and row[0].visual_row_id == line.visual_row_id
            and row[0].font_signature == line.font_signature
        ]
        if same:
            same[0].append(line)
        else:
            rows.append([line])
    by_source = {line.source_index: row for row in rows for line in row}
    boxes = {id(row): _bbox_union_many([line.bbox for line in row]) for row in rows}
    hits = {id(row): set() for row in rows}
    for index, (bounds, sources) in enumerate(regions):
        for source in sources:
            row = by_source.get(source)
            if row is not None:
                boxes[id(row)] = _bbox_union_many([boxes[id(row)], bounds])
                hits[id(row)].add(index)
    pending = sorted(rows, key=lambda row: boxes[id(row)][1])
    group = max((line.paragraph_group for line in lines if line.paragraph_group is not None), default=-1) + 1
    while pending:
        run = [pending.pop(0)]
        em = statistics.median(_line_effective_height(line, line.bbox) for line in run[0])
        while pending:
            before = boxes[id(run[-1])]
            current = pending[0]
            bounds = boxes[id(current)]
            if (
                current[0].font_signature != run[0][0].font_signature
                or any(not 0.85 <= _line_effective_height(line, line.bbox) / em <= 1.15 for line in current)
                or not -0.15 * em <= bounds[1] - before[3] <= 0.8 * em
                or _bbox_axis_overlap_ratio(before, bounds, axis="x") < 0.5
            ):
                break
            run.append(pending.pop(0))
        text = " ".join(line.text for row in run for line in row)
        if (
            len(run) < 3
            or len(set().union(*(hits[id(row)] for row in run))) < 2
            or len(re.findall(r"\b[A-Za-z]{2,}\b", text)) < 20
        ):
            continue
        for row in run:
            for line in row:
                line.paragraph_group = group
                line.semantic_type = "text"
                line.title_suppressed = True
        group += 1


def _standalone_author_year_regions(page: _PreparedPage) -> list[tuple]:
    """单页缺少参考文献标题时，以重复作者行、年代和悬挂缩进共同确认局部参考栏。"""
    lines = [
        line
        for line in page.remaining_lines
        if line.angle == 0 and line.semantic_type not in {"header", "footer", "page_number", "page_footnote"}
    ]
    if len(lines) < 15:
        return []
    layout = build_layout_evidence(lines, page.page_size, barriers=[block["bbox"] for block in page.fixed_blocks])
    regions = []
    for left, right in sorted(set(layout.corridor((lane.left, 0, lane.right, 1)) for lane in layout.lanes)):
        members = [line for line in lines if left <= (line.bbox[0] + line.bbox[2]) / 2 < right]
        if len(members) < 15:
            continue
        em = statistics.median(_line_effective_height(line, line.bbox) for line in members)
        edge = min(line.bbox[0] for line in members)
        starts = [
            line
            for line in members
            if abs(line.bbox[0] - edge) <= 0.4 * em
            and line.text.count(",") >= 2
            and re.match(r"^[A-Z][^,]+,\s*[A-Z]", line.text)
        ]
        indented = [line for line in members if 0.5 * em < line.bbox[0] - edge < 2 * em]
        dated = [line for line in members if re.search(r"\b(?:19|20)\d{2}[a-z]?\.", line.text)]
        if len(starts) >= 4 and len(dated) >= 4 and len(indented) >= len(members) / 3:
            regions.append(
                (
                    left,
                    min(line.bbox[1] for line in members) - 0.2 * em,
                    right,
                    max(line.bbox[3] for line in members) + 0.2 * em,
                )
            )
    return regions


def _numbered_reference_regions(page: _PreparedPage, active: bool) -> list[tuple]:
    """由连续编号的重复左缘恢复局部参考栏，栏顶不沿整页正文栏传播。"""
    candidates = []
    for line in page.remaining_lines:
        match = _REFERENCE_NUMBER.match(line.text)
        if line.angle == 0 and match and line.semantic_type not in {"header", "footer", "page_number"}:
            candidates.append((line, int(match.group(1) or match.group(2)), bool(match.group(1))))
    clusters = []
    for item in sorted(candidates, key=lambda item: item[0].bbox[0]):
        em = _line_effective_height(item[0], item[0].bbox)
        if clusters and abs(item[0].bbox[0] - statistics.median(v[0].bbox[0] for v in clusters[-1])) <= 0.8 * em:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    accepted = []
    for cluster in clusters:
        cluster.sort(key=lambda item: item[0].bbox[1])
        # 参考标题所在栏可能仅容纳首条，随后条目移到下一栏；标题和方括号共同确认它。
        if (
            len(cluster) == 1
            and cluster[0][2]
            and any(
                _is_reference_heading(line.text)
                and line.bbox[3] <= cluster[0][0].bbox[1]
                and _bbox_axis_overlap_ratio(line.bbox, cluster[0][0].bbox, axis="x") >= 0.5
                for line in page.remaining_lines
            )
        ):
            accepted.append(cluster)
            continue
        minimum = 2 if active else 3
        if len(cluster) < minimum or sum(b[1] == a[1] + 1 for a, b in zip(cluster, cluster[1:])) < minimum - 1:
            continue
        if not active and not all(item[2] for item in cluster):
            continue
        accepted.append(cluster)
    if not accepted:
        return []
    page.bracketed_reference_numbers = sum(item[2] for cluster in accepted for item in cluster) >= 0.75 * sum(
        len(cluster) for cluster in accepted
    )
    accepted.sort(key=lambda c: c[0][0].bbox[0])
    regions = []
    for i, cluster in enumerate(accepted):
        em = statistics.median(_line_effective_height(v[0], v[0].bbox) for v in cluster)
        left = statistics.median(v[0].bbox[0] for v in cluster)
        next_left = min(v[0].bbox[0] for v in accepted[i + 1]) if i + 1 < len(accepted) else page.page_size[0]
        top = min(v[0].bbox[1] for v in cluster)
        headings = [line for line in page.remaining_lines if _is_reference_heading(line.text)]
        local_headings = [line for line in headings if left - em <= line.bbox[0] < next_left - em]
        floor = max(line.bbox[3] for line in local_headings) if local_headings else 0.0
        # 后栏首项前可能有上一条的续行，但必须与参考文字同字号、同左缘。
        if active or i:
            peers = [
                peer
                for peer in page.remaining_lines
                if left - em <= peer.bbox[0] < left + 3 * em
                and peer.bbox[2] < next_left
                and floor <= peer.bbox[1] < top
                and peer.semantic_type not in {"header", "footer", "page_number"}
                and not peer.structural_title
                and not peer.explicit_section_title
                and not _is_reference_heading(peer.text)
                and 0.85 <= _line_effective_height(peer, peer.bbox) / em <= 1.15
            ]
            if peers:
                top = min(top, min(peer.bbox[1] for peer in peers))
        endings = [
            line.bbox[1]
            for line in page.remaining_lines
            if top < line.bbox[1]
            and left - em <= line.bbox[0] < next_left - em
            and (
                re.match(r"^\s*(?:appendix\b|附录)", line.text, re.I)
                or line.semantic_type == "paragraph_title"
                and (line.structural_title or line.explicit_section_title)
                and not _REFERENCE_NUMBER.match(line.text)
                and all(
                    line.bbox[1] - other.bbox[3] >= 0.6 * em
                    for other in page.remaining_lines
                    if other is not line
                    and left - em <= other.bbox[0] < next_left - em
                    and other.bbox[1] < line.bbox[1]
                    and _bbox_axis_overlap_ratio(other.bbox, line.bbox, axis="x") >= 0.2
                )
            )
        ]
        # 后页译文的通栏题名和独立摘要形成参考区屏障，不能把译文压进左栏末条文献。
        abstract_tops = [
            line.bbox[1]
            for line in page.remaining_lines
            if re.match(r"^\s*(?:abstract|摘要)\s*[:：]", unicodedata.normalize("NFKC", line.text), re.I)
        ]
        endings.extend(
            line.bbox[1]
            for line in page.remaining_lines
            if line.bbox[1] > top
            and line.bbox[2] - line.bbox[0] >= 0.6 * page.page_size[0]
            and abs((line.bbox[0] + line.bbox[2]) / 2 - page.page_size[0] / 2) <= 0.05 * page.page_size[0]
            and len(re.findall(r"[A-Za-z]", unicodedata.normalize("NFKC", line.text))) >= 20
            and not _REFERENCE_NUMBER.match(line.text)
            and any(2 * em <= abstract - line.bbox[3] <= 20 * em for abstract in abstract_tops)
        )
        regions.append((left - em, top - 0.2 * em, next_left - em, min(endings) if endings else 0.94 * page.page_size[1]))
    return regions


def mark_document_reference_regions(pages: list[_PreparedPage]) -> None:
    """沿实际栏序处理参考文献开始及章节结束事件，同页附录不丢弃此前的有效条目。"""
    active = False
    numbered_active = False
    for page in pages:
        _width, height = page.page_size
        lines = page.remaining_lines
        headings = {line.source_index for line in lines if _is_reference_heading(line.text)}
        # 作者年代式参考区不能为后续附录的普通数字列表提供编号文献证据。
        numbered = _numbered_reference_regions(page, numbered_active or bool(headings))
        if numbered:
            page.numbered_references = True
            page.reference_regions.extend(numbered)
            for line in lines:
                if line.source_index in headings:
                    line.semantic_type = "paragraph_title"
                    line.structural_title = True
                    page.page_footnote_groups = [
                        g - {line.source_index} for g in page.page_footnote_groups if g - {line.source_index}
                    ]
            active = numbered[-1][3] == 0.94 * height
            numbered_active = active
            continue
        # 编号式参考区必须在下一页继续得到编号证据，不能退化为整页作者年代区。
        if numbered_active:
            active = False
        numbered_active = False
        if not active and not headings:
            page.reference_regions.extend(_standalone_author_year_regions(page))
            continue
        layout = build_layout_evidence(lines, page.page_size, barriers=[block["bbox"] for block in page.fixed_blocks])
        corridors = sorted(set(layout.corridor((lane.left, 0, lane.right, 1)) for lane in layout.lanes))
        for left, right in corridors:
            top = 0.055 * height if active else None
            members = sorted(
                [line for line in lines if left <= (line.bbox[0] + line.bbox[2]) / 2 < right], key=lambda line: line.bbox[1]
            )
            for line in members:
                if line.source_index in headings:
                    if top is not None and line.bbox[1] > top:
                        page.reference_regions.append((left, top, right, line.bbox[1]))
                    top, active = line.bbox[3], True
                elif active and (
                    re.match(r"^\s*(?:appendix\b|附录)", line.text, re.IGNORECASE)
                    or line.semantic_type == "paragraph_title"
                    and (line.structural_title or line.explicit_section_title)
                ):
                    if top is not None and line.bbox[1] > top:
                        page.reference_regions.append((left, top, right, line.bbox[1]))
                    top, active = None, False
            if top is not None and top < 0.94 * height:
                page.reference_regions.append((left, top, right, 0.94 * height))


def group_reference_lines(lines: list[_LineItem], page: _PreparedPage) -> None:
    """按稳定首行左缘与悬挂续行分组，允许连续单行条目和断词链接。"""
    group = 0
    for region in page.reference_regions:
        members = [
            line
            for line in lines
            if line.angle == 0
            and region[0] <= (line.bbox[0] + line.bbox[2]) / 2 < region[2]
            and region[1] <= _bbox_center_y(line.bbox) <= region[3]
            and line.semantic_type not in {"page_number", "doc_title", "header", "footer"}
            and (page.numbered_references or line.semantic_type != "page_footnote")
            and (not page.numbered_references or line.bbox[2] - line.bbox[0] <= 1.15 * (region[2] - region[0]))
            and (not page.numbered_references or re.search(r"[^\W_]", line.text))
        ]
        if len(members) < (1 if page.numbered_references else 3):
            continue
        em = statistics.median(_line_effective_height(line, line.bbox) for line in members)
        left = min(line.bbox[0] for line in members)
        members.sort(key=lambda line: (_bbox_center_y(line.bbox), line.bbox[0]))
        biography_tops = [
            line.bbox[1]
            for line in members
            if re.search(r"[(（]\s*\d{4}\s*[—–-]", line.text)
            and re.search(r"研究|博士|本科|教授|research|born", line.text, re.IGNORECASE)
        ]
        biography_tops.extend(
            line.bbox[1]
            for line in members
            if re.match(
                r"^(?:publisher[’']s\s+note|author\s+(?:information|contributions)|competing\s+interests)\b",
                line.text.strip(),
                re.IGNORECASE,
            )
        )
        biography_tops.extend(
            line.bbox[1] for line in members if re.match(r"^\s*[［\[]?\s*(?:责任编辑|编者|编辑说明)\s*[:：]", line.text)
        )
        if biography_tops:
            members = [line for line in members if line.bbox[1] < min(biography_tops)]
        # 没有重复首行和续行缩进就不推断条目，以免普通附注变成逐行段落。
        if sum(abs(line.bbox[0] - left) <= 0.5 * em for line in members) < 1:
            continue
        numbered = page.numbered_references and any(_reference_entry_number(line.text, page) for line in members)
        if not numbered and sum(0.5 * em < line.bbox[0] - left < 4 * em for line in members) < 2:
            continue
        rows: list[list[_LineItem]] = []
        for line in members:
            if (
                rows
                and abs(_bbox_center_y(line.bbox) - statistics.median(_bbox_center_y(item.bbox) for item in rows[-1]))
                < 0.6 * em
            ):
                rows[-1].append(line)
            else:
                rows.append([line])
        for index, row in enumerate(rows):
            row.sort(key=lambda line: line.bbox[0])
            starts = bool(_reference_entry_number(row[0].text, page)) or (not numbered and row[0].bbox[0] - left <= 0.5 * em)
            if index == 0 or starts:
                group += 1
            for line in row:
                line.paragraph_group = group
                if numbered:
                    line.reference_start = starts
                line.semantic_type = None
                line.title_suppressed = True
        claimed = {line.source_index for line in members}
        page.page_footnote_groups = [group - claimed for group in page.page_footnote_groups if group - claimed]


def _reference_entry_number(text: str, page: _PreparedPage):
    """沿已确认的文献编号样式判定首行，拒绝方括号条目中以页码句点开头的续行。"""
    match = _REFERENCE_NUMBER.match(text)
    return match if match and (not page.bracketed_reference_numbers or match.group(1)) else None


def _reassemble_member_lines(members: list[_LineItem], previous_content: str = "") -> str:
    """按实际行和水平顺序重建合并内容，嵌在段落中间的短 run 不会被追加到段尾。"""
    unique = {line.source_index: line for line in members}
    ordered = sorted(unique.values(), key=lambda line: (_bbox_center_y(line.ink_bbox or line.bbox), line.bbox[0]))
    rows: list[list[_LineItem]] = []
    for line in ordered:
        bbox = line.ink_bbox or line.bbox
        if rows:
            row = rows[-1]
            center = statistics.median(_bbox_center_y(item.ink_bbox or item.bbox) for item in row)
            em = max(
                _line_effective_height(line, bbox), statistics.median(_line_effective_height(item, item.bbox) for item in row)
            )
            if abs(_bbox_center_y(bbox) - center) <= 0.55 * em:
                row.append(line)
                continue
        rows.append([line])
    contents = []
    for row in rows:
        ordered_row = sorted(row, key=lambda item: item.bbox[0])
        content = ordered_row[0].text
        for previous, line in zip(ordered_row, ordered_row[1:]):
            if re.match(r"^\[\d+\]", line.text) and previous.text[-1:].isalnum() and previous.bbox[2] >= line.bbox[0]:
                content += line.text
            else:
                content = _merge_text_line_content([content, line.text])
        contents.append(content)
    rebuilt = _merge_text_line_content(contents)
    # 仅当成员次序需要修复时重建文字，已完成的空格与断词恢复不能被原始 run 覆盖。
    if previous_content and re.sub(r"\s+", "", previous_content) == re.sub(r"\s+", "", rebuilt):
        return previous_content
    return rebuilt


def order_body_above_reference_band(blocks: list[dict], page: _PreparedPage) -> list[dict]:
    """混合栏页的上方正文按自身栏序排列，上标顶边不能把右栏首段提前。"""
    if not page.numbered_references or not page.reference_regions:
        return blocks
    top = min(region[1] for region in page.reference_regions)
    body = [
        block
        for block in blocks
        if block.get("type") == "text"
        and block["bbox"][3] < top
        and block.get("_text_lines")
        and block.get("_reference_group") is None
        and block["bbox"][2] - block["bbox"][0]
        >= 3 * statistics.median(_line_effective_height(line, line.bbox) for line in block["_text_lines"])
    ]
    if len(body) < 4:
        return blocks
    layout = build_layout_evidence([line for block in body for line in block["_text_lines"]], page.page_size)
    if len(layout.lanes) < 2:
        return blocks
    keys = {}
    for block in body:
        lane = layout.corridor(block["bbox"])
        if lane is None or lane == (0.0, layout.width):
            return blocks
        keys[id(block)] = (lane[0], block["bbox"][1])
    indices = [index for index, block in enumerate(blocks) if id(block) in keys]
    if any(
        block.get("type") in {"image", "table", "equation", "paragraph_title"}
        for block in blocks[min(indices) : max(indices) + 1]
    ):
        return blocks
    ordered = iter(sorted(body, key=lambda block: keys[id(block)]))
    return [next(ordered) if id(block) in keys else block for block in blocks]


def group_front_matter_lines(lines: list[_LineItem], page_size: tuple) -> None:
    """以首页摘要之前连续的机构编号组织作者附属信息，邮箱不另立小标题。"""
    abstracts = [line for line in lines if text_role(line.text) == "abstract"]
    if not abstracts:
        # 无显式摘要标题时，独立的出版日期带仍是作者机构区的可靠下界。
        abstracts = [line for line in lines if re.match(r"^\s*[（(]?(?:received|accepted|published)\b", line.text, re.I)]
    if not abstracts:
        return
    end = min(line.bbox[1] for line in abstracts)
    candidates = [line for line in lines if 0.12 * page_size[1] <= line.bbox[1] < end and line.angle == 0]
    markers = [(line, re.match(r"^\s*(\d{1,2})\s+\D", line.text)) for line in candidates]
    markers = [(line, int(match.group(1))) for line, match in markers if match is not None]
    if not markers:
        return
    # 单个机构也可成立，但须有机构或联系方式证据，避免把首页普通编号列表当作者单位。
    if not any(
        re.search(
            r"university|institute|department|laboratory|centre|center|大学|学院|研究所|实验室|医院", line.text, re.IGNORECASE
        )
        or metadata_field(line.text) == "contact"
        for line in candidates
    ):
        return
    first_y = min(line.bbox[1] for line, _ in markers)
    marker_left = statistics.median(line.bbox[0] for line, _ in markers)
    marker_by_source = {line.source_index: number for line, number in markers}
    for line in candidates:
        if re.fullmatch(r"\d{1,2}", line.text.strip()) and abs(line.bbox[0] - marker_left) < line.effective_height:
            marker_by_source[line.source_index] = int(line.text.strip())
    group = None
    previous = None
    for line in sorted(candidates, key=lambda line: (_bbox_center_y(line.ink_bbox or line.bbox), line.bbox[0])):
        if line.bbox[1] < first_y:
            continue
        if line.source_index in marker_by_source:
            group = -marker_by_source[line.source_index]
        elif previous is not None and line.bbox[1] - previous.bbox[3] > 1.5 * max(
            line.effective_height, previous.effective_height
        ):
            group = None
        if re.match(r"^(?:received|accepted|date|draft)\b", line.text, re.IGNORECASE):
            group = None
        if group is not None:
            line.paragraph_group = group
            line.semantic_type = None
            line.title_suppressed = True
        previous = line


def merge_overlapping_member_blocks(blocks: list[dict], page_size: tuple) -> list[dict]:
    """以同栏实际行的重叠关系回收段内小块，不能仅因外接矩形相交就跨栏合并。"""
    output = list(blocks)
    layouts = {
        angle: build_layout_evidence(
            [line for block in blocks for line in block.get("_text_lines", [])], page_size, angle=angle
        )
        for angle in {line.angle for block in blocks for line in block.get("_text_lines", [])}
    }
    groups: dict[int, list[int]] = {}
    for index, block in enumerate(output):
        if block.get("_reference_group") is not None:
            groups.setdefault(block["_reference_group"], []).append(index)
    consumed = set()
    for indices in groups.values():
        merged = _merge_internal_text_block_group(output, indices)
        merged["content"] = _reassemble_member_lines(merged["_text_lines"], merged["content"])
        output[indices[0]] = merged
        consumed.update(indices[1:])
    output = [block for index, block in enumerate(output) if index not in consumed]
    while True:
        match = None
        for i, first in enumerate(output):
            if first.get("type") != "text" or not first.get("_text_lines"):
                continue
            for j in range(i + 1, len(output)):
                second = output[j]
                if second.get("type") != "text" or not second.get("_text_lines"):
                    continue
                fb, sb = first["bbox"], second["bbox"]
                later = second if sb[1] >= fb[1] else first
                if later.get("_geometry_break_before"):
                    continue
                if later.get("_explicit_break_before") or later.get("_rule_break_before"):
                    if not (first.get("_visual_row_ids", set()) & second.get("_visual_row_ids", set())):
                        continue
                first_text = str(first.get("content", "")).rstrip()
                continuation = (
                    first_text.endswith("-")
                    and str(second.get("content", ""))[:1].islower()
                    or re.match(r"^[•●▪]", first_text)
                    and first_text.endswith((":", "："))
                )
                em = statistics.median(_line_effective_height(line, line.bbox) for line in first["_text_lines"])
                adjacent = continuation and 0 <= sb[1] - fb[3] <= 1.5 * em
                numbered_continuation = (
                    re.match(r"^\d{1,2}[.]\s", first_text) is not None
                    and 0.5 * em <= sb[0] - fb[0] <= 4 * em
                    and 0 <= sb[1] - fb[3] <= em
                    and second.get("_protected_hard_break_before") is not True
                )
                adjacent = adjacent or numbered_continuation
                math_continuation = (
                    (
                        first.get("_inline_math_regions")
                        or first.get("_paragraph_formula_context")
                        or any(line.paragraph_formula_context for line in first["_text_lines"])
                    )
                    and str(second.get("content", ""))[:1].islower()
                    and not re.search(r"[.!?。！？]$", first_text)
                    and abs(fb[0] - sb[0]) <= em
                    and -em <= sb[1] - fb[3] <= em
                    and second.get("_protected_hard_break_before") is not True
                )
                adjacent = adjacent or bool(math_continuation)
                narrow, host = sorted((first, second), key=lambda block: block["bbox"][2] - block["bbox"][0])
                nb, hb = narrow["bbox"], host["bbox"]
                inline_prefix = (
                    len(narrow["_text_lines"]) == 1
                    and re.fullmatch(r"[^\W\d_]{1,2}", str(narrow["content"]).strip())
                    and nb[2] - nb[0] <= 1.5 * em
                    and 0 <= hb[0] - nb[2] <= 1.5 * em
                    and hb[1] <= _bbox_center_y(nb) <= hb[3]
                    and len(host.get("_font_signatures", set())) >= 2
                )
                if (
                    not adjacent
                    and min(fb[3], sb[3]) <= max(fb[1], sb[1])
                    or _bbox_axis_overlap_ratio(fb, sb, axis="x") < 0.75
                    and not inline_prefix
                ):
                    continue
                if first.get("_reference_group") != second.get("_reference_group"):
                    continue
                if not adjacent and first.get("_hanging_indent_group") != second.get("_hanging_indent_group"):
                    continue
                connected = adjacent
                for a in [] if connected else first["_text_lines"]:
                    for b in second["_text_lines"]:
                        if a.angle != b.angle:
                            continue
                        ab, bb = a.ink_bbox or a.bbox, b.ink_bbox or b.bbox
                        em = max(_line_effective_height(a, ab), _line_effective_height(b, bb))
                        # 两条完整文字行横向分离时是相邻栏，外接矩形重叠不能让它们互相认领。
                        # 小型上下标、公式碎片仍可依靠邻接归入宿主正文。
                        across_columns = layouts[a.angle].separated(
                            _rotate_bbox_to_upright(ab, page_size, a.angle),
                            _rotate_bbox_to_upright(bb, page_size, b.angle),
                        )
                        if across_columns and max(ab[0] - bb[2], bb[0] - ab[2]) > 0.5 * em:
                            continue
                        if (
                            _bbox_axis_overlap_ratio(ab, bb, axis="y") >= 0.25
                            and max(0.0, ab[0] - bb[2], bb[0] - ab[2]) <= 5 * em
                        ):
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    match = (i, j)
                    break
            if match:
                break
        if match is None:
            return output
        i, j = match
        merged = _merge_internal_text_block_group(output, [i, j])
        widths = [output[index]["bbox"][2] - output[index]["bbox"][0] for index in (i, j)]
        heights = [_line_effective_height(line, line.bbox) for index in (i, j) for line in output[index]["_text_lines"]]
        if min(widths) < 0.4 * max(widths) and min(heights) < 0.85 * max(heights):
            merged["_paragraph_formula_context"] = True
        merged["content"] = _reassemble_member_lines(merged["_text_lines"], merged["content"])
        output[i] = merged
        output.pop(j)
