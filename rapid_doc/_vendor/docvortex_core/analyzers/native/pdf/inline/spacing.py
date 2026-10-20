"""在自然语言块最终物化时补词界，避免影响代码、公式及布局分类。"""

from dataclasses import dataclass

from .....document.pdf.text.spacing import is_generated_cjk_space, is_ligature_expansion_space, needs_tight_space
from .....foundation._text import is_hyphen_at_line_end
from .common import _normalize_match_fragment
from .matching import _assign_lines_to_blocks, _project_content_chars, _match_line_without_terminal_hyphen


@dataclass(frozen=True, slots=True)
class PDFTextSpacingLine:
    """只保留存在缺失词界的紧凑行证据，不延长字符几何的生命周期。"""

    bbox: tuple[float, float, float, float]
    text: str
    source_index: int
    boundaries: tuple[int, ...]
    suppressed_boundaries: tuple[int, ...] = ()


def prepare_spacing_lines(lines, space_before=None):
    """按原视觉行提取可恢复的词界偏移，原始行文本和字符保持只读。"""
    output = []
    for line in lines:
        if line.formula_candidate_only:
            continue
        positions = {
            index
            for index in range(1, len(line.chars))
            if (
                needs_tight_space(line.chars[index - 1], line.chars[index])
                if space_before is None
                else line.chars[index].get("char_idx") in space_before
                and line.chars[index - 1].get("char_idx", -2) + 1 == line.chars[index].get("char_idx")
            )
        }
        # 断词续行即使没有新词界也保留首尾上下文，复用既有连字符投影规则。
        continuation = bool(
            output and output[-1].source_index + 1 == line.source_index and is_hyphen_at_line_end(output[-1].text)
        )
        suppressed = {
            index
            for index in range(1, len(line.chars) - 1)
            if is_generated_cjk_space(line.chars[index - 1], line.chars[index], line.chars[index + 1])
            or is_ligature_expansion_space(line.chars, index)
        }
        if not positions and not continuation and not suppressed:
            continue
        parts, boundaries, suppressed_boundaries = [], [], []
        cursor = 0
        for index, char in enumerate(line.chars):
            if index in positions:
                boundaries.append(cursor)
            if index in suppressed:
                suppressed_boundaries.append(cursor)
            fragment = _normalize_match_fragment(char.get("char"))
            parts.append(fragment)
            cursor += len(fragment)
        if boundaries or continuation or suppressed_boundaries:
            output.append(
                PDFTextSpacingLine(
                    line.bbox, "".join(parts), line.source_index, tuple(boundaries), tuple(suppressed_boundaries)
                )
            )
    return output


def apply_spacing_lines(blocks, lines, page_size):
    """仅对完整匹配且原文中没有分隔符的自然语言行补空格。"""
    if not lines:
        return
    for block_index, assigned in _assign_lines_to_blocks(blocks, lines, page_size).items():
        block = blocks[block_index]
        content = block["content"]
        projected = _project_content_chars(content)
        text = "".join(token.value for token in projected)
        offsets = set()
        suppressed_ranges = set()
        cursor = 0
        for line_index, line in enumerate(assigned):
            start = text.find(line.text, cursor)
            if start < 0:
                next_line = assigned[line_index + 1] if line_index + 1 < len(assigned) else None
                match = _match_line_without_terminal_hyphen(text, line, next_line, cursor)
                if match is None:
                    continue
                start, cursor = match.start, match.end
            else:
                cursor = start + len(line.text)
            for boundary in line.boundaries:
                index = start + boundary
                if not 0 < index < len(projected):
                    continue
                left, right = projected[index - 1], projected[index]
                if left.raw_end == right.raw_start and not right.formula_gap_before:
                    offsets.add(right.raw_start)
            for boundary in line.suppressed_boundaries:
                index = start + boundary
                if 0 < index < len(projected):
                    left, right = projected[index - 1], projected[index]
                    if content[left.raw_end : right.raw_start] == " " and not right.formula_gap_before:
                        suppressed_ranges.add((left.raw_end, right.raw_start))
        if offsets or suppressed_ranges:
            parts, cursor = [], 0
            edits = [(offset, offset, " ") for offset in offsets] + [(start, end, "") for start, end in suppressed_ranges]
            for start, end, replacement in sorted(edits):
                parts.extend((content[cursor:start], replacement))
                cursor = end
            parts.append(content[cursor:])
            block["content"] = "".join(parts)
