"""按原生绘制颜色和正文嵌套关系排除独立浅灰数字噪声，不依赖字符的具体内容或字体名称。"""

import re
import math
from collections import defaultdict

from .geometry import _bbox_overlap_in_first, _bbox_union_many
from ...._compute_backend import get_native


def _gutter_noise_references(source, line, visual_bboxes):
    """完整图体之间的空白缝不属于任一图；短数字还需原生浅灰颜色与正文绘制对照才能删除。"""
    em = line.effective_height
    if not any(
        a[2] + 0.2 * em <= line.bbox[0]
        and line.bbox[2] <= b[0] - 0.2 * em
        and max(a[1], b[1]) <= line.bbox[1]
        and line.bbox[3] <= min(a[3], b[3])
        and a[3] - a[1] >= 8 * em
        and b[3] - b[1] >= 8 * em
        for a in visual_bboxes
        for b in visual_bboxes
    ):
        return []
    references = [
        peer
        for peer in source.lines
        if peer.font_signature != line.font_signature and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", peer.text)) >= 5
    ]
    return references if len(references) >= 3 else []


def _numeric_noise_native(source, visual_bboxes):
    """当前输入状态一次计算文字特征；特殊签名和数值继续执行 Python 参考路径。"""
    native = get_native()
    if native is None:
        return None
    lines = source.lines
    digits = [re.fullmatch(r"\d{1,3}", line.text.strip()) is not None for line in lines]
    eligible = [
        line.angle == 0 and line.semantic_type is None and bool(line.font_signature) and digit and line.effective_height > 0
        for line, digit in zip(lines, digits)
    ]
    if not any(eligible):
        return []
    if any(
        line.font_signature is not None
        and (type(line.font_signature) is not tuple or any(type(v) not in (str, int, type(None)) for v in line.font_signature))
        for line in lines
    ):
        return None
    numbers = [v for line in lines for v in (*line.bbox, line.effective_height)]
    numbers += [v for box in visual_bboxes for v in box]
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2**50 for v in numbers):
        return None
    groups, records = {}, []
    for line, digit, candidate in zip(lines, digits, eligible):
        font = groups.setdefault(line.font_signature, len(groups))
        english = len(re.findall(r"[A-Za-z]{2,}", line.text))
        chinese = len(re.findall(r"[\u3400-\u9fff]", line.text))
        records.append(
            (
                line.bbox,
                line.effective_height,
                line.angle == 0,
                bool(line.font_signature),
                font,
                digit,
                english + chinese >= 5,
                english >= 5 or chinese >= 12,
                re.match(r"\s*\d{1,2}[.)、]\s*[A-Za-z\u3400-\u9fff]", line.text) is not None,
                candidate,
            )
        )
    return [(lines[i], [lines[j] for j in refs]) for i, refs in native.numeric_noise_candidates(records, visual_bboxes)]


def _nested_prose_numeric_outliers(source, visual_bboxes=()):
    """优先批量匹配数值候选，返回同样的原始行对象与有序正文证据。"""
    candidates = _numeric_noise_native(source, visual_bboxes)
    return candidates if candidates is not None else _nested_prose_numeric_outliers_python(source, visual_bboxes)


def _nested_prose_numeric_outliers_python(source, visual_bboxes=()):
    """寻找独立字体的短数字叠入正常多行正文的情形；图表刻度、同字体数字和小号脚注不列为候选。"""
    candidates = []
    # 普通字体签名一次分组；非标准替身仍逐项比较，保留原等值与异常行为。
    font_groups = None
    if all(
        peer.font_signature is None
        or type(peer.font_signature) is tuple
        and all(type(value) in (str, int, type(None)) for value in peer.font_signature)
        for peer in source.lines
    ):
        font_groups = defaultdict(list)
        for peer in source.lines:
            font_groups[peer.font_signature].append(peer)
    for line in source.lines:
        if (
            line.angle != 0
            or line.semantic_type is not None
            or not line.font_signature
            or re.fullmatch(r"\d{1,3}", line.text.strip()) is None
            or line.effective_height <= 0
        ):
            continue
        same_font = (
            font_groups[line.font_signature]
            if font_groups is not None
            else [peer for peer in source.lines if peer.font_signature == line.font_signature]
        )
        gutter = _gutter_noise_references(source, line, visual_bboxes)
        if gutter:
            candidates.append((line, gutter))
            continue
        if len(same_font) > 4 or any(re.fullmatch(r"\d{1,3}", peer.text.strip()) is None for peer in same_font):
            continue
        prose = [
            peer
            for peer in source.lines
            if peer.angle == 0
            and peer.font_signature
            and peer.font_signature != line.font_signature
            and 0.7 * peer.effective_height <= line.effective_height <= 1.05 * peer.effective_height
            and (len(re.findall(r"[A-Za-z]{2,}", peer.text)) >= 5 or len(re.findall(r"[\u3400-\u9fff]", peer.text)) >= 12)
            and peer.bbox[0] <= line.bbox[0]
            and line.bbox[2] <= peer.bbox[2]
            and abs((peer.bbox[1] + peer.bbox[3] - line.bbox[1] - line.bbox[3]) / 2) <= 3 * peer.effective_height
        ]
        if len(prose) >= 2 and _bbox_overlap_in_first(line.bbox, _bbox_union_many([peer.bbox for peer in prose])) >= 0.95:
            candidates.append((line, prose))
            continue
        # 目录中的页码应与条目同行；大条目框间的孤立小数字、独立字体和浅色绘制需要另行裁决。
        entries = [
            peer
            for peer in source.lines
            if peer.angle == 0
            and peer.font_signature
            and peer.font_signature != line.font_signature
            and line.effective_height <= 0.5 * peer.effective_height
            and re.match(r"\s*\d{1,2}[.)、]\s*[A-Za-z\u3400-\u9fff]", peer.text)
            and peer.bbox[0] <= line.bbox[0]
            and line.bbox[2] <= peer.bbox[2]
        ]
        if len(entries) >= 3:
            em = min(peer.effective_height for peer in entries)
            entries.sort(key=lambda peer: peer.bbox[1])
            if max(peer.bbox[0] for peer in entries) - min(peer.bbox[0] for peer in entries) <= 0.3 * em and any(
                a.bbox[3] + 0.1 * em <= line.bbox[1] and line.bbox[3] <= b.bbox[1] - 0.1 * em
                for a, b in zip(entries, entries[1:])
            ):
                candidates.append((line, entries))
                continue
    return candidates


def exclude_faint_native_noise(source, read_paint, visual_bboxes=()):
    """先证明嵌套的独立数字字体，再与深色正文或背景上的白色条目对照浅灰噪声；证据不足保持原文。"""
    candidates = _nested_prose_numeric_outliers(source, visual_bboxes)
    if not candidates or read_paint is None:
        return set()
    requested = {
        char["char_idx"]
        for line, prose in candidates
        for peer in [line, *prose]
        for char in peer.chars
        if char.get("char", "").strip() and "char_idx" in char
    }
    paint = read_paint(sorted(requested))
    excluded = set()
    for line, prose in candidates:
        colors = [paint.get(char.get("char_idx")) for char in line.chars if char.get("char", "").strip()]
        if not colors or any(
            color is None
            or color[3] == 0
            or min(color[:3]) < 170
            or max(color[:3]) > 225
            or max(color[:3]) - min(color[:3]) > 35
            for color in colors
        ):
            continue
        references = [paint.get(char.get("char_idx")) for peer in prose for char in peer.chars if char.get("char", "").strip()]
        references = [color for color in references if color and color[3] > 0]
        if references and (
            sum(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2] <= 120 for color in references)
            >= 0.8 * len(references)
            or sum(min(color[:3]) >= 245 for color in references) >= 0.8 * len(references)
        ):
            excluded.add(line.source_index)
    if excluded:
        char_indices = {char.get("char_idx") for line in source.lines if line.source_index in excluded for char in line.chars}
        source.lines = [line for line in source.lines if line.source_index not in excluded]
        source.chars = [char for char in source.chars if char.get("char_idx") not in char_indices]
    return excluded
