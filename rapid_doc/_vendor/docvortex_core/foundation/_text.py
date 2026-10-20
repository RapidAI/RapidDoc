"""跨模型共享的文本字符规范化与换行连接规则。"""

import re
import unicodedata
from collections.abc import Sequence

from .language import remove_invalid_surrogates

# 使用固定码点区间覆盖扩展汉字、假名和中日标点，避免不同 Python 的 Unicode 版本改变结果。
_UNSPACED_RANGES = (
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF61, 0xFF9F),
    (0x1AFF0, 0x1AFFF),
    (0x1B000, 0x1B16F),
    (0x20000, 0x2EE5F),
    (0x2F800, 0x2FA1F),
    (0x30000, 0x323AF),
)
_CJK_PUNCTUATION = frozenset("，。！？；：、（）【】《》〈〉「」『』〔〕［］｛｝")
_CLOSING_PUNCTUATION = frozenset(",.;:!?%")
# 明确的复合词前缀与连接词只用于保守保留硬连字符，不承担语言识别或词典分词。
_COMPOUND_PREFIXES = frozenset({"open", "non", "self", "cross", "anti", "pre", "post", "co", "semi", "multi", "quasi"})
_COMPOUND_CONNECTORS = frozenset({"of", "the", "to", "and", "in", "for", "on", "by", "with"})
# 只投影已有的行内样式标记，不把任意 HTML 或公式解析为可改写内容。
_INLINE_STYLE_TAG_RE = re.compile(r"</?(?:sup|sub|b|strong|i|em|u|s|del|strike)\b[^>]*>", re.IGNORECASE)

# PDF 文本抽取时，英文跨行断词可能被编码为多种 hyphen 字符。
# 这里只用于判断“行末英文断词符”，不要扩展到 en/em dash 等普通破折号。
LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"
LINE_END_HYPHEN_RE = re.compile(rf"[A-Za-z]+[{re.escape(LINE_END_HYPHEN_CHARS)}]\s*$")

# URL 候选仅允许 RFC 3986 常见 ASCII 字符，避免把中文正文吞入链接。
_URL_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:https?|ftp)://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.ASCII | re.IGNORECASE,
)
# 下一行自身以完整 URL 开头时，必须保留边界，避免两条独立链接相连。
_URL_AT_LINE_START_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)",
    re.ASCII | re.IGNORECASE,
)


def is_hyphen_at_line_end(line: str) -> bool:
    """判断文本行是否以英文单词的跨行断词符结尾。

    只识别字母后紧跟行末 hyphen 的断词场景，不处理词内连字符或普通破折号。
    """
    return bool(LINE_END_HYPHEN_RE.search(line))


def _url_spans_line_boundary(previous_content: str, next_content: str) -> bool:
    """判断无空格候选中是否存在严格横跨当前物理行边界的 URL。"""
    stripped_previous = previous_content.rstrip()
    stripped_next = next_content.lstrip()
    if not stripped_previous or not stripped_next:
        return False
    candidate = f"{stripped_previous}{stripped_next}"
    boundary = len(stripped_previous)
    return any(match.start() < boundary < match.end() for match in _URL_CANDIDATE_RE.finditer(candidate))


def resolve_text_line_boundary(
    previous_content: str,
    *,
    next_content: str,
) -> tuple[str, str]:
    """返回处理后的上一行内容和本次物理行边界分隔符。

    严格横跨边界的 URL 候选直接连接，但下一行自身为完整 URL 时保留空格。
    按当前可见边界判断空格，不检测整段语言。汉字和假名直接连接，韩文与
    西文保留词间空格；既定 URL 和英文断词规则优先，且只改写真实的行末断词符。
    """
    processed_content = remove_invalid_surrogates(previous_content).rstrip()
    if not processed_content:
        return "", ""
    stripped_next = remove_invalid_surrogates(next_content).lstrip()
    if not stripped_next:
        return processed_content, ""
    if _url_spans_line_boundary(processed_content, stripped_next):
        if _URL_AT_LINE_START_RE.match(stripped_next):
            return processed_content, " "
        return processed_content, ""
    if is_hyphen_at_line_end(processed_content):
        previous_word = re.search(r"([A-Za-z]+)[-\u00ad\u2010\u2011\u2043]$", processed_content)
        # GLGE-difficult 等缩写复合词的连接符有语义，不能当作普通断词符删除。
        acronym = previous_word is not None and len(previous_word[1]) > 1 and previous_word[1].isupper()
        hard_compound = (
            previous_word is not None
            and processed_content.endswith("-")
            and (
                previous_word[1].lower() in _COMPOUND_PREFIXES
                or (
                    previous_word[1].lower() in _COMPOUND_CONNECTORS
                    and previous_word.start() > 0
                    and processed_content[previous_word.start() - 1] == "-"
                )
            )
        )
        if stripped_next[0].islower() and not acronym and not hard_compound:
            return processed_content[:-1], ""
        return processed_content, ""
    if re.search(r"\d[-–]$", processed_content) and stripped_next[0].isdigit():
        return processed_content, ""
    previous_char = _boundary_character(processed_content, from_end=True)
    next_char = _boundary_character(stripped_next, from_end=False)
    if not previous_char or not next_char:
        return processed_content, ""
    if _is_unspaced_character(previous_char) or _is_unspaced_character(next_char):
        return processed_content, ""
    if unicodedata.category(previous_char) in {"Ps", "Pi"}:
        return processed_content, ""
    if unicodedata.category(next_char) in {"Pe", "Pf"} or next_char in _CLOSING_PUNCTUATION:
        return processed_content, ""
    return processed_content, " "


def _boundary_character(content: str, *, from_end: bool) -> str:
    """读取可见边界字符，忽略样式标记及附着的组合字符，不改写原始正文。"""
    visible = _INLINE_STYLE_TAG_RE.sub("", content).strip()
    characters = reversed(visible) if from_end else iter(visible)
    return next((char for char in characters if unicodedata.category(char)[0] not in {"M", "C"}), "")


def _is_unspaced_character(char: str) -> bool:
    """判断汉字、假名及中日标点，韩文字母不属于无空格文字。"""
    return char in _CJK_PUNCTUATION or any(start <= ord(char) <= end for start, end in _UNSPACED_RANGES)


def merge_text_line_contents(
    line_contents: Sequence[str],
) -> str:
    """按累计文本上下文折叠物理行，支持跨越三行以上的 URL 连续拼接。"""

    normalized_lines = [cleaned for content in line_contents if (cleaned := remove_invalid_surrogates(str(content)).strip())]
    if not normalized_lines:
        return ""
    merged_content = normalized_lines[0]
    for current_line in normalized_lines[1:]:
        merged_content, separator = resolve_text_line_boundary(
            merged_content,
            next_content=current_line,
        )
        merged_content = f"{merged_content}{separator}{current_line}"
    return merged_content.strip()


def full_to_half_exclude_marks(text: str) -> str:
    """将全角英文字母和数字转换为半角形式，同时保留全角标点。"""
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return "".join(result)


def full_to_half(text: str) -> str:
    """将全角 ASCII 字母、数字和标点统一转换为半角形式。"""
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return "".join(result)


def clean_isolated_formula(content: str) -> str:
    """移除行间公式外层的反斜杠方括号并清理首尾空白。"""
    latex = content[:]
    if latex.startswith("\\["):
        latex = latex[2:]
    if latex.endswith("\\]"):
        latex = latex[:-2]
    return latex.strip()


def normalize_formula_tag_content(tag_content: str) -> str:
    """归一化公式编号文本，去掉全角字符和包裹括号后用于 \\tag{}。"""
    tag_content = full_to_half(str(tag_content or "").strip())
    if tag_content.startswith(("(", "﹙")):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith((")", "﹚")):
        tag_content = tag_content[:-1].strip()
    return tag_content


def normalize_formula_content_for_tag(formula_content: str) -> str:
    """归一化待合并编号的公式正文，去掉模型可能携带的展示公式分隔符。"""
    return clean_isolated_formula(str(formula_content or ""))


def build_tagged_formula_content(formula_content: str, tag_content: str) -> str | None:
    """将公式正文和编号文本合成为带 LaTeX tag 的纯公式内容。"""
    formula_content = normalize_formula_content_for_tag(formula_content)
    tag_content = normalize_formula_tag_content(tag_content)
    if not formula_content or not tag_content:
        return None
    return f"{formula_content}\\tag{{{tag_content}}}"
