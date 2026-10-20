"""识别通用学术文本角色和数学语法，不引用文档名称、变量名称或出版社名称。"""

import re
import unicodedata

_IDENTIFIER = r"[A-Za-z\u00c0-\u02af\u0370-\u03ff][A-Za-z0-9_\u00c0-\u02af\u0370-\u03ff]*"
_ATOM = r"(?:" + _IDENTIFIER + r"|\d+(?:\.\d+)?)(?:\s*[(\[{][^()\[\]{}\n]*[)\]}])*"
_EXPRESSION = re.compile(rf"{_ATOM}(?:\s*[+−\-*/=<>≤≥≠≈∝∈×÷^_]\s*{_ATOM})+")
_CALL = re.compile(rf"{_IDENTIFIER}\s*\([^()\n]*\)")
_ROLES = {
    "metadata": {"articleinfo", "articleinformation", "文章信息", "论文信息"},
    "abstract": {"abstract", "summary", "摘要", "内容摘要"},
    "affiliations": {"affiliations", "authoraffiliations", "作者单位", "作者机构", "机构信息"},
}
_FIELDS = {
    "received": r"^(?:received|收稿|收稿日期)",
    "revised": r"^(?:revised|修订|修回)",
    "accepted": r"^(?:accepted|录用|接受日期)",
    "published": r"^(?:available\s+online|published|出版日期|发表日期)",
    "keywords": r"^(?:key\s*words?|关键词|关键字)",
    "contact": r"(?:\bcorrespond(?:ing|ence)\b|通讯作者|通信作者|[\w.+-]+@[\w.-]+\.[a-z]{2,})",
}


def text_role(text: str) -> str | None:
    """仅对独立角色标题归一化，正文中出现同词不会升级为标题。"""
    normalized = re.sub(r"[\s:：]+", "", unicodedata.normalize("NFKC", text)).casefold()
    return next((role for role, names in _ROLES.items() if normalized in names), None)


def metadata_field(text: str) -> str | None:
    """提取独立元数据字段类别，多次出现同一日期字段不算多个角色。"""
    return next((role for role, pattern in _FIELDS.items() if re.search(pattern, text.strip(), re.IGNORECASE)), None)


def publication_text(text: str) -> bool:
    """出版标识及独立文章体裁提供页边上下文，不识别具体期刊或出版社名称。"""
    return bool(re.search(r"copyright|©|\bjournal\b|\bissn\b|\bdoi\b|出版|版权所有", text, re.IGNORECASE)) or (
        len(text.split()) <= 4
        and bool(
            re.search(r"\b(?:review|editorial|letter|research article|original article)\b|综述|研究论文", text, re.IGNORECASE)
        )
    )


def prose_residue(text: str) -> str:
    """去除语法连接的数学表达式，留下自然语言；自定义标识符由结构而非名称识别。"""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _EXPRESSION.sub(" ", normalized)
    return _CALL.sub(" ", normalized)


def has_prose(text: str, *, minimum_words: int = 3) -> bool:
    """数学表达式外的连续自然语言才构成正文证据；孤立标识符仍需空间分类。"""
    residue = prose_residue(text)
    return (
        bool(re.search(r"\b(?:where|with|when|from|that|then|the|this|these|which|is|are)\b", residue, re.IGNORECASE))
        or len(re.findall(r"(?<![A-Za-z\d])[A-Za-z]{2,}(?![A-Za-z\d])", residue)) >= minimum_words
        or len(re.findall(r"[\u3400-\u9fff]", residue)) >= minimum_words
    )
