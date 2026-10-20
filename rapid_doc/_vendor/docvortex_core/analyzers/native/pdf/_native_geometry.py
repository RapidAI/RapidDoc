"""为批量数值内核提取轻量输入，保留 Python 的 Unicode 与对象语义。"""

from functools import lru_cache
from ....document.pdf.text._contracts import Bbox


def raw_bbox(value):
    """读取自有 Bbox 的底层数值，其它可迭代对象留给原校验器。"""
    return value.bbox if type(value) is Bbox else value


@lru_cache(maxsize=4096)
def glyph_flags(text: str) -> int:
    """仅缓存不可变字符串的类别，不持有页面或字符记录。"""
    return int(text.isprintable() and not text.isspace()) | (int(text.isspace()) << 1)
