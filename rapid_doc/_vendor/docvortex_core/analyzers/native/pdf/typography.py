"""提供原生文本与布局共享的字体族归一化。"""

from __future__ import annotations

import re
from contextvars import ContextVar
from functools import wraps

_FONT_FAMILY_CACHE = ContextVar("docvortex_page_font_family_cache", default=None)
_FONT_FAMILY_RE_REFERENCES = (re.sub, re._compile, getattr(getattr(re, "_compiler", None), "compile", None))


def _font_family_page_scope(function):
    """每个页面处理阶段建立独立字体文字缓存，结束或异常后释放并恢复外层状态。"""

    @wraps(function)
    def scoped(*args, **kwargs):
        """只复用本页普通字体名称的纯归一化结果，不缓存样式、成员或几何判断。"""
        token = _FONT_FAMILY_CACHE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _FONT_FAMILY_CACHE.reset(token)

    return scoped


def _normalized_font_family(
    signature: tuple[str, int] | None,
) -> str | None:
    """移除 PDF 字体子集前缀并归一化字体族名称，供几何续行作软兼容判断。"""

    if signature is None:
        return None
    raw = signature[0]
    cache = _FONT_FAMILY_CACHE.get()
    ordinary = (
        cache is not None
        and type(raw) is str
        and (re.sub, re._compile, getattr(getattr(re, "_compiler", None), "compile", None)) == _FONT_FAMILY_RE_REFERENCES
    )
    if ordinary and raw in cache:
        return cache[raw]
    name = re.sub(r"^[A-Z]{6}\+", "", raw)
    result = re.sub(r"[\s_-]+", "", name).casefold() or None
    if ordinary:
        if len(cache) >= 4096:
            cache.pop(next(iter(cache)))
        cache[raw] = result
    return result


__all__ = ["_normalized_font_family"]
