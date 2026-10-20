"""PDF 原生子资源关闭，保持原生提取算法与资源语义。"""

from __future__ import annotations

import logging

logger = logging.getLogger("rapid_doc._vendor.docvortex_core.document.pdf._document")


def _try_close(obj: object) -> None:
    """尽力关闭原生子资源，清理失败不得覆盖原有解析异常。"""
    if callable(close := getattr(obj, "close", None)):
        try:
            close()
        except Exception:
            pass
