"""原生分析的内部结构契约；公开 ModelJson 的 wire shape 保持不变。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, TypedDict

from ...schema import BBox

if TYPE_CHECKING:
    from ...document.pdf._document import _PDFPageSnapshot


class RawBlock(TypedDict, total=False):
    """描述 raw 阶段共有字段，格式特有证据仍可附加在普通字典中。"""

    type: str
    bbox: BBox | list[float] | None
    index: int
    content: str | list[dict[str, Any]]
    angle: int
    image_base64: str
    lines: list[dict[str, list[float]]]
    _inline_math_regions: list[BBox]


class NativeBinaryAnalyzer(Protocol):
    """限定 API 分派需要的二进制流预测能力，不创建统一继承框架。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """读取调用者持有的流并返回原生页面，不关闭该流。"""


class NativePdfSource(Protocol):
    """限定原生 PDF 编排所需的页面数量与一次性证据快照。"""

    @property
    def page_count(self) -> int:
        """返回当前所选 PDF 的物理页数。"""

    def _extract_native_page(self, page_idx: int) -> _PDFPageSnapshot:
        """在单个页面生命周期内收集原生证据。"""


__all__ = ["RawBlock", "NativeBinaryAnalyzer", "NativePdfSource"]
