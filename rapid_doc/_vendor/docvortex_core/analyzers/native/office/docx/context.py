"""DOCX 转换共享的 XML 常量和字段栈契约。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, TypeAlias, Union

from pydantic import AnyUrl

from .....foundation.type_identity import preserve_type_module
from .formatting_types import Formatting

_ParagraphHyperlink: TypeAlias = Optional[Union[AnyUrl, Path, str]]
_ParagraphElement: TypeAlias = tuple[str, Optional[Formatting], _ParagraphHyperlink]


@dataclass(slots=True)
class _DocxComplexFieldFrame:
    """保存一个 DOCX 复杂字段的指令、阶段与已解析结果元素。"""

    instruction_parts: list[str] = field(default_factory=list)
    phase: str = "instr"
    result_elements: list[_ParagraphElement] = field(default_factory=list)


class _DocxConstants:
    """保存跨职责使用的固定命名空间与公式载体优先级。"""

    _BLIP_NAMESPACES: Final = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
        "o": "urn:schemas-microsoft-com:office:office",
        "v": "urn:schemas-microsoft-com:vml",
        "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
        "w10": "urn:schemas-microsoft-com:office:word",
        "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    }
    _PARAGRAPH_TRANSPARENT_INLINE_CONTAINERS: Final = {
        "bdo",
        "customXml",
        "dir",
        "fldSimple",
        "ins",
        "moveTo",
        "smartTag",
    }
    _FORMULA_TOKEN_KINDS: Final = frozenset({"omml", "equationxml", "mtef", "image_mtef"})
    _FORMULA_SOURCE_PRIORITY: Final = (
        "omml",
        "equationxml",
        "mtef",
        "image_mtef",
    )
    """
    Word 文档中使用的 XML 命名空间映射。

    这些命名空间用于解析 DOCX 文件中的各种元素，包括：
    - a: DrawingML 主命名空间
    - r: Office 文档关系命名空间
    - w: WordprocessingML 主命名空间
    - wp: Wordprocessing Drawing 命名空间
    - mc: 标记兼容性命名空间
    - v: VML (Vector Markup Language) 命名空间
    - wps: Wordprocessing Shape 命名空间
    - w10: Office Word 命名空间
    - a14: Office 2010 Drawing 命名空间
    """


preserve_type_module(_DocxComplexFieldFrame, "rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter")
