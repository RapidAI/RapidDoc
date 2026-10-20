from __future__ import annotations

import json
import math
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, TypeVar, Union, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from .foundation._hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target

# 这些字符串不能作为公开 Block.type discriminator，只用于 raw 阶段或 Block 内部枚举值。
RawBlockType: TypeAlias = Literal[
    "algorithm",
    "caption",
    "footnote",
    "formula_number",
    "phonetic",
]

RAW_ALGORITHM: RawBlockType = "algorithm"
RAW_CAPTION: RawBlockType = "caption"
RAW_FOOTNOTE: RawBlockType = "footnote"
RAW_FORMULA_NUMBER: RawBlockType = "formula_number"
RAW_PHONETIC: RawBlockType = "phonetic"

RAW_ONLY_BLOCK_TYPES = frozenset(
    {
        RAW_ALGORITHM,
        RAW_CAPTION,
        RAW_FOOTNOTE,
        RAW_FORMULA_NUMBER,
        RAW_PHONETIC,
    }
)

FileSuffix: TypeAlias = Literal[
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "xls",
    "xlsx",
    "rtf",
    "csv",
    "tsv",
    "epub",
    "html",
    "mhtml",
    "ofd",
    "odt",
    "ods",
    "odp",
]
FILE_SUFFIXES: frozenset[FileSuffix] = frozenset(cast(tuple[FileSuffix, ...], get_args(FileSuffix)))


class BlockType(str, Enum):
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    IMAGE_FOOTNOTE = "image_footnote"

    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"

    CHART = "chart"
    CHART_BODY = "chart_body"
    CHART_CAPTION = "chart_caption"
    CHART_FOOTNOTE = "chart_footnote"

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    ALGORITHM_BODY = "algorithm_body"
    CODE_CAPTION = "code_caption"
    CODE_FOOTNOTE = "code_footnote"

    TEXT = "text"
    EQUATION = "equation"  # 行间公式（独立公式）
    LIST = "list"
    INDEX = "index"

    # Added in vlm 2.5
    REF_TEXT = "ref_text"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"

    # Added in pp_doclayout_v2
    DOC_TITLE = "doc_title"
    PARAGRAPH_TITLE = "paragraph_title"

    def __str__(self) -> str:
        return self.value


BlockTypes = Literal[
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.ALGORITHM_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
]

PageBlockTypes = Literal[
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
]

BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.ALGORITHM_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}

PAGE_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}

# 页面装饰与辅助文本不参与正文、列表和视觉对象之间的语义边界判断。
PAGE_AUXILIARY_BLOCK_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
}
# 页面脚注需要参与输出，但不会阻断正文、列表、续表或视觉对象之间的关系判断。
MERGE_TRANSPARENT_BLOCK_TYPES = {
    *PAGE_AUXILIARY_BLOCK_TYPES,
    BlockType.PAGE_FOOTNOTE,
}
VISUAL_RELATION_IGNORED_TYPES = MERGE_TRANSPARENT_BLOCK_TYPES
VISUAL_MAIN_TYPES = {
    BlockType.IMAGE_BODY: BlockType.IMAGE,
    BlockType.TABLE_BODY: BlockType.TABLE,
    BlockType.CHART_BODY: BlockType.CHART,
    BlockType.CODE_BODY: BlockType.CODE,
}
VISUAL_TYPE_MAPPING = {
    BlockType.IMAGE: {
        "body": BlockType.IMAGE_BODY,
        "caption": BlockType.IMAGE_CAPTION,
        "footnote": BlockType.IMAGE_FOOTNOTE,
    },
    BlockType.TABLE: {
        "body": BlockType.TABLE_BODY,
        "caption": BlockType.TABLE_CAPTION,
        "footnote": BlockType.TABLE_FOOTNOTE,
    },
    BlockType.CHART: {
        "body": BlockType.CHART_BODY,
        "caption": BlockType.CHART_CAPTION,
        "footnote": BlockType.CHART_FOOTNOTE,
    },
    BlockType.CODE: {
        "body": BlockType.CODE_BODY,
        "caption": BlockType.CODE_CAPTION,
        "footnote": BlockType.CODE_FOOTNOTE,
    },
}
# ── model types ─────────────────────────────────────────────────────

BBox: TypeAlias = tuple[float, float, float, float]
IntBBox: TypeAlias = tuple[int, int, int, int]


def _remove_block_fields(value: Any, excluded_fields: set[str]) -> Any:
    """递归删除序列化结果中指定的 block 字段，覆盖任意深度的容器。"""
    if isinstance(value, list):
        return [_remove_block_fields(item, excluded_fields) for item in value]
    if not isinstance(value, dict):
        return value

    result = {
        key: _remove_block_fields(item, excluded_fields)
        for key, item in value.items()
        if not ("type" in value and key in excluded_fields)
    }
    return result


class _StrictMiddleModel(BaseModel):
    """Model/Middle JSON 严格模型基类，提供无副作用的统一序列化入口。"""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    def to_dict(
        self,
        *,
        skip_defaults: bool = True,
        exclude_none: bool = False,
        exclude_block_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """序列化对象，并按字段名递归排除任意层级的 block 字段。"""
        payload = self.model_dump(
            mode="json",
            exclude_defaults=skip_defaults,
            exclude_none=exclude_none,
        )
        if exclude_block_fields:
            payload = _remove_block_fields(payload, set(exclude_block_fields))
        return payload

    def to_json(
        self,
        *,
        skip_defaults: bool = True,
        exclude_none: bool = False,
        exclude_block_fields: set[str] | None = None,
        indent: int | None = 4,
    ) -> str:
        """将对象编码为 UTF-8 友好的 JSON 字符串，不执行图片文件写入。"""
        return json.dumps(
            self.to_dict(
                skip_defaults=skip_defaults,
                exclude_none=exclude_none,
                exclude_block_fields=exclude_block_fields,
            ),
            ensure_ascii=False,
            indent=indent,
        )


InlineStyle: TypeAlias = Literal[
    "bold",
    "italic",
    "underline",
    "emphasis",
    "strikethrough",
    "superscript",
    "subscript",
]

INLINE_STYLE_ORDER: tuple[InlineStyle, ...] = (
    "bold",
    "italic",
    "underline",
    "emphasis",
    "strikethrough",
    "superscript",
    "subscript",
)


class TextSpan(_StrictMiddleModel):
    """保存普通行内文字及其可见字体样式。"""

    type: Literal["text"]
    content: str = Field(min_length=1)
    styles: list[InlineStyle] = Field(default_factory=list)

    @field_validator("styles")
    @classmethod
    def _normalize_styles(cls, value: list[InlineStyle]) -> list[InlineStyle]:
        """按公开固定顺序去重样式，并禁止同时声明上下标。"""
        unique = set(value)
        if "superscript" in unique and "subscript" in unique:
            raise ValueError("text span cannot be both superscript and subscript")
        return [style for style in INLINE_STYLE_ORDER if style in unique]


class EquationInlineSpan(_StrictMiddleModel):
    """保存不含外层定界符的行内 LaTeX。"""

    type: Literal["equation_inline"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        """拒绝只包含空白的行内公式，同时保留公式原始空白。"""
        if not value.strip():
            raise ValueError("inline equation content must not be blank")
        return value


class CodeInlineSpan(_StrictMiddleModel):
    """保存需要按字面量显示的行内代码。"""

    type: Literal["code_inline"]
    content: str = Field(min_length=1)


NonLinkInlineSpan: TypeAlias = Annotated[
    Union[TextSpan, EquationInlineSpan, CodeInlineSpan],
    Field(discriminator="type"),
]


class HyperlinkSpan(_StrictMiddleModel):
    """保存安全超链接目标及其非链接行内子节点。"""

    type: Literal["hyperlink"]
    url: str = Field(min_length=1)
    content: list[NonLinkInlineSpan] = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """复用统一策略拒绝危险协议、本地路径、畸形 URL 和控制字符。"""
        normalized = sanitize_hyperlink_target(
            value,
            allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
        )
        if normalized is None:
            raise ValueError("hyperlink span url is unsafe or malformed")
        return normalized


InlineSpan: TypeAlias = Annotated[
    Union[TextSpan, EquationInlineSpan, CodeInlineSpan, HyperlinkSpan],
    Field(discriminator="type"),
]

INLINE_SPAN_ADAPTER = TypeAdapter(InlineSpan)
INLINE_SPAN_LIST_ADAPTER = TypeAdapter(list[InlineSpan])


def _normalize_typed_inline_spans(spans: list[InlineSpan]) -> list[InlineSpan]:
    """递归合并相邻同样式文字及相邻同目标链接。"""
    normalized: list[InlineSpan] = []
    for span in spans:
        current: InlineSpan
        if isinstance(span, HyperlinkSpan):
            children = _normalize_typed_inline_spans(list(span.content))
            non_link_children = [child for child in children if not isinstance(child, HyperlinkSpan)]
            if not non_link_children:
                continue
            current = span.model_copy(update={"content": non_link_children}, deep=True)
        else:
            current = span.model_copy(deep=True)
        if (
            normalized
            and isinstance(normalized[-1], TextSpan)
            and isinstance(current, TextSpan)
            and normalized[-1].styles == current.styles
        ):
            previous = normalized[-1]
            normalized[-1] = previous.model_copy(update={"content": f"{previous.content}{current.content}"})
            continue
        if (
            normalized
            and isinstance(normalized[-1], HyperlinkSpan)
            and isinstance(current, HyperlinkSpan)
            and normalized[-1].url == current.url
        ):
            previous_link = normalized[-1]
            merged_children = _normalize_typed_inline_spans([*previous_link.content, *current.content])
            normalized[-1] = previous_link.model_copy(update={"content": merged_children})
            continue
        normalized.append(current)
    return normalized


def parse_inline_span(value: Any) -> InlineSpan:
    """把字典或现有模型严格解析为一个公开行内 Span。"""
    return INLINE_SPAN_ADAPTER.validate_python(value)


def parse_inline_spans(value: Any) -> list[InlineSpan]:
    """严格解析并规范化完整行内 Span 列表。"""
    return _normalize_typed_inline_spans(INLINE_SPAN_LIST_ADAPTER.validate_python(value))


class BlockBase(_StrictMiddleModel):
    """所有公开 Middle JSON block 的最小公共字段。"""

    type: BlockTypes
    index: int | None = Field(default=None, ge=0)
    bbox: BBox | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, value: Any) -> BBox | None:
        """接受 JSON 数组形式的 bbox，并严格校验归一化坐标。"""
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four numbers")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("bbox values must be numbers")
        bbox = tuple(float(item) for item in value)
        if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in bbox):
            raise ValueError("bbox values must be finite normalized coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox must satisfy x1 > x0 and y1 > y0")
        return bbox  # type: ignore[return-value]


class StringContentBlock(BlockBase):
    """所有字符串内容 block 的共享结构。"""

    content: str


class InlineContentBlock(BlockBase):
    """所有结构化行内内容 block 的共享结构。"""

    content: list[InlineSpan]

    @field_validator("content")
    @classmethod
    def _normalize_content(cls, value: list[InlineSpan]) -> list[InlineSpan]:
        """在严格对象边界合并相邻同语义 Span。"""
        return _normalize_typed_inline_spans(value)


class ContinuableTextBlockBase(InlineContentBlock):
    """顶层正文或参考续文；正文可接前序正文列表的最后一个直属 text，物理块不移动。"""

    continues_prev: bool | None = None


class TextBlock(ContinuableTextBlockBase):
    type: Literal[BlockType.TEXT]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class RefTextBlock(ContinuableTextBlockBase):
    type: Literal[BlockType.REF_TEXT]  # type: ignore[reportIncompatibleVariableOverride]


class TitleBlockBase(InlineContentBlock):
    """文档标题与段落标题的全局层级公共结构。"""

    anchor: str | None = None
    level: int


class DocTitleBlock(TitleBlockBase):
    type: Literal[BlockType.DOC_TITLE]  # type: ignore[reportIncompatibleVariableOverride]
    level: int = Field(ge=1, le=1)


class ParagraphTitleBlock(TitleBlockBase):
    type: Literal[BlockType.PARAGRAPH_TITLE]  # type: ignore[reportIncompatibleVariableOverride]
    level: int = Field(ge=2, le=6)


class PageAuxTextBlock(InlineContentBlock):
    """页眉、页脚、页码和边栏的共享文本结构。"""

    type: Literal[  # type: ignore[reportIncompatibleVariableOverride]
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
    ]


class PageFootnoteBlock(InlineContentBlock):
    """保存需要参与默认输出并可被文档内链接引用的页面脚注。"""

    type: Literal[BlockType.PAGE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class ImagePayloadBlock(BlockBase):
    """统一携带 sidecar、data URI 或远程 URL 的图片 block 基类。"""

    image_base64: str | None = Field(default=None, repr=False)
    image_path: str | None = None
    image_url: str | None = None

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str | None) -> str | None:
        """校验已记录的图片路径只能是安全的 POSIX 相对路径。"""
        if value is None:
            return None
        from .foundation._image_payload import validate_image_sidecar_path

        return validate_image_sidecar_path(value)

    @field_validator("image_url")
    @classmethod
    def _validate_image_url(cls, value: str | None) -> str | None:
        """校验远程图片 URL，禁止活动协议、相对地址与内嵌凭据。"""
        if value is None:
            return None
        from .foundation._image_payload import validate_remote_image_url

        return validate_remote_image_url(value)


class ImagePayloadContentBlock(ImagePayloadBlock):
    """统一携带字符串内容和图片载荷的 block 结构。"""

    content: str


class EquationBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.EQUATION]  # type: ignore[reportIncompatibleVariableOverride]


class ImageBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.IMAGE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class TableBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.TABLE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class ChartBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.CHART_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class CodeBodyBlock(StringContentBlock):
    type: Literal[BlockType.CODE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class AlgorithmBodyBlock(InlineContentBlock):
    """保存预格式算法文字与行内公式 Span。"""

    type: Literal[BlockType.ALGORITHM_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class ImageAnnotationBlock(InlineContentBlock):
    """图片标题与图片脚注的共享结构。"""

    type: Literal[BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class TableAnnotationBlock(InlineContentBlock):
    """表格标题与表格脚注的共享结构。"""

    type: Literal[BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class ChartAnnotationBlock(InlineContentBlock):
    """图表标题与图表脚注的共享结构。"""

    type: Literal[BlockType.CHART_CAPTION, BlockType.CHART_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class CodeAnnotationBlock(InlineContentBlock):
    """代码标题与代码脚注的共享结构。"""

    type: Literal[BlockType.CODE_CAPTION, BlockType.CODE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


ListChildBlock: TypeAlias = Annotated[
    Union[TextBlock, RefTextBlock, "ListBlock"],
    Field(discriminator="type"),
]


class ListBlock(BlockBase):
    type: Literal[BlockType.LIST]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ListChildBlock]
    sub_type: Literal[BlockType.TEXT, BlockType.REF_TEXT] | None = None
    continues_prev: bool | None = None


IndexChildBlock: TypeAlias = Annotated[
    Union[TextBlock, DocTitleBlock, ParagraphTitleBlock, "IndexBlock"],
    Field(discriminator="type"),
]


class IndexBlock(BlockBase):
    type: Literal[BlockType.INDEX]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[IndexChildBlock]


class _VisualBlockBase(BlockBase):
    """视觉父块的共享结构约束。"""

    _body_types: ClassVar[tuple[str, ...]]

    @model_validator(mode="after")
    def _validate_visual_children(self) -> _VisualBlockBase:
        """校验视觉父块只有一个 body，且父子定位字段保持一致。"""
        children = getattr(self, "content", [])
        bodies = [child for child in children if child.type in self._body_types]
        if len(bodies) != 1:
            expected = "/".join(str(item) for item in self._body_types)
            raise ValueError(f"{self.type} must contain exactly one {expected}")
        body = bodies[0]
        if self.index is not None and body.index != self.index:
            raise ValueError(f"{self.type} body index must equal parent index")
        if self.bbox is not None and body.bbox is not None and body.bbox != self.bbox:
            raise ValueError(f"{self.type} body bbox must equal parent bbox")
        return self


ImageChildBlock: TypeAlias = Annotated[
    Union[ImageBodyBlock, ImageAnnotationBlock],
    Field(discriminator="type"),
]


class ImageBlock(_VisualBlockBase):
    type: Literal[BlockType.IMAGE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ImageChildBlock]
    sub_type: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.IMAGE_BODY,)


TableChildBlock: TypeAlias = Annotated[
    Union[TableBodyBlock, TableAnnotationBlock],
    Field(discriminator="type"),
]


class TableBlock(_VisualBlockBase):
    type: Literal[BlockType.TABLE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[TableChildBlock]
    continues_prev: bool | None = None
    cell_merge: list[Literal[0, 1]] | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.TABLE_BODY,)


ChartChildBlock: TypeAlias = Annotated[
    Union[ChartBodyBlock, ChartAnnotationBlock],
    Field(discriminator="type"),
]


class ChartBlock(_VisualBlockBase):
    type: Literal[BlockType.CHART]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ChartChildBlock]
    sub_type: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.CHART_BODY,)


CodeChildBlock: TypeAlias = Annotated[
    Union[CodeBodyBlock, AlgorithmBodyBlock, CodeAnnotationBlock],
    Field(discriminator="type"),
]


class CodeBlock(_VisualBlockBase):
    type: Literal[BlockType.CODE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[CodeChildBlock]
    sub_type: Literal[BlockType.CODE, RAW_ALGORITHM]
    guess_lang: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.CODE_BODY, BlockType.ALGORITHM_BODY)

    @model_validator(mode="after")
    def _validate_language(self) -> CodeBlock:
        """代码块要求语言，算法块则禁止携带代码语言猜测结果。"""
        body = next(child for child in self.content if child.type in self._body_types)
        if self.sub_type == BlockType.CODE:
            if body.type != BlockType.CODE_BODY:
                raise ValueError("code block must contain code_body")
            if not isinstance(self.guess_lang, str) or not self.guess_lang.strip():
                raise ValueError("code block must contain a non-empty guess_lang")
        else:
            if body.type != BlockType.ALGORITHM_BODY:
                raise ValueError("algorithm block must contain algorithm_body")
            if self.guess_lang is not None:
                raise ValueError("algorithm block must not contain guess_lang")
        return self


ListBlock.model_rebuild()
IndexBlock.model_rebuild()


PageBlock: TypeAlias = Annotated[
    Union[
        TextBlock,
        RefTextBlock,
        DocTitleBlock,
        ParagraphTitleBlock,
        PageAuxTextBlock,
        PageFootnoteBlock,
        EquationBlock,
        ListBlock,
        IndexBlock,
        ImageBlock,
        TableBlock,
        ChartBlock,
        CodeBlock,
    ],
    Field(discriminator="type"),
]


Block: TypeAlias = Annotated[
    Union[
        TextBlock,
        RefTextBlock,
        DocTitleBlock,
        ParagraphTitleBlock,
        PageAuxTextBlock,
        PageFootnoteBlock,
        EquationBlock,
        ListBlock,
        IndexBlock,
        ImageBodyBlock,
        ImageAnnotationBlock,
        ImageBlock,
        TableBodyBlock,
        TableAnnotationBlock,
        TableBlock,
        ChartBodyBlock,
        ChartAnnotationBlock,
        ChartBlock,
        CodeBodyBlock,
        AlgorithmBodyBlock,
        CodeAnnotationBlock,
        CodeBlock,
    ],
    Field(discriminator="type"),
]

BLOCK_ADAPTER = TypeAdapter(Block)


def parse_block(value: Any) -> Block:
    """将字典或已有模型严格解析成对应的具体 Block 类型。"""
    return BLOCK_ADAPTER.validate_python(value)


def _iter_child_blocks(block: BlockBase) -> list[BlockBase]:
    """返回容器 block 的直接子块，叶子 block 返回空列表。"""
    content = getattr(block, "content", None)
    if not isinstance(content, list):
        return []
    return [child for child in content if isinstance(child, BlockBase)]


_RAW_INLINE_CONTENT_TYPES = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.IMAGE_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_CAPTION,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    RAW_ALGORITHM,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    RAW_PHONETIC,
}


def _looks_like_raw_inline_span_list(content: list[Any]) -> bool:
    """区分 PDF 扁平 LIST/INDEX 的 Span 载荷与已经成树的文本子块。"""
    if not content:
        return False
    for item in content:
        if not isinstance(item, dict):
            return False
        span_type = item.get("type")
        span_content = item.get("content")
        if span_type in {"text", "equation_inline", "code_inline"} and isinstance(span_content, str):
            continue
        if span_type == "hyperlink" and isinstance(span_content, list) and isinstance(item.get("url"), str):
            continue
        return False
    return True


def _validate_raw_block_inline_content(block: dict[str, Any], *, location: str) -> None:
    """递归校验 raw block 的自然语言 content 已切换为 Span 列表。"""
    block_type = block.get("type")
    content = block.get("content")
    if block_type in _RAW_INLINE_CONTENT_TYPES:
        if not isinstance(content, list):
            raise ValueError(f"ModelJson inline content must be a span list: {location}, type={block_type}")
        try:
            parse_inline_spans(content)
        except ValueError as exc:
            raise ValueError(f"Invalid ModelJson inline spans: {location}, type={block_type}: {exc}") from exc
        return
    if block_type not in {BlockType.LIST, BlockType.INDEX} or not isinstance(content, list):
        return
    if _looks_like_raw_inline_span_list(content):
        try:
            parse_inline_spans(content)
        except ValueError as exc:
            raise ValueError(f"Invalid ModelJson inline spans: {location}, type={block_type}: {exc}") from exc
        return
    for child_index, child in enumerate(content):
        if isinstance(child, dict):
            _validate_raw_block_inline_content(child, location=f"{location}.content[{child_index}]")


class Producer(_StrictMiddleModel):
    """记录语言无关的文档生产者，避免绑定宿主产品元数据。"""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class DocumentProperties(_StrictMiddleModel):
    """保存源文件声明的属性；计数属于完整源文件而非当前解析选页。"""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    subject: str | None = None
    keywords: list[str] = Field(default_factory=list)
    description: str | None = None
    languages: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    publisher: str | None = None
    created_at: str | None = None
    modified_at: str | None = None
    creator_application: str | None = None
    producer_application: str | None = None
    page_count: int | None = Field(default=None, ge=0)
    page_count_kind: Literal["physical", "declared", "slide", "sheet", "spine", "logical"] | None = None

    @field_validator("authors", "keywords", "languages", "identifiers")
    @classmethod
    def _normalize_values(cls, values: list[str]) -> list[str]:
        """去除空白项并稳定去重，不猜测单个字符串中的分隔符。"""
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class DocumentMetadata(_StrictMiddleModel):
    """承载文档格式和真实生产者，读取时不补造来源信息。"""

    file_suffix: FileSuffix
    producer: Producer
    document: DocumentProperties | None = None


def _require_document_wire_identity(schema: dict[str, Any]) -> None:
    """JSON 文档必须显式携带协议身份；构造器的常量默认值仅便利 Python 调用。"""
    identity = "schema" if "schema" in schema["properties"] else "schema_id"
    schema["required"] = list(dict.fromkeys([identity, "schema_version", *schema.get("required", [])]))


_DocumentT = TypeVar("_DocumentT", bound="DocumentModel")


class DocumentModel(_StrictMiddleModel):
    """公共文档封装，只持有生产者及可序列化扩展信息。"""

    model_config = ConfigDict(serialize_by_alias=True, json_schema_extra=_require_document_wire_identity)

    metadata: DocumentMetadata
    extensions: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: Literal["2.0"] = "2.0"
    schema_id: str = Field(alias="schema")

    def to_dict(
        self,
        *,
        skip_defaults: bool = True,
        exclude_none: bool = False,
        exclude_block_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """只对页面树省略块字段，保护具有同名键的来源和应用扩展。"""
        payload = super().to_dict(skip_defaults=skip_defaults, exclude_none=exclude_none)
        if exclude_block_fields and "pages" in payload:
            payload["pages"] = _remove_block_fields(payload["pages"], exclude_block_fields)
        return payload

    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        """保留已声明的协议默认字段，同时尊重调用方显式的字段筛选。"""
        # 不声明通用 dict 返回类型，避免 Pydantic 将序列化 Schema 降为任意对象。
        payload = handler(self)
        use_alias = info.by_alias is not False
        for field_name in ("schema_id", "schema_version", "extensions"):
            if info.exclude and field_name in info.exclude:
                continue
            if info.include is not None and field_name not in info.include:
                continue
            key = "schema" if field_name == "schema_id" and use_alias else field_name
            if key not in payload:
                payload[key] = deepcopy(getattr(self, field_name))
        return payload

    @classmethod
    def from_dict(cls: type[_DocumentT], value: dict[str, Any]) -> _DocumentT:
        """联合校验协议身份及版本后读取文档，不猜测或迁移历史格式。"""
        expected_schema = cls.model_fields["schema_id"].default
        expected_version = cls.model_fields["schema_version"].default
        if (
            not isinstance(value, dict)
            or value.get("schema") != expected_schema
            or value.get("schema_version") != expected_version
        ):
            raise ValueError(f"Expected {expected_schema} schema version {expected_version}; reparse the source document")
        return cls.model_validate(value)

    @classmethod
    def from_json(cls: type[_DocumentT], value: str | bytes) -> _DocumentT:
        """从 JSON 文本恢复共享文档，并复用唯一的协议校验入口。"""
        return cls.from_dict(json.loads(value))


class ModelJson(DocumentModel):
    """Analyze 返回的完整严格 Model JSON 对象。"""

    schema_id: Literal["rapid_doc._vendor.docvortex_core.model"] = Field(default="rapid_doc._vendor.docvortex_core.model", alias="schema")
    pages: list[list[dict[str, Any]]]
    page_index_map: list[int]

    @model_validator(mode="after")
    def _validate_page_index_map(self) -> ModelJson:
        """校验显式抽页映射及每个 raw 文本块的 Span 契约。"""
        if self.page_index_map:
            if len(self.page_index_map) != len(self.pages):
                raise ValueError(f"page_index_map length mismatch: pages={len(self.pages)}, mapping={len(self.page_index_map)}")
            if any(page_idx < 0 for page_idx in self.page_index_map):
                raise ValueError("page_index_map values must be non-negative integers")
            if len(self.page_index_map) != len(set(self.page_index_map)):
                raise ValueError("page_index_map values must be unique")
            if any(current <= previous for previous, current in zip(self.page_index_map, self.page_index_map[1:])):
                raise ValueError("page_index_map values must preserve strictly increasing order")
        for page_index, page in enumerate(self.pages):
            for block_index, block in enumerate(page):
                if isinstance(block, dict):
                    _validate_raw_block_inline_content(block, location=f"pages[{page_index}][{block_index}]")
        return self

    @property
    def is_full_document(self) -> bool:
        """返回当前 Model JSON 是否表示整本文档解析。"""
        return not self.page_index_map

    @property
    def resolved_page_indices(self) -> list[int]:
        """返回显式抽页映射或整本文档的顺序页号副本。"""
        if self.is_full_document:
            return list(range(len(self.pages)))
        return list(self.page_index_map)


class PageInfo(_StrictMiddleModel):
    """一页的严格 Middle JSON 内容。"""

    page_idx: int = Field(ge=0)
    blocks: list[PageBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_page_tree(self) -> PageInfo:
        """校验顶层 index 顺序，并禁止嵌套块携带跨块延续标记。"""
        indices: list[int] = []
        for block in self.blocks:
            if block.index is None:
                raise ValueError("top-level block index is required")
            indices.append(block.index)
        if len(indices) != len(set(indices)):
            raise ValueError("top-level block indices must be unique")
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("top-level block indices must be strictly increasing")

        pending = [child for block in self.blocks for child in _iter_child_blocks(block)]
        while pending:
            child = pending.pop()
            if "continues_prev" in child.model_fields_set:
                raise ValueError("nested blocks must not contain continues_prev")
            pending.extend(_iter_child_blocks(child))
        return self


class MiddleJson(DocumentModel):
    """Analyze 返回的完整严格 Middle JSON 对象。"""

    schema_id: Literal["rapid_doc._vendor.docvortex_core.middle"] = Field(default="rapid_doc._vendor.docvortex_core.middle", alias="schema")
    pages: list[PageInfo]
    is_full_document: bool

    @model_validator(mode="after")
    def _validate_document(self) -> MiddleJson:
        """校验页号唯一有序，并要求固定版式文档顶层 block 均具有 bbox。"""
        page_indices = [page.page_idx for page in self.pages]
        if len(page_indices) != len(set(page_indices)):
            raise ValueError("page_idx values must be unique")
        if any(current <= previous for previous, current in zip(page_indices, page_indices[1:])):
            raise ValueError("page_idx values must be strictly increasing")
        if self.metadata.file_suffix in {"pdf", "ofd"}:
            for page in self.pages:
                for block in page.blocks:
                    if block.bbox is None:
                        raise ValueError(
                            f"Fixed-layout top-level block requires bbox: "
                            f"file_suffix={self.metadata.file_suffix}, page_idx={page.page_idx}, index={block.index}"
                        )
        return self


__all__ = [
    "DocumentMetadata",
    "DocumentProperties",
    "RawBlockType",
    "RAW_ALGORITHM",
    "RAW_CAPTION",
    "RAW_FOOTNOTE",
    "RAW_FORMULA_NUMBER",
    "RAW_PHONETIC",
    "RAW_ONLY_BLOCK_TYPES",
    "FileSuffix",
    "FILE_SUFFIXES",
    "BlockType",
    "BlockTypes",
    "PageBlockTypes",
    "BLOCK_TYPES",
    "PAGE_BLOCK_TYPES",
    "PAGE_AUXILIARY_BLOCK_TYPES",
    "MERGE_TRANSPARENT_BLOCK_TYPES",
    "VISUAL_RELATION_IGNORED_TYPES",
    "VISUAL_MAIN_TYPES",
    "VISUAL_TYPE_MAPPING",
    "BBox",
    "IntBBox",
    "InlineStyle",
    "INLINE_STYLE_ORDER",
    "TextSpan",
    "EquationInlineSpan",
    "CodeInlineSpan",
    "NonLinkInlineSpan",
    "HyperlinkSpan",
    "InlineSpan",
    "INLINE_SPAN_ADAPTER",
    "INLINE_SPAN_LIST_ADAPTER",
    "parse_inline_span",
    "parse_inline_spans",
    "BlockBase",
    "StringContentBlock",
    "InlineContentBlock",
    "ContinuableTextBlockBase",
    "TextBlock",
    "RefTextBlock",
    "TitleBlockBase",
    "DocTitleBlock",
    "ParagraphTitleBlock",
    "PageAuxTextBlock",
    "PageFootnoteBlock",
    "ImagePayloadBlock",
    "ImagePayloadContentBlock",
    "EquationBlock",
    "ImageBodyBlock",
    "TableBodyBlock",
    "ChartBodyBlock",
    "CodeBodyBlock",
    "AlgorithmBodyBlock",
    "ImageAnnotationBlock",
    "TableAnnotationBlock",
    "ChartAnnotationBlock",
    "CodeAnnotationBlock",
    "ListChildBlock",
    "ListBlock",
    "IndexChildBlock",
    "IndexBlock",
    "ImageChildBlock",
    "ImageBlock",
    "TableChildBlock",
    "TableBlock",
    "ChartChildBlock",
    "ChartBlock",
    "CodeChildBlock",
    "CodeBlock",
    "PageBlock",
    "Block",
    "BLOCK_ADAPTER",
    "parse_block",
    "Producer",
    "DocumentModel",
    "ModelJson",
    "PageInfo",
    "MiddleJson",
]
