from io import BytesIO
from typing import BinaryIO, Optional

from loguru import logger
from pptx import Presentation, presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from ..equation.image import OfficeImageEquationDecoder
from ..equation.ooxml import OoxmlEquationDecoder
from ..errors import LegacyOfficeResourceLimitError
from ..streams import read_stream_bytes_from_start, rewind_stream
from .package_normalizer import normalize_pptx_package
from ..._shared.xycut import sort_entries
from .....schema import BlockType

# PPTX_XYCUT_BETA: Final = 0.7
# PPTX 标题占位符角色仅用于单页内部归一化，返回 model_output 前必须清理。


from .context import (
    IGNORED_NOTES_PLACEHOLDER_TYPES as IGNORED_NOTES_PLACEHOLDER_TYPES,
    MIN_PICTURE_DIMENSION_RATIO as MIN_PICTURE_DIMENSION_RATIO,
    MIN_PICTURE_AREA_RATIO as MIN_PICTURE_AREA_RATIO,
    BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO as BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO,
    PPTX_XYCUT_BETA as PPTX_XYCUT_BETA,
    PPTX_XYCUT_DENSITY_THRESHOLD as PPTX_XYCUT_DENSITY_THRESHOLD,
    DRAWINGML_NS as DRAWINGML_NS,
    RELATIONSHIP_NS as RELATIONSHIP_NS,
    SVG_BLIP_NS as SVG_BLIP_NS,
    A14_DRAWING_NS as A14_DRAWING_NS,
    OMML_NS as OMML_NS,
    _EFFECTIVE_FONT_SIZE_KEY as _EFFECTIVE_FONT_SIZE_KEY,
    _EFFECTIVE_ALL_BOLD_KEY as _EFFECTIVE_ALL_BOLD_KEY,
    _PPTX_TITLE_CANDIDATE_KEY as _PPTX_TITLE_CANDIDATE_KEY,
    _PPTX_TITLE_ROLE_KEY as _PPTX_TITLE_ROLE_KEY,
    _PPTX_TITLE_ROLE_CENTER as _PPTX_TITLE_ROLE_CENTER,
    _PPTX_TITLE_ROLE_TITLE as _PPTX_TITLE_ROLE_TITLE,
    _PPTX_TITLE_ROLE_SUBTITLE as _PPTX_TITLE_ROLE_SUBTITLE,
    _PPTX_TITLE_PLACEHOLDER_ROLES as _PPTX_TITLE_PLACEHOLDER_ROLES,
    _SlideTransform as _SlideTransform,
    _FlattenedShape as _FlattenedShape,
)
from .resources import _PptxResources
from .shapes import _PptxShapes
from .text_styles import _PptxTextStyles
from .lists import _PptxLists
from .titles import _PptxTitles

# python-pptx 默认模板幻灯片尺寸（10 × 7.5 英寸，EMU），用于缺 p:sldSz 的包。
DEFAULT_SLIDE_WIDTH_EMU = 9144000
DEFAULT_SLIDE_HEIGHT_EMU = 6858000


class PptxConverter(_PptxResources, _PptxShapes, _PptxTextStyles, _PptxLists, _PptxTitles):
    """编排 PPTX 包读取、逐页遍历、重试及职责处理。"""

    def __init__(self):
        """配置固定命名空间并为当前转换创建独立状态。"""
        self.namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
            "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        self._reset_state()

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        """解析 PPTX；资源异常直接传播，包兼容异常才进入规范化重试。"""
        if rewind_stream(file_stream):
            try:
                self._convert_package_stream(file_stream)
                return
            except (LegacyOfficeResourceLimitError, MemoryError):
                raise
            except Exception as exc:
                file_bytes = read_stream_bytes_from_start(file_stream)
                self._retry_convert_package_bytes_after_normalization(file_bytes, exc)
                return

        file_bytes = file_stream.read()
        try:
            self._convert_package_bytes(file_bytes)
        except (LegacyOfficeResourceLimitError, MemoryError):
            raise
        except Exception as exc:
            self._retry_convert_package_bytes_after_normalization(file_bytes, exc)

    def _reset_state(self) -> None:
        """重置解析状态，确保失败重试时不会残留上一次半解析结果。"""
        self.pages = []
        self.cur_page = []
        self.list_block_stack = []
        self._shape_type_cache = {}
        self.file_stream = None
        self.pptx_obj = None
        self._ooxml_equation_decoder = OoxmlEquationDecoder()
        self._image_equation_decoder = OfficeImageEquationDecoder()
        self._mtef_warned_shapes = set()

    def _convert_package_bytes(self, file_bytes: bytes) -> None:
        """用独立字节流解析 PPTX 包，便于原始包失败后用规范化包重试。"""
        self._convert_package_stream(BytesIO(file_bytes))

    def _convert_package_stream(self, file_stream: BinaryIO) -> None:
        """直接使用可复位的 PPTX 流解析正常路径，避免提前复制完整包字节。"""
        self._reset_state()
        rewind_stream(file_stream)
        self.file_stream = file_stream
        self.pptx_obj = Presentation(self.file_stream)
        self.pages.append(self.cur_page)
        if self.pptx_obj:
            self._walk_linear(self.pptx_obj)
        if self.pages and self.pages[-1] == []:
            self.pages.pop()

    def _retry_convert_package_bytes_after_normalization(
        self,
        file_bytes: bytes,
        exc: Exception,
    ) -> None:
        """首次解析失败后，仅在包规范化确实产生变化时使用规范化字节重试。"""
        normalized_bytes = normalize_pptx_package(file_bytes)
        if normalized_bytes == file_bytes:
            raise exc
        logger.warning(f"Retrying PPTX parsing after package normalization: {exc}")
        self._convert_package_bytes(normalized_bytes)

    def _walk_linear(self, pptx_obj: presentation.Presentation):
        # 缺 p:sldSz 的非标准包 slide_width/height 为 None，按 python-pptx 默认模板尺寸回退。
        slide_width = int(pptx_obj.slide_width or DEFAULT_SLIDE_WIDTH_EMU)
        slide_height = int(pptx_obj.slide_height or DEFAULT_SLIDE_HEIGHT_EMU)
        has_visible_content_slide = False

        # 遍历每一张幻灯片
        for _, slide in enumerate(pptx_obj.slides):
            linear_shapes = self._flatten_slide_shapes(slide.shapes)
            sortable_shape_entries = []
            tail_blocks = []

            # 遍历幻灯片中的每一个形状
            for shape_index, shape_entry in enumerate(linear_shapes):
                shape_blocks = self._collect_shape_blocks(
                    shape_entry,
                    linear_shapes,
                    shape_index,
                    slide_width,
                    slide_height,
                )
                if not shape_blocks:
                    continue

                if shape_entry.bbox is None:
                    tail_blocks.extend(shape_blocks)
                    continue

                sortable_shape_entries.append(
                    {
                        "bbox": shape_entry.bbox,
                        "blocks": shape_blocks,
                    }
                )

            sorted_shape_entries = sort_entries(
                sortable_shape_entries,
                beta=PPTX_XYCUT_BETA,
                density_threshold=PPTX_XYCUT_DENSITY_THRESHOLD,
            )
            for entry in sorted_shape_entries:
                self.cur_page.extend(entry["blocks"])
            self.cur_page.extend(tail_blocks)

            visible_slide_has_content = bool(self.cur_page)
            self._handle_slide_notes(slide)
            self._promote_slide_text_blocks_to_titles(self.cur_page)
            self._finalize_slide_title_types(
                self.cur_page,
                is_first_visible_slide=visible_slide_has_content and not has_visible_content_slide,
            )
            self._cleanup_slide_text_block_metadata(self.cur_page)
            if visible_slide_has_content:
                has_visible_content_slide = True
            self.cur_page = []
            self.pages.append(self.cur_page)

    def _handle_slide_notes(self, slide) -> None:
        if not slide.has_notes_slide:
            return

        try:
            notes_slide = slide.notes_slide
        except Exception as e:
            logger.warning(f"Warning: notes slide cannot be loaded: {e}")
            return

        def handle_notes_shape(shape) -> None:
            shape_type = self._safe_shape_type(shape)
            if shape_type == MSO_SHAPE_TYPE.GROUP:
                for grouped_shape in shape.shapes:
                    handle_notes_shape(grouped_shape)
                return

            if shape_type in {
                MSO_SHAPE_TYPE.EMBEDDED_OLE_OBJECT,
                MSO_SHAPE_TYPE.LINKED_OLE_OBJECT,
            }:
                if self._is_equation_ole_shape(shape):
                    omml_equations = self._pptx_ole_shape_omml(shape)
                    if omml_equations:
                        self.cur_page.extend(
                            {
                                "type": BlockType.PAGE_FOOTNOTE,
                                "content": [{"type": "equation_inline", "content": latex}],
                            }
                            for latex in omml_equations
                        )
                        return
                    latex = self._decode_pptx_ole_equation(shape)
                    if latex:
                        self.cur_page.append(
                            {
                                "type": BlockType.PAGE_FOOTNOTE,
                                "content": [{"type": "equation_inline", "content": latex}],
                            }
                        )
                        return
                    image_latex = self._decode_pptx_shape_image_equation(shape)
                    if image_latex:
                        self.cur_page.append(
                            {
                                "type": BlockType.PAGE_FOOTNOTE,
                                "content": [{"type": "equation_inline", "content": image_latex}],
                            }
                        )
                return

            if shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_latex = self._decode_pptx_shape_image_equation(shape)
                if image_latex:
                    self.cur_page.append(
                        {
                            "type": BlockType.PAGE_FOOTNOTE,
                            "content": [{"type": "equation_inline", "content": image_latex}],
                        }
                    )
                    return

            if self._should_skip_notes_shape(shape):
                return

            for paragraph in shape.text_frame.paragraphs:
                note_text = self._normalize_text_block_content(self._build_paragraph_rich_text(paragraph, shape))
                if not note_text:
                    continue
                self.cur_page.append(
                    {
                        "type": BlockType.PAGE_FOOTNOTE,
                        "content": note_text,
                    }
                )

        for shape in notes_slide.shapes:
            handle_notes_shape(shape)

    @staticmethod
    def _should_skip_notes_shape(shape) -> bool:
        if not PptxConverter._shape_has_raw_text(shape):
            return True

        if not getattr(shape, "is_placeholder", False):
            return False

        try:
            return shape.placeholder_format.type in IGNORED_NOTES_PLACEHOLDER_TYPES
        except Exception:
            return False

    def _handle_text_elements(self, shape):
        self.list_block_stack = []
        list_level_base: Optional[int] = None

        # 遍历段落以构建文本
        for paragraph in shape.text_frame.paragraphs:
            list_info = self._get_paragraph_list_info(shape, paragraph)

            if list_info["is_list"]:
                rich_text = self._normalize_text_block_content(self._build_paragraph_rich_text(paragraph, shape))
                if rich_text:
                    if list_level_base is None:
                        list_level_base = list_info["level"]
                    self._append_list_item(
                        self.list_block_stack,
                        self._normalize_contiguous_list_level(
                            list_info["level"],
                            list_level_base,
                        ),
                        list_info["attribute"],
                        rich_text,
                        list_info.get("start"),
                        list_info.get("start_is_explicit_restart", False),
                    )
                continue

            # 段落不是列表项，关闭当前 shape 的列表上下文
            self.list_block_stack.clear()
            list_level_base = None

            p_text = self._normalize_text_block_content(self._build_paragraph_rich_text(paragraph, shape))
            if not p_text:
                continue

            style_profile = self._build_paragraph_style_profile(shape, paragraph)

            block = {
                "type": BlockType.TEXT,
                "content": p_text,
                _EFFECTIVE_FONT_SIZE_KEY: style_profile["font_size_pt"],
                _EFFECTIVE_ALL_BOLD_KEY: style_profile["all_bold"],
            }
            if shape.is_placeholder:
                placeholder_type = shape.placeholder_format.type
                title_role = _PPTX_TITLE_PLACEHOLDER_ROLES.get(placeholder_type)
                if title_role is not None:
                    block[_PPTX_TITLE_CANDIDATE_KEY] = True
                    block["level"] = 2
                    block[_PPTX_TITLE_ROLE_KEY] = title_role

            self.cur_page.append(block)

        # shape 结束后清理列表上下文，避免跨 shape 污染
        self.list_block_stack.clear()
        return
