"""PPTX 形状层级及页面几何，复用当前转换器的单文档状态。"""

from typing import Optional
from loguru import logger
from pptx.enum.shapes import MSO_SHAPE_TYPE
from .....schema import BlockType

from .context import (
    MIN_PICTURE_DIMENSION_RATIO,
    MIN_PICTURE_AREA_RATIO,
    BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO,
    DRAWINGML_NS,
    OMML_NS,
    _SlideTransform,
    _FlattenedShape,
)


class _PptxShapes:
    """集中维护形状层级及页面几何，不改变文档生命周期和公开入口。"""

    @staticmethod
    def _shape_type_cache_key(
        shape,
    ) -> Optional[tuple[Optional[str], Optional[int], Optional[str]]]:
        """按原有形状层级及页面几何规则执行 _shape_type_cache_key，保持输入顺序与降级行为。"""
        part = getattr(shape, "part", None)
        partname = getattr(part, "partname", None)
        element = getattr(shape, "element", None)
        element_tag = getattr(element, "tag", None)
        try:
            shape_id = shape.shape_id
        except Exception:
            shape_id = None

        if partname is None and shape_id is None and element_tag is None:
            return None

        return (partname, shape_id, element_tag)

    def _safe_shape_type(self, shape) -> Optional[MSO_SHAPE_TYPE]:
        """按原有形状层级及页面几何规则执行 _safe_shape_type，保持输入顺序与降级行为。"""
        shape_key = self._shape_type_cache_key(shape)
        if shape_key is not None and shape_key in self._shape_type_cache:
            return self._shape_type_cache[shape_key]

        try:
            shape_type = shape.shape_type
        except NotImplementedError as exc:
            shape_name = getattr(shape, "name", None)
            shape_element = getattr(shape, "element", None)
            shape_element_tag = getattr(shape_element, "tag", None)
            has_text_frame = getattr(shape, "has_text_frame", None)
            logger.warning(
                "Skipping shape_type-specific handling for unrecognized PPTX shape: "
                f"class={type(shape).__name__}, "
                f"name={shape_name!r}, "
                f"element_tag={shape_element_tag!r}, "
                f"has_text_frame={has_text_frame!r}, "
                f"error={exc}"
            )
            shape_type = None

        if shape_key is not None:
            self._shape_type_cache[shape_key] = shape_type
        return shape_type

    def _flatten_slide_shapes(
        self,
        shapes,
        slide_transform: Optional[_SlideTransform] = None,
    ) -> list[_FlattenedShape]:
        """按原有形状层级及页面几何规则执行 _flatten_slide_shapes，保持输入顺序与降级行为。"""
        if slide_transform is None:
            slide_transform = _SlideTransform()

        linear_shapes: list[_FlattenedShape] = []
        for shape in shapes:
            shape_type = self._safe_shape_type(shape)
            if shape_type == MSO_SHAPE_TYPE.GROUP:
                group_transform = self._group_shape_transform(shape)
                linear_shapes.extend(
                    self._flatten_slide_shapes(
                        shape.shapes,
                        slide_transform.compose(group_transform),
                    )
                )
            else:
                linear_shapes.append(
                    _FlattenedShape(
                        shape=shape,
                        bbox=slide_transform.apply_bbox(self._shape_bbox(shape)),
                    )
                )
        return linear_shapes

    def _collect_shape_blocks(
        self,
        shape_entry: _FlattenedShape,
        linear_shapes: list[_FlattenedShape],
        shape_index: int,
        slide_width: int,
        slide_height: int,
    ) -> list:
        """按原有形状层级及页面几何规则执行 _collect_shape_blocks，保持输入顺序与降级行为。"""
        shape = shape_entry.shape
        shape_blocks = []
        previous_page = self.cur_page
        previous_list_block_stack = self.list_block_stack
        self.cur_page = shape_blocks
        self.list_block_stack = []

        try:
            shape_type = self._safe_shape_type(shape)
            if shape_type in {
                MSO_SHAPE_TYPE.EMBEDDED_OLE_OBJECT,
                MSO_SHAPE_TYPE.LINKED_OLE_OBJECT,
            }:
                if self._is_equation_ole_shape(shape):
                    omml_equations = self._pptx_ole_shape_omml(shape)
                    if omml_equations:
                        self.cur_page.extend(
                            {
                                "type": BlockType.EQUATION,
                                "content": latex,
                            }
                            for latex in omml_equations
                        )
                        return shape_blocks
                    latex = self._decode_pptx_ole_equation(shape)
                    if latex:
                        self.cur_page.append(
                            {
                                "type": BlockType.EQUATION,
                                "content": latex,
                            }
                        )
                    else:
                        self._handle_pictures(shape)
                return shape_blocks

            if shape.has_table:
                self._handle_tables(shape)

            if getattr(shape, "has_chart", False):
                self._handle_chart(shape)

            if shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_latex = self._decode_pptx_shape_image_equation(shape)
                if image_latex:
                    self.cur_page.append(
                        {
                            "type": BlockType.EQUATION,
                            "content": image_latex,
                        }
                    )
                    return shape_blocks
                later_shapes = linear_shapes[shape_index + 1 :]
                if not self._should_skip_picture(
                    shape_entry,
                    later_shapes,
                    slide_width,
                    slide_height,
                ):
                    self._handle_pictures(shape)

            if not getattr(shape, "has_text_frame", False):
                return shape_blocks

            self._handle_text_elements(shape)
            return shape_blocks
        finally:
            self.cur_page = previous_page
            self.list_block_stack = previous_list_block_stack

    @staticmethod
    def _shape_bbox(shape) -> Optional[tuple[float, float, float, float]]:
        """按原有形状层级及页面几何规则执行 _shape_bbox，保持输入顺序与降级行为。"""
        try:
            left = float(shape.left)
            top = float(shape.top)
            width = float(shape.width)
            height = float(shape.height)
        except Exception:
            return None

        if width <= 0 or height <= 0:
            return None

        return (left, top, left + width, top + height)

    @staticmethod
    def _group_shape_transform(shape) -> _SlideTransform:
        """按原有形状层级及页面几何规则执行 _group_shape_transform，保持输入顺序与降级行为。"""
        group_properties = getattr(shape._element, "grpSpPr", None)
        xfrm = getattr(group_properties, "xfrm", None) if group_properties is not None else None
        if xfrm is None:
            return _SlideTransform()

        child_offset = getattr(xfrm, "chOff", None)
        child_extent = getattr(xfrm, "chExt", None)
        if child_offset is None or child_extent is None:
            return _SlideTransform()

        try:
            offset_x = float(xfrm.x)
            offset_y = float(xfrm.y)
            extent_x = float(xfrm.cx)
            extent_y = float(xfrm.cy)
            child_offset_x = float(child_offset.x)
            child_offset_y = float(child_offset.y)
            child_extent_x = float(child_extent.cx)
            child_extent_y = float(child_extent.cy)
        except Exception:
            return _SlideTransform()

        if extent_x <= 0 or extent_y <= 0 or child_extent_x <= 0 or child_extent_y <= 0:
            return _SlideTransform()

        scale_x = extent_x / child_extent_x
        scale_y = extent_y / child_extent_y
        return _SlideTransform(
            scale_x=scale_x,
            scale_y=scale_y,
            translate_x=offset_x - child_offset_x * scale_x,
            translate_y=offset_y - child_offset_y * scale_y,
        )

    @staticmethod
    def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
        """按原有形状层级及页面几何规则执行 _bbox_area，保持输入顺序与降级行为。"""
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _bbox_intersection(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> Optional[tuple[float, float, float, float]]:
        """按原有形状层级及页面几何规则执行 _bbox_intersection，保持输入顺序与降级行为。"""
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)

    @classmethod
    def _rectangles_union_area(cls, bboxes: list[tuple[float, float, float, float]]) -> float:
        """按原有形状层级及页面几何规则执行 _rectangles_union_area，保持输入顺序与降级行为。"""
        if not bboxes:
            return 0.0

        xs = sorted({bbox[0] for bbox in bboxes} | {bbox[2] for bbox in bboxes})
        total_area = 0.0

        for idx in range(len(xs) - 1):
            x_left = xs[idx]
            x_right = xs[idx + 1]
            if x_right <= x_left:
                continue

            y_intervals = []
            for bbox in bboxes:
                if bbox[0] < x_right and bbox[2] > x_left:
                    y_intervals.append((bbox[1], bbox[3]))

            if not y_intervals:
                continue

            y_intervals.sort()
            merged_height = 0.0
            current_y0, current_y1 = y_intervals[0]

            for y0, y1 in y_intervals[1:]:
                if y0 <= current_y1:
                    current_y1 = max(current_y1, y1)
                    continue

                merged_height += max(0.0, current_y1 - current_y0)
                current_y0, current_y1 = y0, y1

            merged_height += max(0.0, current_y1 - current_y0)
            total_area += (x_right - x_left) * merged_height

        return total_area

    @staticmethod
    def _shape_has_raw_text(shape) -> bool:
        """通过shape底层XML判断是否有文本，避免数学公式触发python-pptx文本转换。"""
        if not getattr(shape, "has_text_frame", False):
            return False

        shape_xml = getattr(shape, "_element", None)
        if shape_xml is None:
            return False

        text_tags = {
            f"{{{DRAWINGML_NS}}}t",
            f"{{{OMML_NS}}}t",
        }
        for text_node in shape_xml.iter():
            if getattr(text_node, "tag", None) not in text_tags:
                continue
            if text_node.text and text_node.text.strip():
                return True

        return False

    @staticmethod
    def _is_nonempty_text_shape(shape) -> bool:
        """按原有形状层级及页面几何规则执行 _is_nonempty_text_shape，保持输入顺序与降级行为。"""
        return _PptxShapes._shape_has_raw_text(shape)

    def _is_small_picture(
        self,
        picture_bbox: Optional[tuple[float, float, float, float]],
        slide_width: int,
        slide_height: int,
    ) -> bool:
        """按原有形状层级及页面几何规则执行 _is_small_picture，保持输入顺序与降级行为。"""
        if picture_bbox is None:
            return False

        picture_width = picture_bbox[2] - picture_bbox[0]
        picture_height = picture_bbox[3] - picture_bbox[1]

        if picture_width <= 0 or picture_height <= 0:
            return False

        slide_area = float(slide_width) * float(slide_height)
        if slide_area <= 0:
            return False

        if picture_width < MIN_PICTURE_DIMENSION_RATIO * float(slide_width):
            return True
        if picture_height < MIN_PICTURE_DIMENSION_RATIO * float(slide_height):
            return True

        picture_area_ratio = (picture_width * picture_height) / slide_area
        return picture_area_ratio < MIN_PICTURE_AREA_RATIO

    def _is_background_picture(
        self,
        picture_entry: _FlattenedShape,
        later_shapes: list[_FlattenedShape],
    ) -> bool:
        """按原有形状层级及页面几何规则执行 _is_background_picture，保持输入顺序与降级行为。"""
        picture_bbox = picture_entry.bbox
        if picture_bbox is None:
            return False

        picture_area = self._bbox_area(picture_bbox)
        if picture_area <= 0:
            return False

        overlap_bboxes = []
        for later_shape in later_shapes:
            if not self._is_nonempty_text_shape(later_shape.shape):
                continue

            later_bbox = later_shape.bbox
            if later_bbox is None:
                continue

            overlap_bbox = self._bbox_intersection(picture_bbox, later_bbox)
            if overlap_bbox is not None:
                overlap_bboxes.append(overlap_bbox)

        if not overlap_bboxes:
            return False

        covered_area = self._rectangles_union_area(overlap_bboxes)
        return covered_area / picture_area >= BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO

    def _should_skip_picture(
        self,
        picture_entry: _FlattenedShape,
        later_shapes: list[_FlattenedShape],
        slide_width: int,
        slide_height: int,
    ) -> bool:
        """按原有形状层级及页面几何规则执行 _should_skip_picture，保持输入顺序与降级行为。"""
        return self._is_small_picture(
            picture_entry.bbox,
            slide_width,
            slide_height,
        ) or self._is_background_picture(
            picture_entry,
            later_shapes,
        )
