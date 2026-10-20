"""PPTX 固定规则和形状变换数据契约。"""

from dataclasses import dataclass
from typing import Any, Final, Optional

from pptx.enum.shapes import PP_PLACEHOLDER

from .....foundation.type_identity import preserve_type_module

IGNORED_NOTES_PLACEHOLDER_TYPES: Final = {
    PP_PLACEHOLDER.SLIDE_IMAGE,
    PP_PLACEHOLDER.SLIDE_NUMBER,
    PP_PLACEHOLDER.DATE,
    PP_PLACEHOLDER.FOOTER,
}


MIN_PICTURE_DIMENSION_RATIO: Final = 0.1


MIN_PICTURE_AREA_RATIO: Final = 0.01


BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO: Final = 0.1


PPTX_XYCUT_BETA: Final = 2.0


PPTX_XYCUT_DENSITY_THRESHOLD: Final = 0.9


DRAWINGML_NS: Final = "http://schemas.openxmlformats.org/drawingml/2006/main"


RELATIONSHIP_NS: Final = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


SVG_BLIP_NS: Final = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


A14_DRAWING_NS: Final = "http://schemas.microsoft.com/office/drawing/2010/main"


OMML_NS: Final = "http://schemas.openxmlformats.org/officeDocument/2006/math"


_EFFECTIVE_FONT_SIZE_KEY: Final = "_effective_font_size_pt"


_EFFECTIVE_ALL_BOLD_KEY: Final = "_effective_all_bold"


_PPTX_TITLE_CANDIDATE_KEY: Final = "_pptx_title_candidate"


_PPTX_TITLE_ROLE_KEY: Final = "_pptx_title_role"


_PPTX_TITLE_ROLE_CENTER: Final = "center_title"


_PPTX_TITLE_ROLE_TITLE: Final = "title"


_PPTX_TITLE_ROLE_SUBTITLE: Final = "subtitle"


_PPTX_TITLE_PLACEHOLDER_ROLES: Final = {
    PP_PLACEHOLDER.CENTER_TITLE: _PPTX_TITLE_ROLE_CENTER,
    PP_PLACEHOLDER.TITLE: _PPTX_TITLE_ROLE_TITLE,
    PP_PLACEHOLDER.SUBTITLE: _PPTX_TITLE_ROLE_SUBTITLE,
}


@dataclass(frozen=True)
class _SlideTransform:
    scale_x: float = 1.0
    scale_y: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0

    def apply_bbox(
        self,
        bbox: Optional[tuple[float, float, float, float]],
    ) -> Optional[tuple[float, float, float, float]]:
        """按既有形状缩放和平移规则转换几何，保持嵌套变换顺序。"""
        if bbox is None:
            return None

        left = self.scale_x * bbox[0] + self.translate_x
        top = self.scale_y * bbox[1] + self.translate_y
        right = self.scale_x * bbox[2] + self.translate_x
        bottom = self.scale_y * bbox[3] + self.translate_y
        return (left, top, right, bottom)

    def compose(self, inner: "_SlideTransform") -> "_SlideTransform":
        """按既有形状缩放和平移规则转换几何，保持嵌套变换顺序。"""
        return _SlideTransform(
            scale_x=self.scale_x * inner.scale_x,
            scale_y=self.scale_y * inner.scale_y,
            translate_x=self.scale_x * inner.translate_x + self.translate_x,
            translate_y=self.scale_y * inner.translate_y + self.translate_y,
        )


@dataclass(frozen=True)
class _FlattenedShape:
    shape: Any
    bbox: Optional[tuple[float, float, float, float]]


preserve_type_module(_SlideTransform, "rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.pptx_converter")
preserve_type_module(_FlattenedShape, "rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.pptx_converter")
