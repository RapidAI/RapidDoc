"""仅以可证明的后绘制不透明矩形排除完全遮挡文字，原始文本接口保持不变。"""

from __future__ import annotations

import ctypes
from ..._compute_backend import get_native

import pypdfium2.raw as raw

from .native_coordinates import _transform_drawing_point
from .native_objects import (
    _ClippedObject,
    _clip_object_visual_bbox,
    _filled_rectangle_bbox,
    _get_raw_object_rgba,
    _read_raw_path_subpaths,
    _transform_object_bbox,
    _transform_path_subpath,
    _walk_clipped_objects,
)


def _safe_paint_resources(page):
    """混合、软遮罩、透明组或未知资源均拒绝遮挡推断，避免把不透明外框当作实心覆盖。"""
    visited = set()

    def visit(node):
        """递归核验实际资源中的图形状态和Form，循环引用保守拒绝。"""
        identity = id(node)
        if identity in visited:
            return True
        visited.add(identity)
        if node.get("/Group"):
            return False
        resources = node.get("/Resources", {})
        for state in resources.get("/ExtGState", {}).values():
            state = state.get_object()
            if (
                state.get("/BM", "/Normal") != "/Normal"
                or state.get("/SMask", "/None") != "/None"
                or state.get("/ca", 1) != 1
                or state.get("/CA", 1) != 1
                or getattr(state.get("/OP", False), "value", state.get("/OP", False))
                or getattr(state.get("/op", False), "value", state.get("/op", False))
            ):
                return False
        return all(
            visit(obj.get_object())
            for obj in resources.get("/XObject", {}).values()
            if obj.get_object().get("/Subtype") == "/Form"
        )

    try:
        return page is not None and visit(page)
    except Exception:
        return False


def _rectangular_clip(obj):
    """只接受无裁剪或由轴对齐矩形组成的裁剪，曲线外框不能作为完全覆盖证明。"""
    clip = raw.FPDFPageObj_GetClipPath(obj)
    for path in range(raw.FPDFClipPath_CountPaths(clip) if clip else 0):
        points = []
        for index in range(raw.FPDFClipPath_CountPathSegments(clip, path)):
            segment = raw.FPDFClipPath_GetPathSegment(clip, path, index)
            if raw.FPDFPathSegment_GetType(segment) not in {raw.FPDF_SEGMENT_MOVETO, raw.FPDF_SEGMENT_LINETO}:
                return False
            x, y = ctypes.c_float(), ctypes.c_float()
            if not raw.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
                return False
            points.append((round(x.value, 3), round(y.value, 3)))
        if not points:
            return False
        xs, ys = {p[0] for p in points}, {p[1] for p in points}
        if len(xs) != 2 or len(ys) != 2 or set(points) != {(x, y) for x in xs for y in ys}:
            return False
        if any(a[0] != b[0] and a[1] != b[1] for a, b in zip(points, points[1:] + points[:1])):
            return False
    return True


def _opaque_rectangle(member, page_bbox, rotation):
    """仅核验顶层单一闭合实心矩形，嵌套透明组与复合路径不参与覆盖推断。"""
    if member.depth != 0 or raw.FPDFPageObj_GetType(member.raw) != raw.FPDF_PAGEOBJ_PATH:
        return None
    if not 4 <= raw.FPDFPath_CountSegments(member.raw) <= 5:
        return None
    fill, stroke = ctypes.c_int(), ctypes.c_int()
    if not raw.FPDFPath_GetDrawMode(member.raw, ctypes.byref(fill), ctypes.byref(stroke)) or not fill.value:
        return None
    color = _get_raw_object_rgba(member.raw, raw.FPDFPageObj_GetFillColor)
    if color is None or color[3] != 255 or not _rectangular_clip(member.raw):
        return None
    paths = _read_raw_path_subpaths(member.raw)
    if len(paths) != 1 or not paths[0].closed:
        return None
    transformed = _transform_path_subpath(paths[0], member.matrix, page_bbox, rotation)
    rectangle = _filled_rectangle_bbox(transformed)
    return _clip_object_visual_bbox(rectangle, member.clip, page_bbox, rotation) if rectangle else None


_PAINT_COVER_PREFILTER_REFERENCES = (_opaque_rectangle, _get_raw_object_rgba)


def _allow_native_cover_prefilter():
    """替换遮挡规则或透明度读取时保留全部原路径，不能绕过调用方的证据策略。"""
    return (_opaque_rectangle, _get_raw_object_rgba) == _PAINT_COVER_PREFILTER_REFERENCES


def exclude_fully_overpainted_text(page, page_bbox, rotation, visibility, page_reader, *, roots=None, paint=None):
    """共享Python/Rust可见性结果，仅在绘制顺序、完整包含和资源状态均证明遮挡时排除对象。"""
    covers = []
    texts = []
    try:
        root_covers = None
        if roots is not None:
            # 只有顶层单一不透明矩形能够覆盖文字；没有这种证据就无需再读全部文字框。
            root_covers = {}
            for order, (address, matrix, parent, depth, clip) in roots:
                member = _ClippedObject(ctypes.cast(address, raw.FPDF_PAGEOBJECT), matrix, parent, depth, clip)
                rectangle = _opaque_rectangle(member, page_bbox, rotation)
                if rectangle is not None:
                    root_covers[address] = rectangle
                    covers.append((order, rectangle))
            if not root_covers:
                return visibility
        texts = paint if paint is not None else texts
        for order, member in enumerate(_walk_clipped_objects(page)) if paint is None else ():
            if raw.FPDFPageObj_GetType(member.raw) == raw.FPDF_PAGEOBJ_TEXT:
                address = ctypes.cast(member.raw, ctypes.c_void_p).value
                values = [ctypes.c_float() for _ in range(4)]
                if visibility.get(address, (False, None))[0] and raw.FPDFPageObj_GetBounds(
                    member.raw, *(ctypes.byref(value) for value in values)
                ):
                    bounds = _transform_object_bbox(
                        tuple(value.value for value in values),
                        lambda point: _transform_drawing_point(
                            (
                                member.parent_matrix[0] * point[0]
                                + member.parent_matrix[2] * point[1]
                                + member.parent_matrix[4],
                                member.parent_matrix[1] * point[0]
                                + member.parent_matrix[3] * point[1]
                                + member.parent_matrix[5],
                            ),
                            page_bbox,
                            rotation,
                        ),
                    )
                    texts.append((order, address, bounds))
            else:
                rectangle = (
                    root_covers.get(ctypes.cast(member.raw, ctypes.c_void_p).value)
                    if root_covers is not None
                    else _opaque_rectangle(member, page_bbox, rotation)
                )
                if rectangle:
                    covers.append((order, rectangle))
        native = get_native()
        covered = (
            set(native.overpainted_addresses(texts, covers))
            if paint is not None and native is not None
            else {
                address
                for order, address, bounds in texts
                if any(
                    later > order
                    and box[0] <= bounds[0]
                    and box[1] <= bounds[1]
                    and box[2] >= bounds[2]
                    and box[3] >= bounds[3]
                    for later, box in covers
                )
            }
        )
        if covered and _safe_paint_resources(page_reader()):
            return {address: (False if address in covered else value[0], value[1]) for address, value in visibility.items()}
    except Exception:
        # 任何读取失败都不能以缺失证据删除正文。
        pass
    return visibility
