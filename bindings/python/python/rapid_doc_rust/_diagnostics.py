"""Small supported diagnostic APIs; all computation delegates to Rust."""

import json
import math
import sys
from dataclasses import dataclass
from ._native import recover_pdf_table_json, render_pdf_page_bytes
from ._native import OnnxSession as _OnnxSession
from ._native import ocr_png_json
from ._native import match_table_json
from ._native import match_wired_table_json
from ._native import wired_matches_json
from ._native import legacy_pipeline_json
from ._document import _read_bytes, _prepare_pdf


class OnnxSession:
    """FP32 ONNX 开发接口，要求显式运行库、模型路径及 SHA256。

    run 接收单输入，run_named 接收命名多输入并保留输出类型。CUDA 构建和运行库
    必须都支持所请求设备，否则报错，不静默切换到 CPU。
    """

    def __init__(
        self,
        runtime,
        model,
        sha256,
        *,
        device="cpu",
        threads=1,
        max_elements=None,
        cpu_spinning=None,
        cpu_dynamic_block_base=None,
        cpu_force_spinning_stop=None,
        cpu_shared_arena=None,
        operator_libraries=None,
    ):
        self._native = _OnnxSession(
            str(runtime),
            str(model),
            sha256,
            device,
            threads,
            sys.maxsize // 4 if max_elements is None else max_elements,
            cpu_spinning,
            cpu_dynamic_block_base,
            cpu_force_spinning_stop,
            cpu_shared_arena,
            operator_libraries,
        )

    def run(self, shape, values):
        """执行一次推理，返回按输出名称组织的 shape/values 字典。"""
        return json.loads(self._native.run_json(shape, values))

    def run_named(self, inputs):
        """命名的 FP32 输入字典 {name: {shape, values}}；返回带 dtype 的有类型张量。"""
        return json.loads(self._native.run_named_json(json.dumps(inputs, allow_nan=False)))

    def recognize_ctc(self, shape, values):
        """识别预处理后的图像张量，在 Rust 中使用模型字典完成 CTC 解码。"""
        return json.loads(self._native.recognize_ctc_json(shape, values))

    def recognize_png(self, source):
        """识别一张已裁出的单行 PNG，Rust 完成 BGR/resize/padding、推理及解码。"""
        return json.loads(self._native.recognize_png_json(_read_bytes(source, 256 * 1024 * 1024)))

    def recognize_batch_png(self, sources, *, batch_size=6):
        """旧宽高比排序/共同 padding，结果还原输入顺序；仅执行识别。"""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 64
        ):
            raise ValueError("batch_size must be 1..64")
        if len(sources) > 1000:
            raise ValueError("too many recognition images")
        images = []
        total = 0
        for source in sources:
            data = _read_bytes(source, 64_000_000)
            total += len(data)
            if total > 128_000_000:
                raise ValueError("recognition bytes exceed budget")
            images.append(data)
        return json.loads(self._native.recognize_batch_png_json(images, batch_size))

    def recognize_formula_batch_png(self, sources, *, batch_size=1):
        """PP-FormulaNet Plus单通道384/768、旧BPE解码/LaTeX及默认ftfy6.3.1修复。"""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 64
        ):
            raise ValueError("batch_size must be 1..64")
        if len(sources) > 1000:
            raise ValueError("too many formula images")
        images = []
        streams = {}
        total = 0
        for source in sources:
            if callable(getattr(source, "read", None)):
                if id(source) not in streams:
                    streams[id(source)] = _read_bytes(source, 64_000_000)
                data = streams[id(source)]
            else:
                data = _read_bytes(source, 64_000_000)
            total += len(data)
            if total > 128_000_000:
                raise ValueError("formula bytes exceed budget")
            images.append(data)
        return json.loads(self._native.recognize_formula_batch_png_json(images, batch_size))

    def detect_png(self, source):
        """使用 PP-OCR DB 模型检测 PNG，返回原图像素四角和评分；尚不裁图或识别。"""
        return json.loads(self._native.detect_png_json(_read_bytes(source, 256 * 1024 * 1024)))

    def detect_table_png(self, source):
        """表格DB原始检测：固定960/unclip1.6及内部排序；外层不合并/遮罩切框由表格阶段处理。"""
        return json.loads(self._native.detect_table_png_json(_read_bytes(source, 64_000_000)))

    def detect_seal_png(self, source):
        """专用印章 DB 检测与多边形扩框；返回原图坐标及检测分数。"""
        return json.loads(self._native.detect_seal_png_json(_read_bytes(source, 64_000_000)))

    def detect_model_png(self, source):
        """旧RapidDoc补丁生效的固定960/unclip=1.8/内部排序，单张执行。"""
        return json.loads(self._native.detect_model_png_json(_read_bytes(source, 64_000_000)))

    def recognize_regions_png(self, source, regions):
        """按调用方顺序识别 PNG 的四角区域，Rust 完成透视裁图和竖行旋转。"""
        return json.loads(
            self._native.recognize_regions_png_json(_read_bytes(source, 256 * 1024 * 1024), regions)
        )

    def classify_orientation_png(self, source):
        """匹配 RapidOrientationModel 的 RGB 路径，返回模型角度/评分，不自动旋转。"""
        return json.loads(
            self._native.classify_orientation_png_json(_read_bytes(source, 256 * 1024 * 1024))
        )

    def classify_table_png(self, source, *, variant="paddle", output_type=None):
        """Paddle BGR/Lanczos4 或 Qanything 灰度/Pillow 分类；默认分别按概率/logits 解码。"""
        return json.loads(
            self._native.classify_table_png_json(
                _read_bytes(source, 256 * 1024 * 1024), variant, output_type
            )
        )

    def layout_png(self, source, *, threshold=0.5):
        """PP-DocLayoutV3 矩形模式：类别、评分、原图像素框和模型顺序；不输出 mask 多边形。"""
        return json.loads(
            self._native.layout_png_json(_read_bytes(source, 256 * 1024 * 1024), threshold)
        )

    def table_structure_png(self, source, *, variant="slanet-plus"):
        """SLANet/plus 结构 token、评分和像素单元格框；尚未填入单元格 OCR 文字。"""
        return json.loads(
            self._native.table_structure_png_json(_read_bytes(source, 256 * 1024 * 1024), variant)
        )

    def table_structure_legacy_png(self, source, *, variant="slanet-plus"):
        """旧matcher使用的SLANet结构；将F32坐标完整提升为F64再序列化，避免短十进制改变边界。"""
        return json.loads(
            self._native.table_structure_png_json(_read_bytes(source, 64_000_000), variant, True)
        )

    def table_line_mask_png(self, source):
        """UNET 缩放图的原始类别图（0 背景/1 横线/2 竖线），尚不恢复格拓扑。"""
        return json.loads(
            self._native.table_line_mask_png_json(_read_bytes(source, 256 * 1024 * 1024))
        )

    def table_cells_png(self, source, *, rotated_fix=True):
        """UNET PNG→线图→旋转修正→闭合区域→逻辑格；坐标为原图像素逆时针四角。"""
        return json.loads(
            self._native.table_cells_png_json(_read_bytes(source, 256 * 1024 * 1024), rotated_fix)
        )

    def recognize_blank_table_cells_png(self, source, polygons, *, needed=None):
        """对未匹配格执行轴框内缩、对比度筛选及 CTC，返回接受/过滤原因和原格索引。"""
        if needed is None:
            needed = [True] * len(polygons)
        return json.loads(
            self._native.recognize_blank_table_cells_png_json(
                _read_bytes(source, 256 * 1024 * 1024), polygons, needed
            )
        )


def _wired_request(cells, ocr_lines, blank_cells=()):
    topology = cells["topology"]
    return dict(
        polygons=cells["polygons"],
        logic_points=topology["logic_points"] if topology else [],
        blank_cells=list(blank_cells),
        ocr=[
            dict(points=line["points"], text=line["text"], confidence=line["confidence"])
            for line in ocr_lines
        ],
    )


def match_wired_table(cells, ocr_lines, *, blank_cells=()):
    """UNET 专用 OCR 跨格拆分、同行归并、噪声边缘裁剪与转义 HTML。

    cells 来自 table_cells_png；坐标均为同一原图像素。输出保留原 OCR 索引和拆分标记。
    """
    request = _wired_request(cells, ocr_lines, blank_cells)
    return json.loads(match_wired_table_json(json.dumps(request, allow_nan=False)))


def recognize_wired_table_png(
    source,
    *,
    structure_model,
    detector,
    recognizer,
    rotated_fix=True,
    min_confidence=0.0,
    blank_cell_rec=True,
):
    """显式有线表格 PNG→UNET→OCR→专用匹配→空格补识别→HTML；不自动分类。"""
    data = _read_bytes(source, 256 * 1024 * 1024)
    cells = structure_model.table_cells_png(data, rotated_fix=rotated_fix)
    lines = ocr_png(data, detector=detector, recognizer=recognizer, min_confidence=min_confidence)
    return _complete_wired_table(data, cells, lines, recognizer, blank_cell_rec)


def _complete_wired_table(data, cells, lines, recognizer, blank_cell_rec):
    blank = []
    if blank_cell_rec and cells["polygons"]:
        matched = json.loads(
            wired_matches_json(json.dumps(_wired_request(cells, lines), allow_nan=False))
        )
        needed = [not fragments for fragments in matched["cells"]]
        if any(needed):
            blank = recognizer.recognize_blank_table_cells_png(
                data, cells["polygons"], needed=needed
            )
    return dict(
        cells=cells,
        ocr_lines=lines,
        blank_rec=blank,
        **match_wired_table(cells, lines, blank_cells=blank),
    )


def match_table(structure, ocr_lines):
    """把 OCR 像素四角匹配到 SLANet 格框，返回源索引、文字和转义后的 HTML。

    structure 来自 table_structure_png，ocr_lines 来自同一图像的 ocr_png。
    结构不完整或 token/格框数量不符时报错；单元格内按 OCR 输入顺序以空格连接。
    """
    request = dict(
        tokens=structure["tokens"],
        cell_bboxes=structure["cell_bboxes"],
        ocr=[
            dict(bbox=[v for point in line["points"] for v in point], text=line["text"])
            for line in ocr_lines
        ],
    )
    return json.loads(match_table_json(json.dumps(request, allow_nan=False)))


def recognize_table_png(
    source, *, structure_model, detector, recognizer, variant="slanet-plus", min_confidence=0.0
):
    """显式表格裁图的 SLANet→OCR→匹配→安全 HTML；不自动分类或执行 UNET。"""
    data = _read_bytes(source, 256 * 1024 * 1024)
    structure = structure_model.table_structure_png(data, variant=variant)
    lines = ocr_png(data, detector=detector, recognizer=recognizer, min_confidence=min_confidence)
    return dict(structure=structure, ocr_lines=lines, **match_table(structure, lines))


def ocr_png(source, *, detector, recognizer, min_confidence=0.0):
    """PNG 检测→同行排序→透视裁图→逐行识别。需传入已加载的 DB/CTC 会话。

    此预览不执行方向分类、印章识别、版面识别或段落合并。
    四角坐标以原图左上为原点，单位为像素。
    """
    if (
        not isinstance(min_confidence, (int, float))
        or not math.isfinite(min_confidence)
        or not 0 <= min_confidence <= 1
    ):
        raise ValueError("min_confidence must be finite and within [0, 1]")
    data = _read_bytes(source, 256 * 1024 * 1024)
    results = json.loads(ocr_png_json(detector._native, recognizer._native, data))
    return [line for line in results if line["confidence"] >= min_confidence]


def render_model_pages(pages, *, make_mode="mm_markdown", image_path_prefix=""):
    """Original Markdown/content_list rendering, including native Magika code-language labels."""
    request = dict(pages=pages, make_mode=make_mode, image_path_prefix=image_path_prefix)
    from ._runtime import runtime_path

    try:
        request["magika_runtime"] = str(runtime_path())
    except (OSError, RuntimeError, ImportError):
        pass
    return json.loads(legacy_pipeline_json("render_pages", json.dumps(request, allow_nan=False)))


@dataclass
class RecoveredTable:
    """已知区域的表格，保留原页号、单元格跨度和 HTML。"""

    page_index: int
    table: dict
    html: str


@dataclass
class RenderedPage:
    """独立 PNG 字节和像素尺寸，不需要维持来源文档的生命周期。"""

    width: int
    height: int
    png: bytes


@dataclass
class OcrPageOutput:
    """指定页的栅格 OCR；lines 坐标为渲染图像素，text 仅按行连接。"""

    page_index: int
    width: int
    height: int
    lines: list
    text: str


def ocr_pdf_page(
    source,
    *,
    detector,
    recognizer,
    page_index=0,
    dpi=144.0,
    max_pixels=16_000_000,
    pdfium_library=None,
    password=None,
    min_confidence=0.0,
):
    """渲染所选 PDF 页后执行 OCR；不与原生文字合并，尚未做版面/段落分析。"""
    page = render_pdf_page(
        source,
        page_index=page_index,
        dpi=dpi,
        max_pixels=max_pixels,
        pdfium_library=pdfium_library,
        password=password,
    )
    lines = ocr_png(
        page.png, detector=detector, recognizer=recognizer, min_confidence=min_confidence
    )
    return OcrPageOutput(
        page_index, page.width, page.height, lines, "\n".join(line["text"] for line in lines)
    )


def render_pdf_page(
    source,
    *,
    page_index=0,
    dpi=144.0,
    max_pixels=16_000_000,
    pdfium_library=None,
    password=None,
    color_mode="RGBA",
):
    """按原页号渲染为白底 PNG，渲染前检查像素预算；不运行 OCR。"""
    if color_mode not in ("RGB", "RGBA"):
        raise ValueError("color_mode must be RGB or RGBA")
    library, data = _prepare_pdf(source, pdfium_library, None)
    width, height, png = render_pdf_page_bytes(
        library, data, page_index, dpi, max_pixels, password, color_mode == "RGB"
    )
    return RenderedPage(width, height, png)


def recover_pdf_table(source, bounds, *, page_index=0, pdfium_library=None, password=None):
    """恢复指定页上已知区域的物理网格，bounds 为旋转裁剪后左上原点 point。

    这是区域恢复接口，尚不执行整页表格检测。不完整外边界或非矩形合并
    返回 RapidDocError，不以空表格冒充成功。
    """
    library, data = _prepare_pdf(source, pdfium_library, None)
    return RecoveredTable(
        **json.loads(recover_pdf_table_json(library, data, bounds, page_index, password))
    )


def capabilities():
    """Describe the production SDK; migration checkpoints are development-only."""
    from ._native import model_catalog_json

    return dict(
        stage="native_document_sdk",
        formats=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
        ocr=True,
        layout=True,
        tables=True,
        images=True,
        page_render=True,
        drawing=["layout", "span", "line_sort"],
        parse_methods=["auto", "txt", "ocr"],
        deferred=["enhanced_reading_order", "pp_doclayout_plus_l_pdf"],
        models=json.loads(model_catalog_json()),
    )
