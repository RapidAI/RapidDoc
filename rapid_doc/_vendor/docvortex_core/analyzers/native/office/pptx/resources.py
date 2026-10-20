"""PPTX 图片、图表与公式资源，复用当前转换器的单文档状态。"""

from typing import Any, Optional
from pptx.oxml.text import CT_TextLineBreak
from ..rich_text import OfficeRichTextSegment, build_rich_text_html_from_segments
from loguru import logger
from pptx.enum.shapes import MSO_SHAPE_TYPE
from ..ooxml_chart import extract_chart_html_from_ooxml
from ..image import serialize_office_image
from ..equation.ooxml import is_mathtype_equation_prog_id
from .....schema import BlockType

from .context import DRAWINGML_NS, RELATIONSHIP_NS, SVG_BLIP_NS, OMML_NS


class _PptxResources:
    """集中维护图片、图表与公式资源，不改变文档生命周期和公开入口。"""

    @staticmethod
    def _pptx_ole_format(shape: Any) -> Any | None:
        """安全返回 PPTX graphic-frame 的 OLE format。"""

        try:
            return shape.ole_format
        except (AttributeError, KeyError, ValueError):
            return None

    def _is_equation_ole_shape(self, shape: Any) -> bool:
        """判断 PPTX OLE shape 的 ProgID 是否为 MathType/Equation 公式。"""

        ole_format = self._pptx_ole_format(shape)
        return bool(ole_format is not None and is_mathtype_equation_prog_id(getattr(ole_format, "prog_id", None)))

    def _decode_pptx_ole_equation(self, shape: Any) -> str | None:
        """解码 PPTX 内嵌公式 OLE；链接、图标和坏对象返回空。"""

        ole_format = self._pptx_ole_format(shape)
        if ole_format is None:
            return None
        prog_id = getattr(ole_format, "prog_id", None)
        if not is_mathtype_equation_prog_id(prog_id):
            return None
        if bool(getattr(ole_format, "show_as_icon", False)):
            return None
        if self._safe_shape_type(shape) != MSO_SHAPE_TYPE.EMBEDDED_OLE_OBJECT:
            return None
        try:
            blob = ole_format.blob
        except (AttributeError, KeyError, ValueError):
            blob = None
        latex = self._ooxml_equation_decoder.decode(
            blob,
            prog_id=prog_id,
        )
        if latex is not None:
            return latex

        warning_key = (
            str(getattr(getattr(shape, "part", None), "partname", "")),
            getattr(shape, "shape_id", None),
        )
        if warning_key not in self._mtef_warned_shapes:
            self._mtef_warned_shapes.add(warning_key)
            logger.warning(
                "PPTX_MTEF_FALLBACK: part={!r}, shape_id={!r} has an invalid or unsupported equation OLE object",
                warning_key[0],
                warning_key[1],
            )
        return None

    def _pptx_ole_shape_omml(self, shape: Any) -> list[str]:
        """提取 OLE 兼容容器内可表达的 OMML，作为 MTEF 的高优先级分支。"""

        element = getattr(shape, "_element", None)
        if element is None:
            return []
        equations: list[str] = []
        for math in element.findall(f".//{{{OMML_NS}}}oMath"):
            latex = self._convert_math_node_to_latex(math)
            if latex:
                equations.append(latex)
        return equations

    def _table_paragraph_html(self, paragraph: Any, shape: Any) -> str:
        """按 XML 顺序恢复表格中的样式、换行、公式与对应 run 的链接。"""
        runs = {id(run._r): run for run in paragraph.runs}
        font_sources = self._get_paragraph_font_sources(shape, paragraph)
        parts: list[str] = []
        for node in paragraph._element.content_children:
            if isinstance(node, CT_TextLineBreak):
                parts.append("<br>")
            elif self._is_math_content_node(node):
                latex = self._convert_math_node_to_latex(node)
                if latex:
                    parts.append(f"<eq>{latex}</eq>")
            else:
                text = getattr(node, "text", None)
                if not text:
                    continue
                run = runs.get(id(node))
                parts.append(
                    build_rich_text_html_from_segments(
                        [
                            OfficeRichTextSegment(
                                text,
                                self._get_style_str_from_run(run, font_sources) if run is not None else None,
                                self._resolve_hyperlink_from_run(run, shape) if run is not None else None,
                            )
                        ]
                    )
                )
        return "".join(parts)

    def _handle_tables(self, shape):
        """将PowerPoint表格转换为HTML格式。

        Args:
            shape: 包含表格的形状对象。
            parent_slide: 父幻灯片组。
            slide_ind: 当前幻灯片索引。
            doc: 文档对象(此实现中未使用)。
            slide_size: 幻灯片尺寸。

        Returns:
            str: 表格的HTML字符串，如果没有表格则返回None。
        """
        if not shape.has_table:
            return None

        table = shape.table
        table_xml = shape._element

        # 开始构建HTML表格
        html_parts = ['<table border="1">']

        # 跟踪已被合并单元格占用的位置
        # 格式: {(row, col): True}
        occupied_cells = {}

        for row_idx, row in enumerate(table.rows):
            html_parts.append("  <tr>")

            for col_idx, cell in enumerate(row.cells):
                # 跳过被合并占用的单元格
                if (row_idx, col_idx) in occupied_cells:
                    continue
                # 获取单元格XML以读取跨度信息
                cell_xml = table_xml.xpath(f".//a:tbl/a:tr[{row_idx + 1}]/a:tc[{col_idx + 1}]")

                if not cell_xml:
                    continue

                cell_xml = cell_xml[0]

                # 解析行跨度和列跨度
                row_span = cell_xml.get("rowSpan")
                col_span = cell_xml.get("gridSpan")

                row_span = int(row_span) if row_span else 1
                col_span = int(col_span) if col_span else 1

                # 标记被此单元格占用的位置
                for r in range(row_idx, row_idx + row_span):
                    for c in range(col_idx, col_idx + col_span):
                        if (r, c) != (row_idx, col_idx):
                            occupied_cells[(r, c)] = True

                # 确定标签类型：第一行使用<th>，其他使用<td>
                tag = "th" if row_idx == 0 else "td"

                # 构建属性字符串
                attrs = []
                if row_span > 1:
                    attrs.append(f'rowspan="{row_span}"')
                if col_span > 1:
                    attrs.append(f'colspan="{col_span}"')

                attr_str = " " + " ".join(attrs) if attrs else ""

                # 获取单元格文本内容
                cell_text = "<br>".join(
                    self._table_paragraph_html(paragraph, shape) for paragraph in cell.text_frame.paragraphs
                ).strip()

                html_parts.append(f"    <{tag}{attr_str}>{cell_text}</{tag}>")

            html_parts.append("  </tr>")

        html_parts.append("</table>")

        self.cur_page.append(
            {
                "type": BlockType.TABLE,
                "content": "\n".join(html_parts),
            }
        )

        return None

    def _handle_chart(self, shape) -> None:
        """按原有图片、图表与公式资源规则执行 _handle_chart，保持输入顺序与降级行为。"""
        try:
            chart_part = shape.chart.part
            chart_xml = chart_part.blob
        except Exception as e:
            logger.warning(f"Warning: chart XML cannot be loaded: {e}")
            return

        workbook_bytes = None
        try:
            chart_workbook = chart_part.chart_workbook
            xlsx_part = chart_workbook.xlsx_part
            if xlsx_part is not None:
                workbook_bytes = xlsx_part.blob
        except Exception as e:
            logger.warning(f"Warning: chart workbook cannot be loaded: {e}")

        try:
            chart_html = extract_chart_html_from_ooxml(chart_xml, workbook_bytes)
        except Exception as e:
            logger.warning(f"Warning: chart HTML cannot be extracted: {e}")
            return

        if not chart_html:
            return

        self.cur_page.append(
            {
                "type": BlockType.CHART,
                "content": chart_html,
            }
        )

    def _handle_pictures(self, shape):
        """按原有图片、图表与公式资源规则执行 _handle_pictures，保持输入顺序与降级行为。"""
        image_data = self._get_shape_image_data(shape)
        if image_data is None:
            return

        image_bytes, content_type = image_data

        latex = self._image_equation_decoder.decode(
            image_bytes,
            content_type=content_type,
        )
        if latex:
            self.cur_page.append(
                {
                    "type": BlockType.EQUATION,
                    "content": latex,
                }
            )
            return

        if content_type == "image/svg+xml":
            img_base64 = serialize_office_image(image_bytes, content_type=content_type)
            if img_base64 is None:
                # SVG 光栅化失败时回退到 PowerPoint 为 SVG 原生伴随存储的栅量 blip。
                fallback_data = self._get_shape_image_data(shape, include_svg=False)
                if fallback_data is not None and fallback_data[1] != "image/svg+xml":
                    img_base64 = serialize_office_image(fallback_data[0], content_type=fallback_data[1])
            if img_base64 is not None:
                self.cur_page.append(self._build_image_block(img_base64, shape))
            return

        img_base64 = serialize_office_image(
            image_bytes,
            content_type=content_type,
        )
        if img_base64 is None:
            return

        self.cur_page.append(self._build_image_block(img_base64, shape))

    def _build_image_block(self, img_base64: str, shape) -> dict:
        """构造图片块，形状 cNvPr 的 descr（替代文本）作为识别内容输出。"""
        image_block = {
            "type": BlockType.IMAGE,
            "image_base64": img_base64,
        }
        alt_text = self._pptx_shape_alt_text(shape)
        if alt_text:
            image_block["content"] = alt_text
        return image_block

    @staticmethod
    def _pptx_shape_alt_text(shape) -> str:
        """读取形状非可视属性 cNvPr 上的 descr 替代文本，兼容各形状包装元素。"""
        element = getattr(shape, "_element", None)
        if element is None:
            element = getattr(shape, "element", None)
        if element is None:
            return ""
        for candidate in element.iter():
            tag = str(candidate.tag)
            if tag.endswith("}cNvPr") or tag == "cNvPr":
                descr = (candidate.get("descr") or "").strip()
                if descr:
                    return descr
        return ""

    def _decode_pptx_shape_image_equation(self, shape: Any) -> str | None:
        """从普通 picture 或 OLE preview 的 WMF/GIF comment 恢复公式。"""

        image_data = self._get_shape_image_data(shape)
        if image_data is None:
            return None
        image_bytes, content_type = image_data
        return self._image_equation_decoder.decode(
            image_bytes,
            content_type=content_type,
        )

    @staticmethod
    def _find_first_embedded_image_rid(shape, *, include_svg: bool = True) -> Optional[str]:
        """按原有图片、图表与公式资源规则执行 _find_first_embedded_image_rid，保持输入顺序与降级行为。"""
        if include_svg:
            svg_blips = shape._element.findall(f".//{{{SVG_BLIP_NS}}}svgBlip")
            for svg_blip in svg_blips:
                relationship_id = svg_blip.get(f"{{{RELATIONSHIP_NS}}}embed")
                if relationship_id:
                    return relationship_id

        blips = shape._element.findall(f".//{{{DRAWINGML_NS}}}blip")
        for blip in blips:
            relationship_id = blip.get(f"{{{RELATIONSHIP_NS}}}embed")
            if relationship_id:
                return relationship_id

        return None

    @staticmethod
    def _has_blip_without_relationship(shape) -> bool:
        """判断图片节点是否只有空blip，避免把空fallback误报为图片资源缺失。"""
        if not hasattr(shape, "_element"):
            return False

        blips = [
            *shape._element.findall(f".//{{{SVG_BLIP_NS}}}svgBlip"),
            *shape._element.findall(f".//{{{DRAWINGML_NS}}}blip"),
        ]
        if not blips:
            return False

        for blip in blips:
            if blip.get(f"{{{RELATIONSHIP_NS}}}embed") or blip.get(f"{{{RELATIONSHIP_NS}}}link"):
                return False
        return True

    def _get_shape_image_data(self, shape, *, include_svg: bool = True) -> Optional[tuple[bytes, Optional[str]]]:
        """按原有图片、图表与公式资源规则执行 _get_shape_image_data，保持输入顺序与降级行为。"""
        relationship_id = None
        if hasattr(shape, "_element"):
            relationship_id = self._find_first_embedded_image_rid(shape, include_svg=include_svg)

        if relationship_id:
            try:
                image_part = shape.part.related_part(relationship_id)
                image_bytes = image_part.blob
            except Exception as e:
                logger.warning(f"Warning: embedded image relation {relationship_id} cannot be loaded: {e}")
            else:
                return image_bytes, getattr(image_part, "content_type", None)

        if self._has_blip_without_relationship(shape):
            logger.debug("Skipping PPTX picture with empty blip and no image relation")
            return None

        try:
            image = shape.image
        except KeyError as e:
            logger.warning(f"Warning: shape image relation cannot be loaded: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Warning: shape image cannot be loaded: {e}")
            return None
        except AttributeError:
            return None

        return image.blob, None
