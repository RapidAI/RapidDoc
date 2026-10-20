"""DOCX 公式与图片资源处理；共享当前 Converter 的单文档状态。"""

import hashlib
import re
from typing import Any
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.paragraph import Paragraph
from loguru import logger
from ..ooxml_chart import extract_chart_html_from_ooxml
from ..image import serialize_office_image
from ..equation.ooxml import is_mathtype_equation_prog_id
from ..equation.omml import oMath2Latex
from .....schema import BlockType

from .context import _DocxConstants


class _DocxResources:
    """集中维护公式与图片资源，不自行创建文档或持有跨文档缓存。"""

    def _decode_docx_ole_equation(
        self,
        ole_element: Any,
        part: Any,
    ) -> str | None:
        """从当前 DOCX part 的内部 OLE relationship 解码公式对象。"""

        prog_id = ole_element.get("ProgID") or ole_element.get("ProgId")
        if not is_mathtype_equation_prog_id(prog_id):
            return None
        object_type = (ole_element.get("Type") or "Embed").strip().casefold()
        if object_type != "embed":
            return None
        draw_aspect = (ole_element.get("DrawAspect") or "Content").strip().casefold()
        if draw_aspect == "icon":
            return None

        relationship_id = ole_element.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relationships = getattr(part, "rels", None)
        relationship = relationships.get(relationship_id) if relationships is not None else None
        if relationship is None or getattr(relationship, "is_external", False):
            return None
        reltype = str(getattr(relationship, "reltype", ""))
        if not reltype.rstrip("/").casefold().endswith("/oleobject"):
            return None
        try:
            blob = relationship.target_part.blob
        except (AttributeError, KeyError, ValueError):
            blob = None
        latex = self._ooxml_equation_decoder.decode(
            blob,
            prog_id=prog_id,
        )
        if latex is not None:
            return latex

        warning_key = (self._docx_part_key(part), str(relationship_id))
        if warning_key not in self._mtef_warned_relations:
            self._mtef_warned_relations.add(warning_key)
            logger.warning(
                "DOCX_MTEF_FALLBACK: part={!r}, relationship={!r} has an invalid or unsupported equation OLE object",
                warning_key[0],
                relationship_id,
            )
        return None

    def _decode_docx_equationxml(
        self,
        shape_element: Any,
        part: Any,
    ) -> str | None:
        """解码当前 VML shape 的 ``equationxml`` 并对失败告警去重。"""

        vml_shape_tag = f"{{{_DocxConstants._BLIP_NAMESPACES['v']}}}shape"
        if getattr(shape_element, "tag", None) != vml_shape_tag:
            return None
        equation_xml = shape_element.get("equationxml")
        if equation_xml is None:
            return None

        latex = self._equationxml_decoder.decode(equation_xml)
        if latex is not None:
            return latex

        digest = hashlib.sha256(equation_xml.encode("utf-8", errors="replace")).hexdigest()[:16]
        warning_key = (
            self._docx_part_key(part),
            str(shape_element.get("id") or ""),
            digest,
        )
        if warning_key not in self._equationxml_warned_shapes:
            self._equationxml_warned_shapes.add(warning_key)
            logger.warning(
                "DOCX_EQUATIONXML_FALLBACK: part={!r}, shape_id={!r}, payload_sha256={!r} is malformed or unsupported",
                warning_key[0],
                warning_key[1],
                warning_key[2],
            )
        return None

    @staticmethod
    def _docx_image_relationship_id(image: Any) -> str | None:
        """读取 DrawingML/VML 图片元素的内部 relationship id。"""

        relationship_id = image.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        if not relationship_id:
            relationship_id = image.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        return str(relationship_id) if relationship_id else None

    @classmethod
    def _docx_image_part(cls, image: Any, part: Any) -> Any | None:
        """通过当前 part 的内部关系解析图片 part。"""

        relationship_id = cls._docx_image_relationship_id(image)
        relationships = getattr(part, "rels", None)
        if not relationship_id or relationships is None:
            return None
        relationship = relationships.get(relationship_id)
        if relationship is None or getattr(relationship, "is_external", False):
            return None
        try:
            return relationship.target_part
        except (AttributeError, KeyError, ValueError):
            return None

    def _decode_docx_image_equation(
        self,
        image: Any,
        part: Any,
    ) -> str | None:
        """从当前图片 part 的 WMF/GIF comment 解码 MTEF。"""

        image_part = self._docx_image_part(image, part)
        if image_part is None:
            return None
        try:
            blob = image_part.blob
        except (AttributeError, KeyError, ValueError):
            return None
        return self._image_equation_decoder.decode(
            blob,
            part_name=getattr(image_part, "partname", None),
            content_type=getattr(image_part, "content_type", None),
        )

    @staticmethod
    def _select_docx_compatibility_tokens(
        token_groups: list[list[tuple[str, str]]],
    ) -> list[tuple[str, str]]:
        """按 OMML、Equation XML、OLE MTEF、图片 MTEF 选择兼容分支。"""

        for token_kind in _DocxConstants._FORMULA_SOURCE_PRIORITY:
            for tokens in token_groups:
                if any(kind == token_kind for kind, _value in tokens):
                    return tokens
        return token_groups[0] if token_groups else []

    def _docx_formula_tokens(
        self,
        element: Any,
        part: Any,
    ) -> list[tuple[str, str]]:
        """按文档顺序提取文本、OMML、Equation XML 和 MTEF。"""

        tag_name = self._local_name(element)
        if tag_name is None:
            return []
        tag = str(getattr(element, "tag", ""))
        word_namespace = _DocxConstants._BLIP_NAMESPACES["w"]

        if tag_name == "AlternateContent":
            branch_tokens = [
                self._docx_formula_tokens(child, part) for child in element if self._local_name(child) in {"Choice", "Fallback"}
            ]
            return self._select_docx_compatibility_tokens(branch_tokens)

        if tag_name == "object" and tag == f"{{{_DocxConstants._BLIP_NAMESPACES['w']}}}object":
            child_tokens = [self._docx_formula_tokens(child, part) for child in element]
            return self._select_docx_compatibility_tokens(child_tokens)

        if tag_name == "txbxContent":
            # 外层段落会单独遍历文本框内容，避免在此重复提取公式和文字。
            return []

        if tag_name == "oMath" and "officeDocument/2006/math" in tag:
            try:
                latex = str(oMath2Latex(element)).strip()
            except Exception as exc:
                logger.debug(f"Failed to convert DOCX OMML equation to LaTeX: {exc}")
                return []
            return [("omml", latex)] if latex else []

        if tag_name == "shape" and tag == f"{{{_DocxConstants._BLIP_NAMESPACES['v']}}}shape":
            latex = self._decode_docx_equationxml(element, part)
            if latex is not None:
                return [("equationxml", latex)]

        if tag_name == "OLEObject":
            latex = self._decode_docx_ole_equation(element, part)
            return [("mtef", latex)] if latex else []

        if (tag_name == "blip" and tag == "{http://schemas.openxmlformats.org/drawingml/2006/main}blip") or (
            tag_name == "imagedata" and tag == f"{{{_DocxConstants._BLIP_NAMESPACES['v']}}}imagedata"
        ):
            latex = self._decode_docx_image_equation(element, part)
            return [("image_mtef", latex)] if latex else []

        if tag_name == "t" and "officeDocument/2006/math" not in tag:
            return [("text", element.text)] if isinstance(element.text, str) else []
        if tag in {f"{{{word_namespace}}}tab", f"{{{word_namespace}}}ptab"}:
            return [("text", "\t")]
        if tag == f"{{{word_namespace}}}cr":
            return [("text", "\n")]
        if tag == f"{{{word_namespace}}}br":
            break_type = element.get(f"{{{word_namespace}}}type")
            return [("text", "\n")] if break_type in {None, "textWrapping"} else []
        if tag == f"{{{word_namespace}}}noBreakHyphen":
            return [("text", "-")]

        tokens: list[tuple[str, str]] = []
        for child in element:
            tokens.extend(self._docx_formula_tokens(child, part))
        return tokens

    def _picture_is_equation_preview(self, image: Any, part: Any) -> bool:
        """判断图片是否属于同一容器中已恢复的公式预览。"""

        if self._decode_docx_image_equation(image, part):
            return True

        for ancestor in image.iterancestors():
            ancestor_name = self._local_name(ancestor)
            if ancestor_name == "shape":
                if self._decode_docx_equationxml(ancestor, part):
                    return True
                continue
            if ancestor_name in {"object", "AlternateContent"}:
                tokens = self._docx_formula_tokens(ancestor, part)
                if any(kind in _DocxConstants._FORMULA_TOKEN_KINDS for kind, _value in tokens):
                    return True
                continue
        return False

    def _handle_pictures(
        self,
        picture_refs: Any,
        *,
        part: Any | None = None,
    ) -> None:
        """
        处理图片。

        Args:
            picture_refs: 图片引用元素列表

        Returns:

        """

        source_part = part or self._require_document_part()

        seen_rel_ids: set[str] = set()
        # 遍历所有图片引用元素，支持 DrawingML blip 和 VML imagedata。
        for image in picture_refs:
            if self._picture_is_equation_preview(image, source_part):
                continue
            rel_id = self._docx_image_relationship_id(image)
            if rel_id and rel_id in seen_rel_ids:
                continue
            if rel_id:
                seen_rel_ids.add(rel_id)
            image_part = self._docx_image_part(image, source_part)
            if image_part is None:
                logger.warning("Warning: image cannot be found")
                continue

            img_base64 = serialize_office_image(
                image_part.blob,
                part_name=getattr(image_part, "partname", None),
                content_type=getattr(image_part, "content_type", None),
            )
            if img_base64 is None:
                continue

            image_block = {
                "type": BlockType.IMAGE,
                "image_base64": img_base64,
            }
            alt_text = self._docx_image_alt_text(image)
            if alt_text:
                image_block["content"] = alt_text
            self.cur_page.append(image_block)

    @staticmethod
    def _docx_image_alt_text(image: Any) -> str:
        """读取 DrawingML wp:docPr 的 descr 或 VML v:shape 的 alt 替代文本。"""
        element = image
        while element is not None:
            tag = str(getattr(element, "tag", ""))
            local_name = tag.rsplit("}", 1)[-1]
            if local_name in {"inline", "anchor"}:
                for child in element:
                    child_tag = str(getattr(child, "tag", ""))
                    if child_tag.rsplit("}", 1)[-1] == "docPr":
                        descr = (child.get("descr") or "").strip()
                        if descr:
                            return descr
                return ""
            if local_name == "shape" and "urn:schemas-microsoft-com:vml" in tag:
                alt = (element.get("alt") or "").strip()
                if alt:
                    return alt
                return ""
            element = element.getparent()
        return ""

    def _handle_drawingml(self, elements: list[BaseOxmlElement]):
        """
        处理 DrawingML 元素，目前先处理 chart 元素。

        Args:
            elements: 包含 DrawingML 元素的列表

        Returns:

        """
        chart_rel_types = {
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart",
            "http://purl.oclc.org/ooxml/officeDocument/relationships/chart",
        }
        package_rel_types = {
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/package",
            "http://purl.oclc.org/ooxml/officeDocument/relationships/package",
        }
        rel_id_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for element in elements:
            chart = element.find(".//c:chart", namespaces=_DocxConstants._BLIP_NAMESPACES)
            if chart is None:
                continue

            chart_block = {
                "type": BlockType.CHART,
                "content": "",
            }
            self.cur_page.append(chart_block)

            rel_id = chart.get(rel_id_attr)
            if not rel_id:
                continue

            try:
                chart_rel = self.docx_obj.part.rels[rel_id]
            except KeyError:
                continue

            if chart_rel.reltype not in chart_rel_types:
                continue

            try:
                chart_part = chart_rel.target_part
                chart_xml = chart_part.blob
            except Exception as e:
                logger.warning(f"Warning: chart XML cannot be loaded: {e}")
                continue

            workbook_bytes = None
            try:
                for rel in chart_part.rels.values():
                    if rel.reltype in package_rel_types:
                        workbook_bytes = rel.target_part.blob
                        break
            except Exception as e:
                logger.warning(f"Warning: chart workbook cannot be loaded: {e}")

            try:
                chart_html = extract_chart_html_from_ooxml(chart_xml, workbook_bytes)
            except Exception as e:
                logger.warning(f"Warning: chart HTML cannot be extracted: {e}")
                continue
            if chart_html:
                chart_block["content"] = chart_html

    def _handle_textbox_content(
        self,
        textbox_elements: list,
    ):
        """
        处理文本框内容并将其添加到文档结构。
        """
        # 收集并组织段落
        container_paragraphs = self._collect_textbox_paragraphs(textbox_elements)

        # 处理所有段落
        all_paragraphs = []

        # 对每个容器内的段落进行排序，然后按容器顺序处理
        for paragraphs in container_paragraphs.values():
            # 按容器内的垂直位置进行排序
            sorted_container_paragraphs = sorted(
                paragraphs,
                key=lambda x: (
                    x[1] is None,
                    x[1] if x[1] is not None else float("inf"),
                ),
            )

            # 将排序后的段落添加到待处理列表
            all_paragraphs.extend(sorted_container_paragraphs)

        # 跟踪已处理段落以避免重复（相同内容和位置）
        processed_paragraphs = set()

        # 处理所有段落
        for p, position in all_paragraphs:
            # 创建 Paragraph 对象以获取文本内容
            paragraph = Paragraph(p, self.docx_obj)
            text_content = self._get_paragraph_text(paragraph)

            # 基于内容和位置创建唯一标识
            paragraph_id = (text_content, position)

            # 如果该段落（相同内容和位置）已处理，则跳过
            if paragraph_id in processed_paragraphs:
                logger.debug(f"Skipping duplicate paragraph: content='{text_content[:50]}...', position={position}")
                continue

            # 将该段落标记为已处理
            processed_paragraphs.add(paragraph_id)

            self._handle_text_elements(p)
        return

    def _collect_textbox_paragraphs(self, textbox_elements):
        """
        从文本框元素中收集并组织段落。
        """
        processed_paragraphs = []
        container_paragraphs = {}

        for element in textbox_elements:
            element_id = id(element)
            # 如果已处理相同元素，则跳过
            if element_id in processed_paragraphs:
                continue

            tag_name = self._local_name(element)
            if tag_name is None:
                continue
            processed_paragraphs.append(element_id)

            # 处理直接找到的段落（VML 文本框）
            if tag_name == "p":
                # 查找包含该段落的文本框或形状元素
                container_id = None
                for ancestor in element.iterancestors():
                    if any(ns in ancestor.tag for ns in ["textbox", "shape", "txbx"]):
                        container_id = id(ancestor)
                        break

                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []
                container_paragraphs[container_id].append((element, self._get_paragraph_position(element)))

            # 处理 txbxContent 元素（Word DrawingML 文本框）
            elif tag_name == "txbxContent":
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append((p, self._get_paragraph_position(p)))
            else:
                # 尝试从未知元素中提取任何段落
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append((p, self._get_paragraph_position(p)))

        return container_paragraphs

    def _get_paragraph_position(self, paragraph_element):
        """
        从段落元素提取垂直位置信息。
        """
        # 先尝试直接从包含顺序相关属性的 w:p 元素获取索引
        if hasattr(paragraph_element, "getparent") and paragraph_element.getparent() is not None:
            parent = paragraph_element.getparent()
            # 获取所有段落兄弟节点
            paragraphs = [p for p in parent.getchildren() if self._local_name(p) == "p"]
            # 查找当前段落在其兄弟节点中的索引
            try:
                paragraph_index = paragraphs.index(paragraph_element)
                return paragraph_index  # 使用索引作为位置以保证一致的排序
            except ValueError:
                pass

        # 在元素及其祖先中查找位置提示属性
        for elem in (*[paragraph_element], *paragraph_element.iterancestors()):
            # 检查直接的位置信息属性
            for attr_name in ["y", "top", "positionY", "y-position", "position"]:
                value = elem.get(attr_name)
                if value:
                    try:
                        # 移除任何非数字字符（如 'pt', 'px' 等）
                        clean_value = re.sub(r"[^0-9.]", "", value)
                        if clean_value:
                            return float(clean_value)
                    except (ValueError, TypeError):
                        pass

            # 检查 transform 属性中的位移信息
            transform = elem.get("transform")
            if transform:
                # 从 transform 矩阵中提取 translate 的第二个参数
                match = re.search(r"translate\([^,]+,\s*([0-9.]+)", transform)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

            # 检查 Word 格式中的锚点或相对位置指示器
            # 'dist' 类属性可以表示相对位置
            for attr_name in ["distT", "distB", "anchor", "relativeFrom"]:
                if elem.get(attr_name) is not None:
                    return elem.sourceline  # 使用 XML 源行号作为回退

        # 针对 VML 形状，查找特定属性
        for ns_uri in paragraph_element.nsmap.values():
            if "vml" in ns_uri:
                # 尝试从 style 属性提取 top 值
                style = paragraph_element.get("style")
                if style:
                    match = re.search(r"top:([0-9.]+)pt", style)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            pass

        # 如果没有更好的位置指示，则使用 XML 源行号作为顺序的代理
        return paragraph_element.sourceline if hasattr(paragraph_element, "sourceline") else None
