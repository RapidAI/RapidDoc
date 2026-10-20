from io import BytesIO
from typing import Any, BinaryIO, Optional

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.paragraph import Paragraph
from loguru import logger
from lxml import etree

from ..equation.ooxml import OoxmlEquationDecoder
from ..equation.image import OfficeImageEquationDecoder
from .....schema import RAW_CAPTION
from .equationxml import DocxEquationXmlDecoder
from .package_normalizer import normalize_docx_package
from .....schema import BlockType
from .....content.spans import (
    append_text_span,
    extend_inline_spans,
    inline_span_plain_text,
    text_spans,
)


from .context import (
    _DocxConstants,
    _DocxComplexFieldFrame as _DocxComplexFieldFrame,
    _ParagraphElement as _ParagraphElement,
    _ParagraphHyperlink as _ParagraphHyperlink,
)
from .resources import _DocxResources
from .styles import _DocxStyles
from .numbering import _DocxNumbering
from .fields import _DocxFields
from .tables import _DocxTables


class DocxConverter(_DocxConstants, _DocxResources, _DocxStyles, _DocxNumbering, _DocxFields, _DocxTables):
    """编排 DOCX 文档生命周期、段落遍历与各项职责处理。"""

    def __init__(self):
        """配置固定 XML 能力并建立本次文档转换的独立状态。"""
        self.XML_KEY = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
        self.xml_namespaces = {"w": "http://schemas.microsoft.com/office/word/2003/wordml"}
        self.picture_xpath_expr = etree.XPath(".//a:blip | .//v:imagedata", namespaces=DocxConverter._BLIP_NAMESPACES)
        self.equation_bookends: str = "<eq>{EQ}</eq>"  # 公式标记格式
        self._reset_document_state()

    def _reset_document_state(self) -> None:
        """每次转换重新创建可变状态及解码器，避免失败或复用实例残留旧文档。"""
        self._close_mammoth_fallback_context()
        self._fallback_docx_bytes: Optional[bytes] = None  # 仅转换期间保留规范化后的原包
        self.docx_obj = None
        self.pages = []
        self.cur_page = []
        self._mammoth_tables_html: list = []  # 与正文顶层表格对齐的 mammoth 预解析 HTML，None 表示回退解析
        self._mammoth_table_idx: int = 0  # 当前预解析表格游标
        self.pre_num_id: int = -1  # 上一个处理元素的 numId
        self.pre_ilevel: int = -1  # 上一个处理元素的缩进等级, 用于判断列表层级
        self.list_block_stack: list = []  # 列表块堆栈
        self.list_counters: dict[tuple[int, int], int] = {}  # 列表计数器 (numId, ilvl) -> count
        self.index_block_stack: list = []  # 目录索引块堆栈
        self.pre_index_ilevel: int = -1  # 上一个目录项的缩进等级
        self.plain_toc_base_level: Optional[int] = None  # 普通目录段落的起始层级
        self.heading_list_numids: set = set()  # 用作章节标题的列表numId集合
        self.processed_textbox_elements: list = []
        self.toc_anchor_set: set[str] = set()  # TOC 超链接目标锚点集合
        self.toc_anchor_aliases: dict[str, str] = {}  # 同一正文段落内 TOC bookmark 到唯一公开 anchor 的映射
        self._numbering_root: Optional[BaseOxmlElement] = None
        self._numbering_root_loaded: bool = False
        self._numbering_level_cache: dict[tuple[int, int], Optional[BaseOxmlElement]] = {}
        self._numbering_start_cache: dict[tuple[int, int], int] = {}
        self._style_lookup_cache: dict[tuple[Any, Optional[str]], Any] = {}
        self._style_bool_cache: dict[tuple[int, str], Optional[bool]] = {}
        self._ooxml_equation_decoder = OoxmlEquationDecoder()
        self._mtef_warned_relations: set[tuple[str, str]] = set()
        self._equationxml_decoder = DocxEquationXmlDecoder()
        self._equationxml_warned_shapes: set[tuple[str, str, str]] = set()
        self._image_equation_decoder = OfficeImageEquationDecoder()

    @staticmethod
    def _local_name(element: Any) -> Optional[str]:
        """安全获取 XML 元素本地标签名，遇到注释或处理指令等非元素节点时返回 None。"""
        tag = getattr(element, "tag", None)
        if not isinstance(tag, str):
            return None
        try:
            return etree.QName(tag).localname
        except ValueError:
            return None

    def _reset_style_caches(self) -> None:
        """重置样式查询缓存，避免同一 converter 实例多次转换时复用旧文档样式。"""
        self._style_lookup_cache = {}
        self._style_bool_cache = {}

    @staticmethod
    def _docx_part_key(part: Any) -> str:
        """返回用于公式关系缓存和告警去重的 DOCX part 名称。"""

        return str(getattr(part, "partname", ""))

    def _require_document_part(self) -> Any:
        """返回已加载的主文档 part，生命周期异常时立即失败。"""

        if self.docx_obj is None:
            raise ValueError("DOCX document part is not initialized")
        return self.docx_obj.part

    def _sanitize_missing_internal_relationships(self, file_bytes: bytes) -> bytes:
        """规范化 DOCX 包，兼容缺失内部关系和损坏图片成员。"""
        return normalize_docx_package(file_bytes)

    def _start_new_page(self) -> None:
        self.cur_page = []
        self.pages.append(self.cur_page)

    def _is_layout_only_section_break(self, element: BaseOxmlElement) -> bool:
        w_ns = DocxConverter._BLIP_NAMESPACES["w"]
        p_pr = element.find(f"{{{w_ns}}}pPr")
        sect_pr = p_pr.find(f"{{{w_ns}}}sectPr") if p_pr is not None else None
        if sect_pr is None:
            return False

        paragraph = Paragraph(element, self.docx_obj)
        if self._get_paragraph_text(paragraph).strip():
            return False

        if self.picture_xpath_expr(element):
            return False

        sect_type = sect_pr.find(f"{{{w_ns}}}type")
        sect_val = sect_type.get(f"{{{w_ns}}}val", "continuous") if sect_type is not None else "continuous"
        if sect_val != "continuous":
            return False

        pg_mar = sect_pr.find(f"{{{w_ns}}}pgMar")
        if pg_mar is None:
            return False

        for attr in ("header", "footer", "top", "bottom", "left", "right"):
            if pg_mar.get(f"{{{w_ns}}}{attr}", "0") != "0":
                return False
        return True

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        """重置文档状态后按原顺序读取、预解析表格并遍历正文。"""
        self._reset_document_state()
        # 读取文件字节，以便 mammoth 和 python-docx 各自使用独立读取流
        file_bytes = self._sanitize_missing_internal_relationships(file_stream.read())
        self._fallback_docx_bytes = file_bytes
        try:
            # 使用完整 DOCX 上下文预解析顶层表格，避免转换非表格正文带来的资源浪费
            self._mammoth_tables_html = self._preparse_tables_with_mammoth(file_bytes)
            self._mammoth_table_idx = 0
            self.docx_obj = Document(BytesIO(file_bytes))
            self.toc_anchor_set = self._collect_toc_anchor_set()
            self.toc_anchor_aliases = self._collect_toc_anchor_aliases(
                self.toc_anchor_set,
            )
            # 预扫描文档，识别用作章节标题的列表numId
            self.heading_list_numids = self._detect_heading_list_numids()
            self.pages.append(self.cur_page)
            self._walk_linear(self.docx_obj.element.body)
            self._add_header_footer(self.docx_obj)
        finally:
            self._close_mammoth_fallback_context()
            self._fallback_docx_bytes = None

    def _close_active_list(self) -> None:
        """关闭当前活跃列表块，但保留 Word numId 的连续编号计数。"""
        self.pre_num_id = -1
        self.pre_ilevel = -1
        self.list_block_stack = []

    def _reset_index_state(self) -> None:
        """重置目录索引栈，避免相隔的多个目录块被错误合并。"""
        self.index_block_stack = []
        self.pre_index_ilevel = -1
        self.plain_toc_base_level = None

    def _walk_linear(
        self,
        body: BaseOxmlElement,
    ):
        for element in body:
            # 获取元素的标签名（去除命名空间前缀）
            tag_name = self._local_name(element)
            if tag_name is None:
                continue
            # 检查是否存在内联图像（blip元素）
            picture_refs = self.picture_xpath_expr(element)

            # 查找所有绘图元素（用于处理DrawingML）
            drawingml_els = element.findall(".//w:drawing", namespaces=DocxConverter._BLIP_NAMESPACES)
            if drawingml_els:
                self._handle_drawingml(drawingml_els)

            # 检查文本框内容（支持多种文本框格式）
            # 仅当该元素之前未被处理时才处理
            if element not in self.processed_textbox_elements:
                # 现代 Word 文本框
                txbx_xpath = etree.XPath(
                    ".//w:txbxContent|.//v:textbox//w:p",
                    namespaces=DocxConverter._BLIP_NAMESPACES,
                )
                textbox_elements = txbx_xpath(element)

                # 未找到现代文本框，检查替代/旧版文本框格式
                if not textbox_elements and tag_name in ["drawing", "pict"]:
                    # 额外检查 DrawingML 和 VML 格式中的文本框
                    alt_txbx_xpath = etree.XPath(
                        ".//wps:txbx//w:p|.//w10:wrap//w:p|.//a:p//a:t",
                        namespaces=DocxConverter._BLIP_NAMESPACES,
                    )
                    textbox_elements = alt_txbx_xpath(element)

                    # 检查不在标准文本框内的形状文本
                    if not textbox_elements:
                        shape_text_xpath = etree.XPath(
                            ".//a:bodyPr/ancestor::*//a:t|.//a:txBody//a:t",
                            namespaces=DocxConverter._BLIP_NAMESPACES,
                        )
                        shape_text_elements = shape_text_xpath(element)
                        if shape_text_elements:
                            # 从形状文本创建自定义文本元素
                            raw_text = " ".join([t.text for t in shape_text_elements if t.text])
                            text_content = self._normalize_text_block_content(text_spans(raw_text))
                            visible_text = inline_span_plain_text(text_content)
                            if visible_text:
                                logger.debug(f"Found shape text: {visible_text[:50]}...")
                                self.cur_page.append(
                                    {
                                        "type": BlockType.TEXT,
                                        "content": text_content,
                                    }
                                )
                if textbox_elements:
                    self.processed_textbox_elements.append(element)
                    for tb_element in textbox_elements:
                        self.processed_textbox_elements.append(tb_element)

                    logger.debug(f"Found textbox content with {len(textbox_elements)} elements")
                    self._handle_textbox_content(textbox_elements)

            if tag_name == "tbl":
                # 表格是顶层块级元素，会中断活跃列表的上下文。
                # 若不重置列表状态，后续列表项会被追加到表格之前创建的列表块中，
                # 导致表格在 cur_page 中出现在那些列表项之后，产生顺序错乱。
                if self.pre_num_id != -1:
                    self._close_active_list()
                try:
                    # 处理表格元素
                    self._handle_tables(element)
                except Exception as e:
                    # 表格解析失败会导致整表丢失，需以 warning 级别暴露异常详情。
                    logger.warning(f"Could not parse a table, broken docx table: {e}")
            # 检查图片元素
            elif picture_refs:
                # 判断图片是否为锚定（浮动）图片
                is_anchored = bool(
                    element.findall(
                        ".//wp:anchor",
                        namespaces=DocxConverter._BLIP_NAMESPACES,
                    )
                )
                # 锚定图片在段落中浮动定位，段落文本应出现在图片之前
                if is_anchored and tag_name == "p":
                    self._handle_text_elements(element)
                    self._handle_pictures(
                        picture_refs,
                        part=self._require_document_part(),
                    )
                else:
                    # 处理图片元素
                    self._handle_pictures(
                        picture_refs,
                        part=self._require_document_part(),
                    )
                    # 如果是段落元素，同时处理其中的文本内容（如描述性文字）
                    if tag_name == "p":
                        self._handle_text_elements(element)
            # 检查 sdt 元素
            elif tag_name == "sdt":
                sdt_content = element.find(".//w:sdtContent", namespaces=DocxConverter._BLIP_NAMESPACES)
                if sdt_content is not None:
                    if self._is_toc_sdt(element):
                        # 处理目录SDT，转换为INDEX块
                        self._handle_sdt_as_index(sdt_content)
                    else:
                        # 其他SDT元素，按普通文本处理
                        paragraphs = sdt_content.findall(".//w:p", namespaces=DocxConverter._BLIP_NAMESPACES)
                        for p in paragraphs:
                            self._handle_text_elements(p)
            # 检查文本段落元素
            elif tag_name == "p":
                # 处理文本元素（包括段落属性如"tcPr", "sectPr"等）
                self._handle_text_elements(element)

            # 忽略其他未知元素并记录日志
            else:
                logger.debug(f"Ignoring element in DOCX with tag: {tag_name}")

    def _handle_text_elements(
        self,
        element: BaseOxmlElement,
    ):
        """
        处理文本元素。

        Args:
            element: 元素对象
            doc: DoclingDocument 对象

        Returns:

        """
        is_section_end = False
        has_section_break = element.find(".//w:sectPr", namespaces=DocxConverter._BLIP_NAMESPACES) is not None
        if has_section_break and not self._is_layout_only_section_break(element):
            # 如果没有text内容
            if element.text == "":
                self._start_new_page()
            else:
                # 标记本节结束，处理完文本之后再分节
                is_section_end = True
        paragraph = Paragraph(element, self.docx_obj)
        paragraph_elements = self._get_paragraph_elements(paragraph)
        paragraph_text = self._get_paragraph_text(paragraph)
        paragraph_anchor = self._extract_paragraph_bookmark(element)
        text, equations = self._handle_equations_in_text(
            element=element,
            text=paragraph_text,
            part=paragraph.part,
        )

        if text is None:
            return None
        text = text.strip()

        if self._handle_plain_toc_paragraph_as_index(
            paragraph=paragraph,
            paragraph_element=element,
            paragraph_elements=paragraph_elements,
            text=text,
            equations=equations,
        ):
            # 普通 TOC 是列表边界，避免后续同 numId 列表项继续合并到目录前的列表块。
            if self.pre_num_id != -1:
                self._close_active_list()
            # 普通 TOC 段落被转换为 INDEX 后，也要保留段落末尾分节分页语义。
            if is_section_end:
                self._start_new_page()
            return None
        self._reset_index_state()

        # 常见的项目符号和编号列表样式。
        # "List Bullet", "List Number", "List Paragraph"
        # 识别列表是否为编号列表
        p_style_id, p_level = self._get_label_and_level(paragraph)
        p_style_id = p_style_id or "Normal"
        numid, ilevel = self._get_numId_and_ilvl(paragraph)

        if numid == 0:
            numid = None

        # 处理列表
        if numid is not None and ilevel is not None and p_style_id not in ["Title", "Heading"]:
            # 通过检查 numFmt 来确认这是否实际上是编号列表
            is_numbered = self._is_numbered_list(numid, ilevel)

            if numid in self.heading_list_numids:
                # 该列表被用作章节标题（列表项间穿插了正文内容），直接转换为title block
                # 先关闭任何活跃的普通列表
                if self.pre_num_id != -1:
                    self._close_active_list()
                content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
                if content_text:
                    title_block = {
                        "type": BlockType.PARAGRAPH_TITLE,
                        "level": ilevel + 2,
                        "is_numbered_style": is_numbered,
                        "content": content_text,
                    }
                    if paragraph_anchor:
                        title_block["anchor"] = paragraph_anchor
                    self.cur_page.append(title_block)
            else:
                self._add_list_item(
                    numid=numid,
                    ilevel=ilevel,
                    elements=paragraph_elements,
                    is_numbered=is_numbered,
                    text=text,
                    equations=equations,
                )
            # 列表项已处理，返回
            return None
        elif (  # 列表结束处理
            numid is None and self.pre_num_id != -1 and p_style_id not in ["Title", "Heading"]
        ):  # 关闭列表
            # 重置列表状态
            self._close_active_list()

        if p_style_id in ["Title"]:
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
            if content_text:
                title_block = {
                    "type": BlockType.DOC_TITLE,
                    "level": 1,
                    "content": content_text,
                }
                if paragraph_anchor:
                    title_block["anchor"] = paragraph_anchor
                self.cur_page.append(title_block)

        elif "Heading" in p_style_id:
            is_numbered_style = numid is not None and ilevel is not None and self._is_numbered_list(numid, ilevel)
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
            if content_text:
                h_block = {
                    "type": BlockType.PARAGRAPH_TITLE,
                    "level": max(p_level + 1, 2) if p_level is not None else 2,
                    "is_numbered_style": is_numbered_style,
                    "content": content_text,
                }
                if paragraph_anchor:
                    h_block["anchor"] = paragraph_anchor
                self.cur_page.append(h_block)

        elif equations:
            equation_values = [value for kind, value in equations if kind == "equation"]
            if (paragraph_text is None or len(paragraph_text.strip()) == 0) and equation_values:
                # 独立公式
                eq_block = {
                    "type": BlockType.EQUATION,
                    "content": equation_values[0] if len(equation_values) == 1 else "\n".join(equation_values),
                }
                self.cur_page.append(eq_block)
            else:
                # 包含行内公式的文本块，同时支持超链接
                content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
                content_text = self._normalize_text_block_content(content_text)
                if content_text:
                    text_with_inline_eq_block = {
                        "type": BlockType.TEXT,
                        "content": content_text,
                    }
                    if paragraph_anchor:
                        text_with_inline_eq_block["anchor"] = paragraph_anchor
                    self.cur_page.append(text_with_inline_eq_block)
        elif p_style_id in [
            "Paragraph",
            "Normal",
            "Subtitle",
            "Author",
            "DefaultText",
            "ListParagraph",
            "ListBullet",
            "Quote",
        ]:
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
            content_text = self._normalize_text_block_content(content_text)
            if content_text:
                text_block = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                if paragraph_anchor:
                    text_block["anchor"] = paragraph_anchor
                self.cur_page.append(text_block)
        # 判断是否是 Caption
        elif self._is_caption(element):
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
            if content_text:
                caption_block = {
                    "type": RAW_CAPTION,
                    "content": content_text,
                }
                if paragraph_anchor:
                    caption_block["anchor"] = paragraph_anchor
                self.cur_page.append(caption_block)
        else:
            # 文本样式名称不仅有默认值，还可能有用户自定义值
            # 因此我们将所有其他标签视为纯文本
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)
            content_text = self._normalize_text_block_content(content_text)
            if content_text:
                text_block = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                if paragraph_anchor:
                    text_block["anchor"] = paragraph_anchor
                self.cur_page.append(text_block)

        if is_section_end:
            self._start_new_page()

    def _process_header_footer_paragraph(self, paragraph: Paragraph) -> list[dict[str, Any]]:
        """
        处理页眉/页脚中的单个段落，支持行内公式和超链接。

        Args:
            paragraph: 段落对象

        Returns:
            list[dict]: 处理后的结构化 Span
        """
        paragraph_elements = self._get_paragraph_elements(paragraph)
        paragraph_text = self._get_paragraph_text(paragraph)
        text, equations = self._handle_equations_in_text(
            element=paragraph._element,
            text=paragraph_text,
            part=paragraph.part,
        )

        text = text.strip()
        if not text and not equations:
            return []

        # 构建包含公式和超链接的文本
        content_text = self._build_text_with_equations_and_hyperlinks(paragraph_elements, text, equations)

        return content_text

    def _add_header_footer(self, docx_obj: DocxDocument) -> None:
        """
        处理页眉和页脚，按照分节顺序添加到 pages 列表中，过滤掉空字符串和纯数字内容
        分为整个文档是否启用奇偶页不同和每一节是否启用首页不同两种情况，
        支持行内公式和超链接，并根据类型去重
        """
        is_odd_even_different = docx_obj.settings.odd_and_even_pages_header_footer
        for sec_idx, section in enumerate(docx_obj.sections):
            # 用于去重的集合
            added_headers = set()
            added_footers = set()

            hdrs = [section.header]
            if is_odd_even_different:
                hdrs.append(section.even_page_header)
            if section.different_first_page_header_footer:
                hdrs.append(section.first_page_header)
            for hdr in hdrs:
                # 处理每个段落，支持公式和超链接
                processed_parts: list[list[dict[str, Any]]] = []
                for par in hdr.paragraphs:
                    content = self._process_header_footer_paragraph(par)
                    if content:
                        processed_parts.append(content)
                spans: list[dict[str, Any]] = []
                for part in processed_parts:
                    if spans:
                        append_text_span(spans, " ")
                    extend_inline_spans(spans, part)
                visible = inline_span_plain_text(spans)
                if spans and not visible.isdigit() and visible not in added_headers:
                    added_headers.add(visible)
                    try:
                        self.pages[sec_idx].append(
                            {
                                "type": BlockType.HEADER,
                                "content": spans,
                            }
                        )
                    except IndexError:
                        logger.error("Section index out of range when adding header.")

            ftrs = [section.footer]
            if is_odd_even_different:
                ftrs.append(section.even_page_footer)
            if section.different_first_page_header_footer:
                ftrs.append(section.first_page_footer)
            for ftr in ftrs:
                # 处理每个段落，支持公式和超链接
                processed_parts = []
                for par in ftr.paragraphs:
                    content = self._process_header_footer_paragraph(par)
                    if content:
                        processed_parts.append(content)
                spans = []
                for part in processed_parts:
                    if spans:
                        append_text_span(spans, " ")
                    extend_inline_spans(spans, part)
                visible = inline_span_plain_text(spans)
                if spans and not visible.isdigit() and visible not in added_footers:
                    added_footers.add(visible)
                    try:
                        self.pages[sec_idx].append(
                            {
                                "type": BlockType.FOOTER,
                                "content": spans,
                            }
                        )
                    except IndexError:
                        logger.error("Section index out of range when adding footer.")

    def _is_caption(self, element: BaseOxmlElement) -> bool:
        """
        根据 insertText 中是否有 SEQ 字段来判断是否为 caption

        Args:
            element: 段落元素对象

        Returns:
            bool: 如果是标题返回 True，否则返回 False
        """
        instr_texts = element.findall(".//w:instrText", namespaces=DocxConverter._BLIP_NAMESPACES)

        for instr in instr_texts:
            if instr.text and "SEQ" in instr.text:
                return True

        return False
