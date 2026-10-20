"""DOCX 表格处理；共享当前 Converter 的单文档状态。"""

import base64
import re
from contextlib import ExitStack
from io import BytesIO
from typing import Any, Optional
from docx import Document
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from ..rich_text import OfficeRichTextSegment, build_rich_text_html_from_segments, formatting_to_style_str
from docx.oxml.xmlchemy import BaseOxmlElement
from loguru import logger
from mammoth import docx as mammoth_docx, images as mammoth_images
from mammoth import read_embedded_style_map
from mammoth.conversion import convert_document_element_to_html
from mammoth.options import read_options
from mammoth.zips import open_zip
from .office_xml import read_str
from .....schema import BlockType
from .....content.links import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target

from .context import _DocxConstants
from ..image import is_valid_vector_image_payload, is_vector_image_part, serialize_office_image


@mammoth_images.img_element
def _convert_mammoth_table_image(image: Any) -> dict[str, str]:
    """把表格内 WMF/EMF 转为可导出的图片，普通位图保持原始载荷。"""
    with image.open() as stream:
        payload = stream.read()
    if is_vector_image_part(content_type=image.content_type) or is_valid_vector_image_payload(payload):
        source = serialize_office_image(payload, content_type=image.content_type)
    else:
        source = f"data:{image.content_type};base64,{base64.b64encode(payload).decode('ascii')}"
    return {"src": source}


class _DocxTables:
    """集中维护表格，不自行创建文档或持有跨文档缓存。"""

    @staticmethod
    def _mammoth_top_level_table_document(document):
        """只保留 DOCX 正文顶层表格节点，避免 Mammoth 转换非表格正文。"""
        from mammoth import documents as _mammoth_documents

        return document.copy(children=[child for child in document.children if isinstance(child, _mammoth_documents.Table)])

    def _preparse_tables_with_mammoth(self, file_bytes: bytes) -> list:
        """
        使用 mammoth 在完整 DOCX 上下文中预解析所有顶层表格的 HTML。

        孤立模式下（仅传入 <w:tbl> XML 片段），mammoth 缺少编号定义
        （word/numbering.xml）、样式（word/styles.xml）和关系
        （word/_rels/document.xml.rels）等上下文，在遇到含列表项或图片
        的单元格时会抛出 AttributeError。这里让 mammoth 读取完整 DOCX
        包上下文，但通过 transform_document 只转换顶层表格节点，避免把
        非表格正文转换成巨大的 HTML 字符串。

        图片会被 mammoth 转换为内联 data-URI base64 格式（<img src="data:...">）。

        注意：mammoth 不支持 OMML、Equation XML 或 MTEF OLE 公式，会静默丢弃
        表格单元格内的公式。本方法在获取 mammoth HTML 后，会同步遍历原始 DOCX XML，
        将丢失的公式重新注入对应的 HTML 单元格。

        Returns:
            list[str | None]: 与正文顶层表格对齐的 HTML 列表；None 表示该表格
                未找到可靠 Mammoth 结果，后续走完整文档上下文回退解析
        """
        try:
            import mammoth as _mammoth
            from bs4 import BeautifulSoup as _BeautifulSoup

            result = _mammoth.convert_to_html(
                BytesIO(file_bytes),
                transform_document=self._mammoth_top_level_table_document,
                convert_image=_convert_mammoth_table_image,
            )
            soup = _BeautifulSoup(result.value, "html.parser")

            # 仅保留顶层表格，排除嵌套在其他表格单元格内的子表格
            all_tables = soup.find_all("table")
            top_level_tables = [t for t in all_tables if not t.find_parent("table")]

            # 同步加载 DOCX XML，获取所有顶层表格元素，用于公式注入
            docx_obj = Document(BytesIO(file_bytes))
            xml_top_tables = [elem for elem in docx_obj.element.body if self._local_name(elem) == "tbl"]

            logger.debug(f"Pre-parsed {len(top_level_tables)} top-level tables via filtered mammoth conversion")

            result_tables = self._align_mammoth_tables_to_xml_tables(
                top_level_tables,
                xml_top_tables,
                docx_obj.part,
            )
            return result_tables
        except Exception as e:
            logger.debug(f"Could not pre-parse tables with filtered mammoth conversion: {e}")
            return []

    def _align_mammoth_tables_to_xml_tables(
        self,
        html_tables: Any,
        xml_tables: Any,
        source_part: Any,
    ) -> list:
        """
        将 Mammoth 输出表格按正文顶层 XML 表格重新对齐。

        某些 DOCX 会在文本框、图片形状或兼容结构中包含表格，Mammoth 完整
        文档转换时可能把这些结构表格也输出为顶层 HTML table；但正文遍历
        只会在真实 body/w:tbl 上调用 _handle_tables。这里按 XML 表格的
        顺序扫描 Mammoth 候选表，跳过不属于正文顶层表格的候选，避免后续
        _mammoth_table_idx 顺序消费时发生错位。
        """
        aligned_tables = []
        html_index = 0
        matched_count = 0

        for xml_table in xml_tables:
            matched_html_table = None
            scan_index = html_index
            while scan_index < len(html_tables):
                candidate = html_tables[scan_index]
                if self._mammoth_table_matches_xml_table(candidate, xml_table):
                    matched_html_table = candidate
                    html_index = scan_index + 1
                    matched_count += 1
                    break
                scan_index += 1

            if matched_html_table is None:
                aligned_tables.append(None)
                continue

            matched_html_table = self._inject_equations_into_table(
                matched_html_table,
                xml_table,
                source_part,
            )
            aligned_tables.append(str(matched_html_table))

        if len(html_tables) != len(xml_tables):
            logger.debug(f"Aligned {matched_count}/{len(xml_tables)} body tables from {len(html_tables)} mammoth tables")
        return aligned_tables

    @staticmethod
    def _mammoth_table_matches_xml_table(html_table, xml_table) -> bool:
        """
        判断 Mammoth HTML 表格是否对应当前正文 XML 表格。

        文本表优先比较去空白后的表格文本，避免同为 1x1 的图片/文本框表格
        误占正文表格位置；无文本表格再使用结构和图片数量兜底。
        """
        xml_signature = _DocxTables._xml_table_signature(xml_table)
        html_signature = _DocxTables._html_table_signature(html_table)

        if xml_signature["text"] or html_signature["text"]:
            if not _DocxTables._table_text_matches(xml_signature["text"], html_signature["text"]):
                return False
            return (
                xml_signature["cell_count"] == html_signature["cell_count"]
                or xml_signature["row_count"] == html_signature["row_count"]
            )

        return (
            xml_signature["row_count"] == html_signature["row_count"]
            and xml_signature["cell_count"] == html_signature["cell_count"]
            and xml_signature["image_count"] == html_signature["image_count"]
        )

    @staticmethod
    def _xml_table_char_fragment(node: Any) -> str:
        """按 Mammoth 规则把字符级 OOXML 元素渲染为表格签名文本。

        仅拼接 w:t 会遗漏不间断连字符、软连字符和 Symbol 字符，
        导致 XML 签名与 Mammoth HTML 签名不一致。未映射的 w:sym 在
        Mammoth 中会被忽略，此处同样返回空字符串。
        """
        w_ns = _DocxConstants._BLIP_NAMESPACES["w"]
        if node.tag == f"{{{w_ns}}}t":
            return node.text or ""
        if node.tag == f"{{{w_ns}}}noBreakHyphen":
            # NON-BREAKING HYPHEN U+2011，与 Mammoth 的 HTML 渲染一致。
            return "‑"
        if node.tag == f"{{{w_ns}}}softHyphen":
            # SOFT HYPHEN U+00AD，与 Mammoth 的 HTML 渲染一致。
            return "­"
        if node.tag == f"{{{w_ns}}}sym":
            try:
                from mammoth.docx.dingbats import dingbats
            except ImportError:
                return ""

            font = node.get(f"{{{w_ns}}}font", "")
            char = node.get(f"{{{w_ns}}}char", "")
            try:
                code = dingbats.get((font, int(char, 16)))
                if code is None and re.match(r"^F0..", char):
                    code = dingbats.get((font, int(char[2:], 16)))
            except (TypeError, ValueError):
                return ""
            return chr(code) if code is not None else ""
        return ""

    @staticmethod
    def _xml_table_signature(xml_table) -> dict:
        """提取与 Mammoth 渲染规则一致的 XML 表格轻量签名。

        按文档顺序收集 w:t 及特殊字符元素，并且只处理 WordprocessingML
        命名空间，以排除 Mammoth 不渲染的 OMML m:t 公式文本。纵向合并的
        continuation 单元格会被 Mammoth 折叠掉，因此其整个子树也不参与文本签名。
        """
        w_ns = _DocxConstants._BLIP_NAMESPACES["w"]
        char_tags = (
            f"{{{w_ns}}}t",
            f"{{{w_ns}}}noBreakHyphen",
            f"{{{w_ns}}}softHyphen",
            f"{{{w_ns}}}sym",
        )
        continuation_nodes = set()
        for cell in xml_table.iter(f"{{{w_ns}}}tc"):
            properties = cell.find(f"{{{w_ns}}}tcPr")
            vmerge = properties.find(f"{{{w_ns}}}vMerge") if properties is not None else None
            if vmerge is None:
                continue
            # Mammoth 将缺少 val 或 val=continue 的单元格视为 continuation，
            # 并在计算 rowspan 时丢弃该单元格的全部内容。
            if vmerge.get(f"{{{w_ns}}}val") in (None, "continue"):
                continuation_nodes.update(cell.iter())

        ignored_nodes = set(continuation_nodes)
        mc_ns = "http://schemas.openxmlformats.org/markup-compatibility/2006"
        for alternate in xml_table.iter(f"{{{mc_ns}}}AlternateContent"):
            # Mammoth 的 Office XML 读取器仅展开 Fallback 分支。
            for branch in alternate:
                if branch.tag != f"{{{mc_ns}}}Fallback":
                    ignored_nodes.update(branch.iter())
        for simple_field in xml_table.iter(f"{{{w_ns}}}fldSimple"):
            # Mammoth 不解析简单域，其内部的缓存文字也不参与匹配。
            ignored_nodes.update(simple_field.iter())

        wordml_ns = "http://schemas.microsoft.com/office/word/2010/wordml"
        for content_control in xml_table.iter(f"{{{w_ns}}}sdt"):
            properties = content_control.find(f"{{{w_ns}}}sdtPr")
            if properties is None or properties.find(f"{{{wordml_ns}}}checkbox") is None:
                continue
            content = content_control.find(f"{{{w_ns}}}sdtContent")
            if content is None:
                continue
            # Mammoth 以复选框替换内容控件里的首个非空 Text 节点。
            first_text = next(
                (
                    node
                    for node in content.iter()
                    if node not in ignored_nodes and node.tag in char_tags and _DocxTables._xml_table_char_fragment(node)
                ),
                None,
            )
            if first_text is not None:
                ignored_nodes.add(first_text)

        text = "".join(
            _DocxTables._xml_table_char_fragment(node)
            for node in xml_table.iter()
            if node not in ignored_nodes and node.tag in char_tags
        )
        return {
            "row_count": len(xml_table.xpath('.//*[local-name()="tr"]')),
            "cell_count": len(xml_table.xpath('.//*[local-name()="tc"]')),
            "image_count": len(xml_table.xpath('.//*[local-name()="blip" or local-name()="imagedata"]')),
            "text": _DocxTables._normalize_table_match_text(text),
        }

    @staticmethod
    def _html_table_signature(html_table) -> dict:
        """提取 HTML 表格的轻量签名，用于过滤 Mammoth 额外生成的表格。"""
        text_fragments = []
        for fragment in html_table.strings:
            anchor = fragment.find_parent("a")
            if anchor is not None and anchor.parent.name == "sup":
                marker = re.fullmatch(r"(footnote|endnote)-ref-(\d+)", anchor.get("id", ""))
                if marker and anchor.get("href") == f"#{marker.group(1)}-{marker.group(2)}":
                    # 只忽略 Mammoth 生成的脚注引用，保留正文中的普通 [1]。
                    continue
            text_fragments.append(fragment.strip())
        return {
            "row_count": len(html_table.find_all("tr")),
            "cell_count": len(html_table.find_all(["td", "th"])),
            "image_count": len(html_table.find_all("img")),
            "text": _DocxTables._normalize_table_match_text("".join(text_fragments)),
        }

    @staticmethod
    def _normalize_table_match_text(text: str) -> str:
        """统一表格匹配文本，消除 Word 拆字和 Mammoth 空白差异。"""
        return re.sub(r"\s+", "", text or "")

    @staticmethod
    def _table_text_matches(xml_text: str, html_text: str) -> bool:
        """比较表格文本是否指向同一个正文表格。"""
        if not xml_text or not html_text:
            return False
        if xml_text == html_text:
            return True
        return xml_text.startswith(html_text) or html_text.startswith(xml_text)

    def _inject_equations_into_table(
        self,
        html_table: Any,
        xml_table: Any,
        source_part: Any,
    ) -> Any:
        """
        将 DOCX XML 表格中的 OMML/Equation XML/MTEF 公式注入 mammoth HTML 表格。

        mammoth 会静默丢弃 OMML、Equation XML 与 MTEF OLE 公式，导致含公式
        的表格单元格在 HTML 中为空。本方法并行遍历 HTML 表格（BeautifulSoup 对象）
        和 XML 表格（lxml 元素），对含有 OMML 公式的单元格用包含公式占位符的内容
        替换原来的空内容。

        Args:
            html_table: BeautifulSoup 的 Tag 对象，代表 mammoth 生成的 <table> 元素
            xml_table: lxml 的 Element 对象，代表原始 DOCX 中对应的 <w:tbl> 元素

        Returns:
            BeautifulSoup Tag: 注入公式后的 <table> 元素（原地修改并返回）
        """
        W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        # 快速检查：该表格是否含有任何公式
        if not any(
            kind in _DocxConstants._FORMULA_TOKEN_KINDS for kind, _value in self._docx_formula_tokens(xml_table, source_part)
        ):
            return html_table

        from bs4 import BeautifulSoup

        html_rows = html_table.find_all("tr")
        xml_rows = xml_table.findall(f"{{{W_NS}}}tr")

        if len(html_rows) != len(xml_rows):
            logger.debug(f"Table row count mismatch when injecting equations: HTML {len(html_rows)} vs XML {len(xml_rows)}")
            return html_table

        for html_row, xml_row in zip(html_rows, xml_rows):
            html_cells = html_row.find_all(["td", "th"])
            xml_cells = xml_row.findall(f"{{{W_NS}}}tc")

            if len(html_cells) != len(xml_cells):
                continue

            for html_cell, xml_cell in zip(html_cells, xml_cells):
                if not any(
                    kind in _DocxConstants._FORMULA_TOKEN_KINDS
                    for kind, _value in self._docx_formula_tokens(
                        xml_cell,
                        source_part,
                    )
                ):
                    continue

                # 该单元格含公式，重建其 HTML 内容以保留公式
                new_content = self._build_cell_html_with_equations(
                    xml_cell,
                    source_part,
                )
                if new_content:
                    html_cell.clear()
                    new_soup = BeautifulSoup(new_content, "html.parser")
                    for child in list(new_soup.children):
                        html_cell.append(child)

        return html_table

    def _build_cell_html_with_equations(
        self,
        xml_cell: Any,
        source_part: Any,
    ) -> str:
        """
        为含 OMML/Equation XML/MTEF 公式的表格单元格构建 HTML 内容字符串。

        遍历单元格内的段落，将普通文本和 OMML/Equation XML/MTEF 公式
        混合在一起，生成与 mammoth 输出风格一致的 HTML 片段。

        Args:
            xml_cell: lxml Element，代表 DOCX 中的 <w:tc> 元素

        Returns:
            str: 单元格内容的 HTML 字符串，如 "<p>text<eq>latex</eq></p>"；
                 若单元格为空则返回空字符串
        """
        parts = []
        for child in xml_cell:
            child_tag = self._local_name(child)
            if child_tag is None:
                continue
            if child_tag == "p":
                para_html = self._build_paragraph_html_with_equations(
                    child,
                    source_part,
                )
                if para_html is not None:
                    parts.append(para_html)
            # 嵌套表格暂不处理，由外层逻辑负责
        return "".join(parts)

    def _inject_equations_into_table_html(
        self,
        html: str,
        xml_table: Any,
        source_part: Any,
    ) -> str:
        """把孤立表格 HTML 包装为 Tag 后复用统一公式注入逻辑。"""

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        html_table = soup.find("table")
        if html_table is None:
            return html
        return str(
            self._inject_equations_into_table(
                html_table,
                xml_table,
                source_part,
            )
        )

    def _build_paragraph_html_with_equations(
        self,
        xml_para: Any,
        source_part: Any,
    ) -> Optional[str]:
        """
        为可能含 OMML/Equation XML/MTEF 公式的段落构建 HTML 字符串。

        使用与 _handle_equations_in_text 相同的迭代逻辑：
        - 普通 <w:t> 元素的文本直接收集
        - <m:oMath> 元素转换为 LaTeX 并包装为公式占位符 <eq>...</eq>
        - <m:t> 等 math 命名空间下的 <t> 元素因标签中含 "math" 而被跳过，
          避免在 oMath2Latex 已处理整个 oMath 子树后重复提取

        Args:
            xml_para: lxml Element，代表 DOCX 中的 <w:p> 元素

        Returns:
            str | None: 格式为 "<p>...</p>" 的 HTML 字符串；段落为空时返回 None
        """
        paragraph = Paragraph(xml_para, source_part.document)

        def render_node(node: Any, target: str | None = None) -> str:
            """在链接边界内复用公式 token 与富文本规则，避免按可见标签猜测 URL。"""
            name = self._local_name(node)
            if name == "hyperlink":
                rid = node.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                anchor = node.get(f"{{{_DocxConstants._BLIP_NAMESPACES['w']}}}anchor")
                relation = source_part.rels.get(rid) if source_part is not None and rid else None
                address = str(relation.target_ref) if relation is not None else ""
                link = f"{address}#{anchor}" if anchor else address
                return "".join(render_node(child, link or None) for child in node)
            if name in {"p", "sdt", "sdtContent", "smartTag", "customXml"}:
                return "".join(render_node(child, target) for child in node)
            style = formatting_to_style_str(self._get_format_from_run(Run(node, paragraph))) if name == "r" else None
            parts: list[str] = []
            for kind, value in self._docx_formula_tokens(node, source_part):
                if kind == "text":
                    markup = build_rich_text_html_from_segments([OfficeRichTextSegment(value, style, target)])
                    parts.append(markup.replace("\n", "<br>"))
                else:
                    parts.append(self.equation_bookends.format(EQ=value))
            return "".join(parts)

        content = render_node(xml_para)
        return f"<p>{content}</p>" if content else None

    def _close_mammoth_fallback_context(self) -> None:
        """关闭当前文档回退解析持有的 ZIP 资源，并清除其缓存。"""
        context = getattr(self, "_mammoth_fallback_context", None)
        self._mammoth_fallback_context = None
        if context is not None:
            context[0].close()

    def _mammoth_fallback_html(self, element: BaseOxmlElement) -> str:
        """在原 DOCX 的样式、编号、关系和图片上下文中转换当前表格。"""
        context = self._mammoth_fallback_context
        if context is None:
            file_bytes = self._fallback_docx_bytes
            if file_bytes is None:
                raise ValueError("DOCX fallback source is unavailable")
            resources = ExitStack()
            try:
                package = resources.enter_context(open_zip(BytesIO(file_bytes), "r"))
                paths = mammoth_docx._find_part_paths(package)
                read_part = mammoth_docx._part_with_body_reader(None, package, paths, False)
                embedded_style_map = read_embedded_style_map(BytesIO(file_bytes))
                conversion_options = read_options(
                    {"embedded_style_map": embedded_style_map, "convert_image": _convert_mammoth_table_image}
                ).value
                context = (resources, read_part, paths.main_document, conversion_options)
                self._mammoth_fallback_context = context
            except Exception:
                resources.close()
                raise

        _, read_part, main_part, conversion_options = context
        table = read_str(element.xml)

        def read_selected_table(_root: Any, body_reader: Any) -> Any:
            """每次获取独立 reader，只解析当前 XML 表格并沿用主文档关系。"""
            return body_reader.read_all([table])

        result = read_part(main_part, read_selected_table)
        return convert_document_element_to_html(result.value[0], **conversion_options).value

    def _handle_tables(self, element: BaseOxmlElement) -> None:
        """按正文顺序输出表格；预匹配失败时使用完整 DOCX 上下文回退。"""
        table_index = self._mammoth_table_idx
        self._mammoth_table_idx += 1
        html = self._mammoth_tables_html[table_index] if table_index < len(self._mammoth_tables_html) else None
        stage = "full-context fallback"
        try:
            if html is None:
                html = self._mammoth_fallback_html(element)
            stage = "colspan normalization"
            html = self._normalize_table_colspans(html)
            if table_index >= len(self._mammoth_tables_html) or self._mammoth_tables_html[table_index] is None:
                stage = "equation injection"
                html = self._inject_equations_into_table_html(
                    html,
                    element,
                    self._require_document_part(),
                )
        except Exception as exc:
            raise RuntimeError(f"table #{table_index + 1}, {stage}: {exc}") from exc
        self.cur_page.append({"type": BlockType.TABLE, "content": self._normalize_table_links(html)})

    @staticmethod
    def _normalize_table_links(markup: str) -> str:
        """在 Mammoth 表格边界校验链接，非法目标保留原来的可见富文本。"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup, "html.parser")
        changed = False
        for link in soup.find_all("a", href=True):
            if (
                sanitize_hyperlink_target(
                    link["href"], allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES, allow_relative=True, allow_fragment=True
                )
                is None
            ):
                link.unwrap()
                changed = True
        return str(soup) if changed else markup

    def _normalize_table_colspans(self, html: str) -> str:
        """
        修正 HTML 表格中因无线表/少线表导致的 colspan 不一致问题。

        在无边框或少边框的 DOCX 表格中，部分行的单元格包含 w:gridSpan 值，
        该值来自 Word 内部虚拟栅格，并不反映实际视觉列数。mammoth 将这些
        w:gridSpan 值直接转换为 HTML colspan 属性，导致不同行的有效列数
        （所有 colspan 之和）不一致，产生行列对不齐的问题。

        本方法检测此类不一致，并将有效列数过多的行的 colspan 缩减至
        最常见的目标列数，从而恢复表格的正确结构。

        算法：
        1. 计算每行的有效列数（该行所有单元格 colspan 之和）
        2. 取最常见的列数作为目标列数
        3. 对有效列数超过目标值的行，从第一个 colspan > 1 的单元格开始缩减

        Args:
            html: 包含表格的 HTML 字符串

        Returns:
            str: 修正后的 HTML 字符串
        """
        try:
            from collections import Counter

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            modified = False

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                # 若表格中存在 rowspan > 1 的单元格，各行的显式 colspan 之和
                # 无法反映真实网格宽度（被 rowspan 占据的列不出现在后续行的 td
                # 列表中），此时算法的假设不成立，跳过该表格以避免误修改合法的
                # colspan。
                all_cells = table.find_all(["td", "th"])
                if any(int(c.get("rowspan", 1)) > 1 for c in all_cells):
                    continue

                # 计算每行的有效列数（所有单元格的 colspan 之和）
                row_col_counts = []
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    total = sum(int(c.get("colspan", 1)) for c in cells)
                    row_col_counts.append(total)

                if not row_col_counts:
                    continue

                # 找到目标列数（出现最多的列数）
                count_freq = Counter(row_col_counts)
                if len(count_freq) == 1:
                    continue  # 各行列数已一致，无需修正

                target = count_freq.most_common(1)[0][0]

                # 修正有效列数超过目标值的行：缩减 colspan > 1 的单元格
                for row, col_count in zip(rows, row_col_counts):
                    if col_count <= target:
                        continue

                    excess = col_count - target
                    cells = row.find_all(["td", "th"])

                    for cell in cells:
                        if excess <= 0:
                            break
                        span = int(cell.get("colspan", 1))
                        if span > 1:
                            reduce_by = min(span - 1, excess)
                            new_span = span - reduce_by
                            if new_span == 1:
                                if "colspan" in cell.attrs:
                                    del cell["colspan"]
                            else:
                                cell["colspan"] = str(new_span)
                            excess -= reduce_by
                            modified = True

            if modified:
                return str(soup)
            return html
        except Exception as e:
            logger.debug(f"Failed to normalize table colspans: {e}")
            return html
