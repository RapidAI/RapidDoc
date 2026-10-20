"""DOCX 富文本与样式处理；共享当前 Converter 的单文档状态。"""

from pathlib import Path
from typing import Any, Iterator, Optional, Union
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from pydantic import AnyUrl
from .formatting_types import Formatting, Script
from .....content.spans import (
    append_equation_span,
    extend_inline_spans,
    inline_span_plain_text,
    slice_span_dicts,
    strip_span_dicts,
)
from ..rich_text import (
    append_rich_text_element,
    build_spans_from_elements,
    formatting_to_style_str,
    has_non_visible_text_style,
    has_visible_style,
    normalize_format_for_text,
    should_keep_group_text,
)

from .context import _DocxConstants, _ParagraphElement


class _DocxStyles:
    """集中维护富文本与样式，不自行创建文档或持有跨文档缓存。"""

    def _get_style_id_from_property(
        self,
        xml_element: Optional[BaseOxmlElement],
        property_tag: str,
        style_tag: str,
    ) -> Optional[str]:
        """从段落或 run 的直接属性节点读取样式 ID，避免触发 python-docx 样式查找。"""
        if xml_element is None:
            return None

        property_element = xml_element.find(
            property_tag,
            namespaces=_DocxConstants._BLIP_NAMESPACES,
        )
        if property_element is None:
            return None

        style_element = property_element.find(
            style_tag,
            namespaces=_DocxConstants._BLIP_NAMESPACES,
        )
        if style_element is None:
            return None

        return style_element.get(self.XML_KEY) or None

    def _get_cached_docx_style(
        self,
        part: Any,
        style_id: Optional[str],
        style_type: Any,
    ) -> Any:
        """按 style id 和类型缓存 python-docx 样式对象，避免大 styles.xml 被反复线性扫描。"""
        if part is None:
            return None

        cache_key = (style_type, style_id)
        if cache_key not in self._style_lookup_cache:
            self._style_lookup_cache[cache_key] = part.get_style(
                style_id,
                style_type,
            )
        return self._style_lookup_cache[cache_key]

    def _get_paragraph_style(self, paragraph: Optional[Paragraph]) -> Any:
        """读取段落样式；无显式 pStyle 时缓存默认段落样式查询结果。"""
        if paragraph is None:
            return None
        style_id = self._get_style_id_from_property(
            paragraph._element,
            "w:pPr",
            "w:pStyle",
        )
        return self._get_cached_docx_style(
            paragraph.part,
            style_id,
            WD_STYLE_TYPE.PARAGRAPH,
        )

    def _get_run_style(self, run: Optional[Run]) -> Any:
        """读取 run 字符样式；无显式 rStyle 时缓存默认字符样式查询结果。"""
        if run is None:
            return None
        style_id = self._get_style_id_from_property(
            run._element,
            "w:rPr",
            "w:rStyle",
        )
        return self._get_cached_docx_style(
            run.part,
            style_id,
            WD_STYLE_TYPE.CHARACTER,
        )

    @staticmethod
    def _escape_hyperlink_text(text: str) -> str:
        """
        转义超链接文本中的方括号。

        Args:
            text: 要转义的文本

        Returns:
            str: 转义后的文本
        """
        if not text:
            return text
        # 转义方括号
        text = text.replace("[", "\\[").replace("]", "\\]")
        return text

    @staticmethod
    def _escape_hyperlink_url(url: str) -> str:
        """
        转义超链接 URL 中的括号。

        Args:
            url: 要转义的 URL

        Returns:
            str: 转义后的 URL
        """
        if not url:
            return url
        # 对括号进行 URL 编码
        url = url.replace("(", "%28").replace(")", "%29")
        return url

    @staticmethod
    def _get_style_str_from_format(format_obj) -> Optional[str]:
        """
        从 Formatting 对象提取样式字符串。

        Args:
            format_obj: Formatting 对象

        Returns:
            Optional[str]: 样式字符串（如 "bold,italic"），无样式时返回 None
        """
        return formatting_to_style_str(format_obj)

    @staticmethod
    def _has_visible_style(format_obj) -> bool:
        """
        检查格式是否包含可见样式（下划线或删除线）。

        空白文本在有这些样式时仍然是可见的，应当保留。

        Args:
            format_obj: Formatting 对象

        Returns:
            bool: 是否包含可见样式
        """
        return has_visible_style(format_obj)

    @staticmethod
    def _has_non_visible_text_style(format_obj) -> bool:
        """判断格式是否只有空白文本不可见的字形样式。"""
        return has_non_visible_text_style(format_obj)

    @classmethod
    def _normalize_format_for_text(
        cls,
        format_obj: Optional[Formatting],
        text: str,
        *,
        preserve_blank_non_visible_style: bool = False,
    ) -> Optional[Formatting]:
        """按文本内容收敛 run 格式，避免空白 run 把不可见样式传给输出。

        preserve_blank_non_visible_style 用于保留同一文本片段内空白 run 的
        bold/italic：这些样式自身不让空格可见，但可能是连续同样式文本的一部分。
        """
        return normalize_format_for_text(
            format_obj,
            text,
            preserve_blank_non_visible_style=preserve_blank_non_visible_style,
        )

    def _find_adjacent_non_blank_run_format(
        self,
        inline_contents: list[Any],
        current_index: int,
        step: int,
    ) -> Optional[Formatting]:
        """查找相邻方向上最近的非空白普通 run 格式，用于判断空白 run 是否属于同一段样式文本。"""
        index = current_index + step
        while 0 <= index < len(inline_contents):
            content = inline_contents[index]
            # 超链接是独立输出边界，不跨越超链接借用样式上下文。
            if isinstance(content, Hyperlink):
                return None
            if not isinstance(content, Run):
                index += step
                continue
            if self._is_hidden_run(content):
                index += step
                continue
            text = content.text or ""
            if text.strip():
                return self._get_format_from_run(content)
            index += step
        return None

    def _should_preserve_blank_non_visible_style(
        self,
        inline_contents: list[Any],
        current_index: int,
        text: str,
        format_obj: Optional[Formatting],
    ) -> bool:
        """判断空白 run 的 bold/italic 是否应保留，以便连续同样式文本合并成一个 span。"""
        if not text or text.strip():
            return False
        if not self._has_non_visible_text_style(format_obj):
            return False

        previous_format = self._find_adjacent_non_blank_run_format(
            inline_contents,
            current_index,
            -1,
        )
        if format_obj == previous_format:
            return True

        next_format = self._find_adjacent_non_blank_run_format(
            inline_contents,
            current_index,
            1,
        )
        return format_obj == next_format

    @classmethod
    def _should_keep_group_text(
        cls,
        text: str,
        format_obj: Optional[Formatting],
        *,
        preserve_plain_blank: bool = False,
    ) -> bool:
        """判断当前累积 run 是否需要输出，保留夹在可见样式之间的普通空白。"""
        return should_keep_group_text(
            text,
            format_obj,
            preserve_plain_blank=preserve_plain_blank,
        )

    @staticmethod
    def _append_paragraph_element(
        paragraph_elements: list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]],
        text: str,
        format_obj: Optional[Formatting],
        hyperlink: Optional[Union[AnyUrl, Path, str]],
    ) -> None:
        """追加段落元素；相邻同超链接且同格式的 run 合并为一个元素。"""
        append_rich_text_element(paragraph_elements, text, format_obj, hyperlink)

    @staticmethod
    def _normalize_hyperlink_group_boundaries(
        paragraph_elements: list[_ParagraphElement],
    ) -> list[_ParagraphElement]:
        """把链接组边界空白移为普通文本，仅裁剪整段首尾并保留组内空白。"""

        output: list[_ParagraphElement] = []
        index = 0
        while index < len(paragraph_elements):
            element = paragraph_elements[index]
            hyperlink = element[2]
            if hyperlink is None:
                output.append(element)
                index += 1
                continue
            group_end = index + 1
            while (
                group_end < len(paragraph_elements)
                and paragraph_elements[group_end][2] is not None
                and str(paragraph_elements[group_end][2]) == str(hyperlink)
            ):
                group_end += 1
            group = list(paragraph_elements[index:group_end])
            first_text, first_format, first_hyperlink = group[0]
            last_text, last_format, last_hyperlink = group[-1]
            if len(group) == 1 and not first_text.strip():
                leading_space = ""
                trailing_space = first_text
                group[0] = ("", first_format, first_hyperlink)
            else:
                leading_length = len(first_text) - len(first_text.lstrip())
                trailing_length = len(last_text) - len(last_text.rstrip())
                leading_space = first_text[:leading_length]
                trailing_space = last_text[len(last_text) - trailing_length :] if trailing_length else ""
                if len(group) == 1:
                    group[0] = (
                        first_text[leading_length : len(first_text) - trailing_length if trailing_length else len(first_text)],
                        first_format,
                        first_hyperlink,
                    )
                else:
                    group[0] = (
                        first_text[leading_length:],
                        first_format,
                        first_hyperlink,
                    )
                    group[-1] = (
                        last_text[: len(last_text) - trailing_length] if trailing_length else last_text,
                        last_format,
                        last_hyperlink,
                    )
            if leading_space and output:
                output.append((leading_space, first_format, None))
            output.extend(item for item in group if item[0])
            if trailing_space and group_end < len(paragraph_elements):
                output.append((trailing_space, last_format, None))
            index = group_end
        return output

    @staticmethod
    def _is_hidden_run(run: Run) -> bool:
        """Check whether a run is marked as hidden text in Word."""
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        rpr = run._element.find(f"{{{_W}}}rPr")
        if rpr is None:
            return False
        # webHidden: commonly used by TOC page-number field runs
        if rpr.find(f"{{{_W}}}webHidden") is not None:
            return True
        # vanish: generic hidden text
        if rpr.find(f"{{{_W}}}vanish") is not None:
            return True
        return False

    @classmethod
    def _build_spans_from_elements(
        cls,
        paragraph_elements: list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]],
    ) -> list[dict[str, Any]]:
        """按连续同 URL hyperlink 分组，直接生成结构化 Span。"""
        return build_spans_from_elements(paragraph_elements)

    def _build_text_from_elements(
        self,
        paragraph_elements: list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]],
    ) -> list[dict[str, Any]]:
        """
        从 paragraph_elements 重组文本，应用超链接格式和字体样式。

        Args:
            paragraph_elements: 段落元素列表

        Returns:
            list[dict]: 重组后的 Span
        """
        return self._build_spans_from_elements(paragraph_elements)

    @staticmethod
    def _normalize_text_block_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        规范化普通文本块导出内容。

        DOCX 常用段首/段尾空格模拟版式对齐，导出普通文本块前去除这些前后空白。
        """
        return strip_span_dicts(content)

    def _build_text_with_equations_and_hyperlinks(
        self,
        paragraph_elements: list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]],
        text_with_equations: str,
        equations: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """
        构建同时包含公式、超链接和字体样式的文本。

        Args:
            paragraph_elements: 段落元素列表，包含格式和超链接信息
            text_with_equations: 不含公式的原始可见文本
            equations: 按源顺序排列的 text/equation token

        Returns:
            list[dict]: 包含公式、超链接和字体样式的 Span
        """
        if not equations:
            return self._build_text_from_elements(paragraph_elements)
        styled_spans = self._build_text_from_elements(paragraph_elements)
        styled_visible = inline_span_plain_text(styled_spans)
        plain_visible = "".join(value for kind, value in equations if kind == "text")
        if plain_visible != styled_visible:
            return styled_spans

        output: list[dict[str, Any]] = []
        visible_cursor = 0
        for kind, value in equations:
            if kind == "equation":
                append_equation_span(output, value)
                continue
            next_cursor = visible_cursor + len(value)
            extend_inline_spans(output, slice_span_dicts(styled_spans, visible_cursor, next_cursor))
            visible_cursor = next_cursor
        return strip_span_dicts(output)

    @staticmethod
    def _get_paragraph_text_from_contents(
        inner_contents: list[Union[Run, Hyperlink]],
    ) -> str:
        """Rebuild paragraph plain text from visible inline containers."""
        return "".join(content.text or "" for content in inner_contents)

    def _get_paragraph_text(self, paragraph: Paragraph) -> str:
        """Return paragraph plain text, including inline ``w:sdt`` content."""
        return self._get_paragraph_text_from_contents(list(self._iter_paragraph_inner_content(paragraph)))

    def _resolve_style_chain_bool(
        self,
        style_obj,
        attr_name: str,
    ) -> Optional[bool]:
        """从样式继承链中解析布尔字体属性。"""
        if style_obj is None:
            return None

        cache_key = (id(style_obj), attr_name)
        if cache_key in self._style_bool_cache:
            return self._style_bool_cache[cache_key]

        style = style_obj
        result = None
        visited_styles = set()
        while style is not None:
            style_element = getattr(style, "_element", None)
            style_id = getattr(style, "style_id", None)
            style_type = getattr(style, "type", None)
            # DOCX 可能存在 basedOn 自引用或环形引用，记录已访问样式避免继承链死循环。
            style_marker = (
                id(style_element) if style_element is not None else id(style),
                style_type,
                style_id,
            )
            if style_marker in visited_styles:
                break
            visited_styles.add(style_marker)

            font = getattr(style, "font", None)
            if font is not None:
                if attr_name == "underline":
                    value = font.underline
                elif attr_name == "strikethrough":
                    value = font.strike
                else:
                    value = getattr(font, attr_name, None)
                if value is not None:
                    result = bool(value)
                    break
            style = getattr(style, "base_style", None)
        self._style_bool_cache[cache_key] = result
        return result

    def _resolve_run_bool_with_inheritance(
        self,
        run: Run,
        attr_name: str,
    ) -> bool:
        """解析 run 的字体属性，支持 run/字符样式/段落样式继承。"""
        if attr_name == "underline":
            direct_value = run.underline
        elif attr_name == "strikethrough":
            direct_value = run.font.strike
        else:
            direct_value = getattr(run, attr_name, None)

        if direct_value is not None:
            return bool(direct_value)

        # 先看 run 级字符样式链（跳过 Hyperlink 默认字符样式，避免把默认下划线
        # 误当作正文强调样式注入到解析结果中）
        run_style = self._get_run_style(run)
        run_style_id = str(getattr(run_style, "style_id", "") or "").lower()
        run_style_name = str(getattr(run_style, "name", "") or "").lower()
        is_hyperlink_style = run_style_id == "hyperlink" or "hyperlink" in run_style_name
        if not is_hyperlink_style:
            inherited = self._resolve_style_chain_bool(run_style, attr_name)
            if inherited is not None:
                return inherited

        # 再看所在段落样式链
        parent = getattr(run, "_parent", None)
        inherited = self._resolve_style_chain_bool(
            self._get_paragraph_style(parent),
            attr_name,
        )
        if inherited is not None:
            return inherited

        return False

    @staticmethod
    def _get_direct_underline_style(run: Run) -> str:
        """读取 run 级下划线类型，用于区分 words 这类不作用于空格的下划线。"""
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        rPr = run._element.find(f"{{{_W}}}rPr")
        if rPr is None:
            return ""
        underline = rPr.find(f"{{{_W}}}u")
        if underline is None:
            return ""
        return underline.get(f"{{{_W}}}val", "single")

    def _get_format_from_run(self, run: Run) -> Optional[Formatting]:
        """
        从 Run 对象获取格式信息。

        Args:
            run: Run 对象

        Returns:
            Optional[Formatting]: 格式对象
        """
        is_bold = self._resolve_run_bool_with_inheritance(run, "bold")
        is_italic = self._resolve_run_bool_with_inheritance(run, "italic")
        is_strikethrough = self._resolve_run_bool_with_inheritance(run, "strikethrough")
        is_underline = self._resolve_run_bool_with_inheritance(run, "underline")
        underline_style = self._get_direct_underline_style(run)

        # 检测着重符号 (w:em)：独立保留为 emphasis，避免和真实下划线混淆。
        is_emphasis = False
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        rPr = run._element.find(f"{{{_W}}}rPr")
        if rPr is not None:
            em = rPr.find(f"{{{_W}}}em")
            if em is not None:
                em_val = em.get(f"{{{_W}}}val", "")
                if em_val and em_val != "none":
                    is_emphasis = True

        is_sub = run.font.subscript or False
        is_sup = run.font.superscript or False
        script = Script.SUB if is_sub else Script.SUPER if is_sup else Script.BASELINE

        return Formatting(
            bold=is_bold,
            italic=is_italic,
            underline=is_underline,
            underline_style=underline_style,
            emphasis=is_emphasis,
            strikethrough=is_strikethrough,
            script=script,
        )

    def _handle_equations_in_text(
        self,
        element: Any,
        text: str,
        *,
        part: Any | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        处理文本中的公式。

        Args:
            element: 元素对象
            text: 文本内容
            part: 当前段落所属的 OOXML part，用于解析局部 relationship

        Returns:
            tuple: (原始可见文本, 含公式时的有序 text/equation token)
        """
        source_part = part or self._require_document_part()
        only_texts: list[str] = []
        formula_values: list[str] = []
        tokens: list[tuple[str, str]] = []
        for token_kind, value in self._docx_formula_tokens(element, source_part):
            if token_kind == "text":
                only_texts.append(value)
                tokens.append(("text", value))
                continue
            formula_values.append(value)
            tokens.append(("equation", value))

        if not formula_values:
            return text, []

        if "".join(only_texts) != text:
            # 如果我们无法重构初始原始文本
            # 不要尝试解析公式并返回原始文本
            return text, []

        return text, tokens

    def _get_label_and_level(self, paragraph: Paragraph) -> tuple[str, Optional[int]]:
        """
        获取段落的标签和层级。

        Args:
            paragraph: 段落对象

        Returns:
            tuple[str, Optional[int]]: (标签, 层级) 元组
        """
        paragraph_style = self._get_paragraph_style(paragraph)
        if paragraph_style is None:
            return "Normal", None

        label = paragraph_style.style_id
        name = paragraph_style.name

        if label is None:
            return "Normal", None

        for style in self._iter_style_chain(paragraph_style):
            style_label = getattr(style, "style_id", None)
            style_name = getattr(style, "name", None)

            if style_label and ":" in style_label:
                parts = style_label.split(":")
                if len(parts) == 2:
                    return parts[0], self._str_to_int(parts[1], None)

            for candidate in (style_label, style_name):
                if candidate and "heading" in candidate.lower():
                    return self._get_heading_and_level(candidate)

        outline_level = self._get_effective_outline_level(paragraph)
        if outline_level is not None:
            return "Heading", outline_level + 1

        return name or label or "Normal", None

    def _iter_style_chain(self, style: Any) -> Iterator[Any]:
        """Yield a style and its base-style chain once each."""
        seen: set[int] = set()
        current = style
        while current is not None:
            current_id = id(current)
            if current_id in seen:
                break
            seen.add(current_id)
            yield current
            current = getattr(current, "base_style", None)

    def _get_paragraph_property_child(
        self, xml_element: Optional[BaseOxmlElement], child_tag: str
    ) -> Optional[BaseOxmlElement]:
        """Read a direct child from w:pPr without matching nested descendants."""
        if xml_element is None:
            return None

        namespaces = getattr(xml_element, "nsmap", None) or _DocxConstants._BLIP_NAMESPACES
        pPr = xml_element.find("w:pPr", namespaces=namespaces)
        if pPr is None:
            return None
        return pPr.find(child_tag, namespaces=namespaces)

    def _get_effective_numPr(self, paragraph: Paragraph) -> Optional[BaseOxmlElement]:
        """Resolve paragraph numbering from direct properties, then style inheritance."""
        numPr = self._get_paragraph_property_child(paragraph._element, "w:numPr")
        if numPr is not None:
            return numPr

        for style in self._iter_style_chain(self._get_paragraph_style(paragraph)):
            style_element = getattr(style, "element", None)
            numPr = self._get_paragraph_property_child(style_element, "w:numPr")
            if numPr is not None:
                return numPr

        return None

    def _get_effective_outline_level(self, paragraph: Paragraph) -> Optional[int]:
        """Resolve outline level from paragraph properties or inherited styles."""
        outline_lvl = self._get_paragraph_property_child(paragraph._element, "w:outlineLvl")
        if outline_lvl is None:
            for style in self._iter_style_chain(self._get_paragraph_style(paragraph)):
                style_element = getattr(style, "element", None)
                outline_lvl = self._get_paragraph_property_child(style_element, "w:outlineLvl")
                if outline_lvl is not None:
                    break

        if outline_lvl is None:
            return None

        return self._str_to_int(outline_lvl.get(self.XML_KEY), None)
