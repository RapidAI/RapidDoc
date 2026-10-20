"""PPTX 文字样式和继承，复用当前转换器的单文档状态。"""

from typing import Any, Optional
from loguru import logger
from lxml import etree
from pptx.oxml.text import CT_TextLineBreak
from ..equation.omml import oMath2Latex
from .....content.spans import append_equation_span, extend_inline_spans, strip_span_dicts
from ..rich_text import OfficeRichTextSegment, build_rich_text_from_segments

from .context import DRAWINGML_NS, A14_DRAWING_NS, OMML_NS


class _PptxTextStyles:
    """集中维护文字样式和继承，不改变文档生命周期和公开入口。"""

    @staticmethod
    def _normalize_xml_toggle_attr(value: Optional[str]) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _normalize_xml_toggle_attr，保持输入顺序与降级行为。"""
        if value is None:
            return None

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "on"}:
            return True
        if normalized in {"0", "false", "f", "off", "none"}:
            return False
        return None

    @classmethod
    def _parse_toggle_attr_from_rpr(
        cls,
        rpr: Optional[etree._Element],
        attr_name: str,
    ) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _parse_toggle_attr_from_rpr，保持输入顺序与降级行为。"""
        if rpr is None:
            return None
        return cls._normalize_xml_toggle_attr(rpr.get(attr_name))

    @classmethod
    def _parse_underline_from_rpr(
        cls,
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _parse_underline_from_rpr，保持输入顺序与降级行为。"""
        if rpr is None:
            return None

        underline = rpr.get("u")
        if underline is None:
            return None

        normalized = str(underline).strip().lower()
        if normalized in {"0", "false", "f", "off", "none"}:
            return False
        return True

    @classmethod
    def _parse_strikethrough_from_rpr(
        cls,
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _parse_strikethrough_from_rpr，保持输入顺序与降级行为。"""
        if rpr is None:
            return None

        strike = rpr.get("strike")
        if strike is None:
            return None

        normalized = str(strike).strip().lower()
        if normalized in {"0", "false", "f", "off", "none", "nostrike"}:
            return False
        return True

    def _get_run_rpr(
        self,
        run,
    ) -> Optional[etree._Element]:
        """按原有文字样式和继承规则执行 _get_run_rpr，保持输入顺序与降级行为。"""
        if run is None:
            return None

        run_xml = getattr(run, "_r", None)
        if run_xml is None:
            return None

        try:
            return run_xml.find("a:rPr", namespaces=self.namespaces)
        except Exception:
            return None

    @staticmethod
    def _get_run_raw_text(run) -> str:
        """从run底层XML读取文本，避免数学run触发python-pptx的to_latex诊断输出。"""
        run_xml = getattr(run, "_r", None)
        if run_xml is None:
            return ""

        text_parts = []
        text_tags = {
            f"{{{DRAWINGML_NS}}}t",
            f"{{{OMML_NS}}}t",
        }
        for text_node in run_xml.iter():
            if getattr(text_node, "tag", None) not in text_tags:
                continue
            if text_node.text:
                text_parts.append(text_node.text)

        return "".join(text_parts)

    def _resolve_effective_run_bool(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
        parser,
    ) -> bool:
        """按原有文字样式和继承规则执行 _resolve_effective_run_bool，保持输入顺序与降级行为。"""
        for source in [self._get_run_rpr(run), *paragraph_font_sources]:
            resolved = parser(source)
            if resolved is not None:
                return resolved
        return False

    def _resolve_effective_run_italic(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        """按原有文字样式和继承规则执行 _resolve_effective_run_italic，保持输入顺序与降级行为。"""
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_italic_from_rpr,
        )

    def _resolve_effective_run_underline(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        """按原有文字样式和继承规则执行 _resolve_effective_run_underline，保持输入顺序与降级行为。"""
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_underline_from_rpr,
        )

    def _resolve_effective_run_strikethrough(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        """按原有文字样式和继承规则执行 _resolve_effective_run_strikethrough，保持输入顺序与降级行为。"""
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_strikethrough_from_rpr,
        )

    def _get_style_str_from_run(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> Optional[str]:
        """从PPTX run对象提取可序列化的生效字体样式字符串。"""
        if run is None:
            return None

        styles = []
        if self._resolve_effective_run_bold(run, paragraph_font_sources):
            styles.append("bold")
        if self._resolve_effective_run_italic(run, paragraph_font_sources):
            styles.append("italic")
        if self._resolve_effective_run_underline(run, paragraph_font_sources):
            styles.append("underline")
        if self._resolve_effective_run_strikethrough(run, paragraph_font_sources):
            styles.append("strikethrough")

        return ",".join(styles) if styles else None

    def _resolve_hyperlink_from_run(self, run, shape) -> Optional[str]:
        """解析 run 对应的超链接，优先公开 API，回退到 XML + rels。"""
        try:
            if hasattr(run, "hyperlink") and run.hyperlink is not None:
                address = run.hyperlink.address
                if address and str(address).strip():
                    return str(address).strip()
        except Exception:
            pass

        try:
            rPr = run._r.find("a:rPr", namespaces=self.namespaces)
            if rPr is None:
                return None

            hlink_click = rPr.find("a:hlinkClick", namespaces=self.namespaces)
            if hlink_click is None:
                return None

            rid = hlink_click.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            if not rid:
                return None

            rels = shape.part.rels
            if rid not in rels:
                return None

            rel = rels[rid]
            target_ref = getattr(rel, "target_ref", None)
            if target_ref and str(target_ref).strip():
                return str(target_ref).strip()

            target_part = getattr(rel, "target_part", None)
            if target_part is not None:
                partname = getattr(target_part, "partname", None)
                if partname and str(partname).strip():
                    return str(partname).strip()
        except Exception:
            return None

        return None

    def _build_paragraph_plain_text(self, paragraph) -> str:
        """构建段落纯文本（保留软换行为空格）。"""
        p = paragraph._element
        text_parts = []
        for node in p.content_children:
            if isinstance(node, CT_TextLineBreak):
                text_parts.append(" ")
                continue

            node_text = getattr(node, "text", None)
            if node_text is not None:
                text_parts.append(node_text)

        return "".join(text_parts)

    @staticmethod
    def _is_math_content_node(node) -> bool:
        """按原有文字样式和继承规则执行 _is_math_content_node，保持输入顺序与降级行为。"""
        tag = getattr(node, "tag", None)
        return tag in {
            f"{{{A14_DRAWING_NS}}}m",
            f"{{{OMML_NS}}}oMath",
            f"{{{OMML_NS}}}oMathPara",
        }

    @staticmethod
    def _strip_math_delimiters(math_text: str) -> str:
        """按原有文字样式和继承规则执行 _strip_math_delimiters，保持输入顺序与降级行为。"""
        stripped = math_text.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
            return stripped[2:-2].strip()
        if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
            return stripped[1:-1].strip()
        return stripped

    def _convert_math_node_to_latex(self, node) -> Optional[str]:
        """按原有文字样式和继承规则执行 _convert_math_node_to_latex，保持输入顺序与降级行为。"""
        omath = None
        if getattr(node, "tag", None) == f"{{{OMML_NS}}}oMath":
            omath = node
        else:
            omath = node.find(".//m:oMath", namespaces=self.namespaces)

        if omath is not None:
            try:
                latex = str(oMath2Latex(omath)).strip()
            except Exception as exc:
                logger.debug(f"Failed to convert PPTX OMML equation to LaTeX: {exc}")
            else:
                if latex:
                    return latex

        fallback_text = getattr(node, "text", None)
        if isinstance(fallback_text, str):
            latex = self._strip_math_delimiters(fallback_text)
            if latex:
                return latex

        return None

    def _build_paragraph_rich_text(self, paragraph, shape) -> list[dict[str, Any]]:
        """按 run 维度构建段落 Span，支持样式、公式与超链接。"""
        paragraph_font_sources = self._get_paragraph_font_sources(shape, paragraph)
        run_map = {}
        for run in paragraph.runs:
            try:
                run_map[id(run._r)] = run
            except Exception:
                continue

        output: list[dict[str, Any]] = []
        segments: list[OfficeRichTextSegment] = []

        def flush_segments() -> None:
            """在公式边界输出累计普通富文本 Span。"""
            if not segments:
                return
            extend_inline_spans(output, build_rich_text_from_segments(list(segments)))
            segments.clear()

        for node in paragraph._element.content_children:
            if isinstance(node, CT_TextLineBreak):
                segments.append(OfficeRichTextSegment(" "))
                continue

            if self._is_math_content_node(node):
                latex = self._convert_math_node_to_latex(node)
                if latex:
                    flush_segments()
                    append_equation_span(output, latex)
                    continue

            node_text = getattr(node, "text", None)
            if node_text is None:
                continue
            if node_text == "":
                continue

            run = run_map.get(id(node))
            if run is None:
                segments.append(OfficeRichTextSegment(node_text))
                continue

            segments.append(
                OfficeRichTextSegment(
                    node_text,
                    self._get_style_str_from_run(
                        run,
                        paragraph_font_sources,
                    ),
                    self._resolve_hyperlink_from_run(run, shape),
                )
            )

        flush_segments()
        return strip_span_dicts(output)

    @staticmethod
    def _trim_rich_text_segments(segments: list[dict]) -> list[dict]:
        """按原有文字样式和继承规则执行 _trim_rich_text_segments，保持输入顺序与降级行为。"""
        trimmed_segments = [dict(segment) for segment in segments if segment.get("text") is not None]
        if not trimmed_segments:
            return []

        start_idx = 0
        while start_idx < len(trimmed_segments):
            normalized_text = trimmed_segments[start_idx]["text"].lstrip()
            if normalized_text:
                trimmed_segments[start_idx]["text"] = normalized_text
                break
            start_idx += 1

        if start_idx == len(trimmed_segments):
            return []

        trimmed_segments = trimmed_segments[start_idx:]
        end_idx = len(trimmed_segments) - 1
        while end_idx >= 0:
            normalized_text = trimmed_segments[end_idx]["text"].rstrip()
            if normalized_text:
                trimmed_segments[end_idx]["text"] = normalized_text
                break
            end_idx -= 1

        if end_idx < 0:
            return []

        return trimmed_segments[: end_idx + 1]

    @staticmethod
    def _normalize_text_block_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """裁剪提取文本首尾空白，同时保留 Span 样式和链接。"""
        return strip_span_dicts(content)

    @staticmethod
    def _parse_font_size_pt_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[float]:
        """按原有文字样式和继承规则执行 _parse_font_size_pt_from_rpr，保持输入顺序与降级行为。"""
        if rpr is None:
            return None

        size = rpr.get("sz")
        if size is None:
            return None

        try:
            return round(int(size) / 100, 1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_bold_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _parse_bold_from_rpr，保持输入顺序与降级行为。"""
        return _PptxTextStyles._parse_toggle_attr_from_rpr(rpr, "b")

    @staticmethod
    def _parse_italic_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        """按原有文字样式和继承规则执行 _parse_italic_from_rpr，保持输入顺序与降级行为。"""
        return _PptxTextStyles._parse_toggle_attr_from_rpr(rpr, "i")

    def _find_def_rpr(
        self,
        paragraph_properties: Optional[etree._Element],
    ) -> Optional[etree._Element]:
        """按原有文字样式和继承规则执行 _find_def_rpr，保持输入顺序与降级行为。"""
        if paragraph_properties is None:
            return None
        return paragraph_properties.find("a:defRPr", namespaces=self.namespaces)

    def _find_end_para_rpr(
        self,
        paragraph: Optional[etree._Element],
    ) -> Optional[etree._Element]:
        """按原有文字样式和继承规则执行 _find_end_para_rpr，保持输入顺序与降级行为。"""
        if paragraph is None:
            return None
        return paragraph.find("a:endParaRPr", namespaces=self.namespaces)

    def _get_font_sources_from_paragraph(
        self,
        paragraph: Optional[etree._Element],
    ) -> list[etree._Element]:
        """按原有文字样式和继承规则执行 _get_font_sources_from_paragraph，保持输入顺序与降级行为。"""
        if paragraph is None:
            return []

        sources = []
        paragraph_properties = paragraph.find("a:pPr", namespaces=self.namespaces)
        paragraph_def_rpr = self._find_def_rpr(paragraph_properties)
        if paragraph_def_rpr is not None:
            sources.append(paragraph_def_rpr)

        end_para_rpr = self._find_end_para_rpr(paragraph)
        if end_para_rpr is not None:
            sources.append(end_para_rpr)

        return sources

    def _get_font_sources_from_text_body(
        self,
        tx_body: Optional[etree._Element],
        level: int,
    ) -> list[etree._Element]:
        """按原有文字样式和继承规则执行 _get_font_sources_from_text_body，保持输入顺序与降级行为。"""
        if tx_body is None:
            return []

        lst_style = tx_body.find("a:lstStyle", namespaces=self.namespaces)
        if lst_style is None:
            return []

        sources = []
        level_properties = self._find_level_properties_in_list_style(
            lst_style,
            level,
        )
        level_def_rpr = self._find_def_rpr(level_properties)
        if level_def_rpr is not None:
            sources.append(level_def_rpr)

        default_properties = lst_style.find("a:defPPr", namespaces=self.namespaces)
        default_def_rpr = self._find_def_rpr(default_properties)
        if default_def_rpr is not None:
            sources.append(default_def_rpr)

        return sources

    def _get_font_sources_from_text_style_bucket(
        self,
        style_bucket: Optional[etree._Element],
        level: int,
    ) -> list[etree._Element]:
        """按原有文字样式和继承规则执行 _get_font_sources_from_text_style_bucket，保持输入顺序与降级行为。"""
        if style_bucket is None:
            return []

        sources = []
        level_properties = style_bucket.find(
            f"a:lvl{level + 1}pPr",
            namespaces=self.namespaces,
        )
        level_def_rpr = self._find_def_rpr(level_properties)
        if level_def_rpr is not None:
            sources.append(level_def_rpr)

        default_properties = style_bucket.find(
            "a:defPPr",
            namespaces=self.namespaces,
        )
        default_def_rpr = self._find_def_rpr(default_properties)
        if default_def_rpr is not None:
            sources.append(default_def_rpr)

        return sources

    def _resolve_layout_placeholder(self, shape):
        """按原有文字样式和继承规则执行 _resolve_layout_placeholder，保持输入顺序与降级行为。"""
        if not getattr(shape, "is_placeholder", False):
            return None

        try:
            idx = shape.placeholder_format.idx
            layout = shape.part.slide.slide_layout
            layout_ph = layout.placeholders.get(idx)
        except Exception:
            layout_ph = None

        if layout_ph is not None:
            return layout_ph

        try:
            placeholder_type = shape.placeholder_format.type
            layout = shape.part.slide.slide_layout
            for candidate in layout.placeholders:
                if candidate.placeholder_format.type == placeholder_type:
                    return candidate
        except Exception:
            return None

        return None

    def _get_paragraph_font_sources(self, shape, paragraph) -> list[etree._Element]:
        """按原有文字样式和继承规则执行 _get_paragraph_font_sources，保持输入顺序与降级行为。"""
        level = self._get_paragraph_level(paragraph._element)
        sources = self._get_font_sources_from_paragraph(paragraph._element)

        tx_body = shape._element.find(".//p:txBody", namespaces=self.namespaces)
        sources.extend(self._get_font_sources_from_text_body(tx_body, level))

        if getattr(shape, "is_placeholder", False):
            layout_placeholder = self._resolve_layout_placeholder(shape)
            if layout_placeholder is not None:
                layout_tx_body = layout_placeholder._element.find(
                    ".//p:txBody",
                    namespaces=self.namespaces,
                )
                sources.extend(self._get_font_sources_from_text_body(layout_tx_body, level))

            try:
                placeholder_type = shape.placeholder_format.type
                slide_master = shape.part.slide.slide_layout.slide_master
            except Exception:
                return sources

            style_bucket = self._get_master_text_style_node(
                slide_master,
                placeholder_type,
            )
            sources.extend(
                self._get_font_sources_from_text_style_bucket(
                    style_bucket,
                    level,
                )
            )

        return sources

    def _resolve_effective_run_font_size_pt(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> Optional[float]:
        """按原有文字样式和继承规则执行 _resolve_effective_run_font_size_pt，保持输入顺序与降级行为。"""
        for source in [self._get_run_rpr(run), *paragraph_font_sources]:
            font_size_pt = self._parse_font_size_pt_from_rpr(source)
            if font_size_pt is not None:
                return font_size_pt
        return None

    def _resolve_effective_run_bold(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        """按原有文字样式和继承规则执行 _resolve_effective_run_bold，保持输入顺序与降级行为。"""
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_bold_from_rpr,
        )

    def _build_paragraph_style_profile(self, shape, paragraph) -> dict[str, Optional[float] | bool]:
        """按原有文字样式和继承规则执行 _build_paragraph_style_profile，保持输入顺序与降级行为。"""
        paragraph_font_sources = self._get_paragraph_font_sources(shape, paragraph)
        effective_font_size_pt = None
        all_bold = True
        has_non_whitespace_run = False

        for run in paragraph.runs:
            run_text = self._get_run_raw_text(run)
            if not run_text.strip():
                continue

            has_non_whitespace_run = True

            run_font_size_pt = self._resolve_effective_run_font_size_pt(
                run,
                paragraph_font_sources,
            )
            if run_font_size_pt is not None:
                if effective_font_size_pt is None:
                    effective_font_size_pt = run_font_size_pt
                else:
                    effective_font_size_pt = max(
                        effective_font_size_pt,
                        run_font_size_pt,
                    )

            if self._resolve_effective_run_bold(run, paragraph_font_sources) is not True:
                all_bold = False

        return {
            "font_size_pt": effective_font_size_pt,
            "all_bold": has_non_whitespace_run and all_bold,
        }
