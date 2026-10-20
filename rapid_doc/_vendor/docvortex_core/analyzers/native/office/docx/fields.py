"""DOCX 字段与目录处理；共享当前 Converter 的单文档状态。"""

import re
from pathlib import Path
from typing import Iterator, Optional, Union
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from loguru import logger
from .....schema import BlockType
from .formatting_types import Formatting

from .context import _DocxConstants, _DocxComplexFieldFrame, _ParagraphElement, _ParagraphHyperlink


class _DocxFields:
    """集中维护字段与目录，不自行创建文档或持有跨文档缓存。"""

    def _collect_toc_anchor_set(self) -> set[str]:
        """从真实超链接和复杂域中收集整份文档的 TOC bookmark 目标。"""
        anchor_attr = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}anchor"
        anchors: set[str] = set()
        for hl in self.docx_obj.element.body.findall(".//w:hyperlink", namespaces=_DocxConstants._BLIP_NAMESPACES):
            anchor = hl.get(anchor_attr, "").strip()
            if anchor and anchor.startswith("_Toc"):
                anchors.add(anchor)
        for paragraph in self.docx_obj.element.body.findall(
            ".//w:p",
            namespaces=_DocxConstants._BLIP_NAMESPACES,
        ):
            for instruction in self._complex_field_instructions(paragraph):
                target, is_internal = self._complex_field_hyperlink_target(instruction)
                anchor = target.removeprefix("#") if target and is_internal else ""
                if anchor.startswith("_Toc"):
                    anchors.add(anchor)
        return anchors

    @classmethod
    def _paragraph_bookmark_names(
        cls,
        paragraph_element: BaseOxmlElement,
    ) -> list[str]:
        """按文档顺序返回段落内可公开的 bookmark 名称，并排除 Word 导航标记。"""

        bookmark_name_attr = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}name"
        names: list[str] = []
        for bookmark in paragraph_element.findall(
            ".//w:bookmarkStart",
            namespaces=cls._BLIP_NAMESPACES,
        ):
            name = bookmark.get(bookmark_name_attr, "").strip()
            if name and not name.startswith("_GoBack"):
                names.append(name)
        return names

    def _collect_toc_anchor_aliases(
        self,
        referenced_anchors: set[str],
    ) -> dict[str, str]:
        """把同一段落的多个 TOC bookmark 收敛到一个 Middle JSON canonical anchor。"""

        aliases: dict[str, str] = {}
        for paragraph in self.docx_obj.element.body.findall(
            ".//w:p",
            namespaces=_DocxConstants._BLIP_NAMESPACES,
        ):
            toc_names = [name for name in self._paragraph_bookmark_names(paragraph) if name.startswith("_Toc")]
            if not toc_names:
                continue
            referenced_names = [name for name in toc_names if name in referenced_anchors]
            canonical = referenced_names[0] if referenced_names else toc_names[0]
            for name in toc_names:
                aliases.setdefault(name, canonical)
        return aliases

    def _canonical_toc_anchor(self, anchor: str) -> str:
        """返回 bookmark alias 对应的唯一公开 anchor，未知名称保持原值。"""

        return self.toc_anchor_aliases.get(anchor, anchor)

    @staticmethod
    def _complex_field_hyperlink_target(instruction: str) -> tuple[str | None, bool]:
        """从复杂字段指令中提取外部 URL 或内部 bookmark fragment。"""
        external_match = re.search(r'\bHYPERLINK\s+"([^"]+)"', instruction, re.IGNORECASE)
        bookmark_match = re.search(r'\\l\s+"([^"]+)"', instruction, re.IGNORECASE)
        address = external_match.group(1).strip() if external_match else ""
        bookmark = bookmark_match.group(1).strip() if bookmark_match else ""
        if address:
            return (f"{address}#{bookmark}" if bookmark else address), False
        if bookmark:
            return f"#{bookmark}", True
        return None, False

    @classmethod
    def _complex_field_instructions(
        cls,
        paragraph_element: BaseOxmlElement,
    ) -> list[str]:
        """按字段边界与嵌套顺序合并段落中被拆分的复杂字段指令。"""

        word_namespace = cls._BLIP_NAMESPACES["w"]
        field_char_tag = f"{{{word_namespace}}}fldChar"
        instruction_tag = f"{{{word_namespace}}}instrText"
        field_type_attr = f"{{{word_namespace}}}fldCharType"
        field_stack: list[_DocxComplexFieldFrame] = []
        instructions: list[str] = []

        def append_instruction(frame: _DocxComplexFieldFrame) -> None:
            """把一个字段已累计的非空指令追加到输出。"""

            instruction = "".join(frame.instruction_parts).strip()
            if instruction:
                instructions.append(instruction)

        for element in paragraph_element.iter():
            if element.tag == field_char_tag:
                field_type = element.get(field_type_attr)
                if field_type == "begin":
                    field_stack.append(_DocxComplexFieldFrame())
                elif field_type == "separate" and field_stack:
                    frame = field_stack[-1]
                    if frame.phase == "instr":
                        append_instruction(frame)
                        frame.phase = "result"
                elif field_type == "end" and field_stack:
                    frame = field_stack.pop()
                    if frame.phase == "instr":
                        append_instruction(frame)
                continue
            if element.tag != instruction_tag:
                continue
            text = element.text or ""
            if field_stack and field_stack[-1].phase == "instr":
                field_stack[-1].instruction_parts.append(text)
            elif text.strip():
                # 兼容缺少 fldChar 包裹、但过去可被逐节点解析的非规范指令。
                instructions.append(text.strip())

        for frame in field_stack:
            if frame.phase == "instr":
                append_instruction(frame)
        return instructions

    @staticmethod
    def _python_docx_hyperlink_target(hyperlink: Hyperlink) -> _ParagraphHyperlink:
        """把 python-docx Hyperlink 的地址或 fragment 转换为行内目标。"""
        address = hyperlink.address
        fragment = hyperlink.fragment
        if address and fragment:
            return f"{address}#{fragment}"
        if address and "://" in address:
            return address
        if address:
            return Path(address)
        if fragment:
            return f"#{fragment}"
        return Path(".")

    def _resolve_complex_field_elements(
        self,
        frame: _DocxComplexFieldFrame,
        *,
        suppress_internal_links: bool,
    ) -> list[_ParagraphElement]:
        """闭合复杂字段，并把字段结果绑定到解析出的超链接目标。"""
        target, is_internal = self._complex_field_hyperlink_target("".join(frame.instruction_parts))
        if target is None or (is_internal and suppress_internal_links):
            return frame.result_elements
        return [(text, format_obj, existing_target or target) for text, format_obj, existing_target in frame.result_elements]

    def _flatten_paragraph_elements(
        self,
        paragraph: Paragraph,
        inner_contents: list[Union[Run, Hyperlink]],
    ) -> list[_ParagraphElement]:
        """按文档顺序展开普通 run、真实超链接与可嵌套复杂字段。"""
        elements: list[_ParagraphElement] = []
        field_stack: list[_DocxComplexFieldFrame] = []
        suppress_internal_links = self._get_toc_item_level(paragraph) is not None
        word_namespace = _DocxConstants._BLIP_NAMESPACES["w"]

        for content_index, content in enumerate(inner_contents):
            if isinstance(content, Hyperlink):
                hyperlink_target = self._python_docx_hyperlink_target(content)
                if suppress_internal_links and isinstance(hyperlink_target, str) and hyperlink_target.startswith("#"):
                    hyperlink_target = None
                hyperlink_elements: list[_ParagraphElement] = []
                for hyperlink_run in content.runs:
                    if self._is_hidden_run(hyperlink_run):
                        continue
                    text = hyperlink_run.text or ""
                    format_obj = self._normalize_format_for_text(
                        self._get_format_from_run(hyperlink_run),
                        text,
                        preserve_blank_non_visible_style=True,
                    )
                    if text != "" or self._has_visible_style(format_obj):
                        hyperlink_elements.append((text, format_obj, hyperlink_target))
                if field_stack and field_stack[-1].phase == "result":
                    field_stack[-1].result_elements.extend(hyperlink_elements)
                else:
                    elements.extend(hyperlink_elements)
                continue

            if not isinstance(content, Run):
                continue

            field_char = content._element.find(f"{{{word_namespace}}}fldChar")
            if field_char is not None:
                field_type = field_char.get(f"{{{word_namespace}}}fldCharType")
                if field_type == "begin":
                    field_stack.append(_DocxComplexFieldFrame())
                elif field_type == "separate" and field_stack:
                    field_stack[-1].phase = "result"
                elif field_type == "end" and field_stack:
                    frame = field_stack.pop()
                    resolved = self._resolve_complex_field_elements(
                        frame,
                        suppress_internal_links=suppress_internal_links,
                    )
                    if field_stack and field_stack[-1].phase == "result":
                        field_stack[-1].result_elements.extend(resolved)
                    else:
                        elements.extend(resolved)
                continue

            instruction = content._element.find(f"{{{word_namespace}}}instrText")
            if instruction is not None and field_stack and field_stack[-1].phase == "instr":
                if instruction.text:
                    field_stack[-1].instruction_parts.append(instruction.text)
                continue

            text = content.text or ""
            raw_format = self._get_format_from_run(content)
            preserve_blank_non_visible_style = self._should_preserve_blank_non_visible_style(
                inner_contents,
                content_index,
                text,
                raw_format,
            )
            format_obj = self._normalize_format_for_text(
                raw_format,
                text,
                preserve_blank_non_visible_style=preserve_blank_non_visible_style,
            )
            element = (text, format_obj, None)
            if field_stack:
                if field_stack[-1].phase == "result":
                    field_stack[-1].result_elements.append(element)
                continue
            elements.append(element)

        while field_stack:
            frame = field_stack.pop()
            resolved = self._resolve_complex_field_elements(
                frame,
                suppress_internal_links=suppress_internal_links,
            )
            if field_stack and field_stack[-1].phase == "result":
                field_stack[-1].result_elements.extend(resolved)
            else:
                elements.extend(resolved)
        return elements

    def _get_paragraph_elements(self, paragraph: Paragraph) -> list[_ParagraphElement]:
        """
        提取段落元素及其格式和超链接信息。

        Args:
            paragraph: 段落对象

        Returns:
            list[_ParagraphElement]:
            段落元素列表，每个元素包含文本、格式和超链接信息
        """

        inner_contents = list(self._iter_paragraph_inner_content(paragraph))
        paragraph_text = self._get_paragraph_text_from_contents(inner_contents)

        # 目前保留空段落以保持向后兼容性:
        if paragraph_text.strip() == "":
            # 检查是否存在带可见样式（下划线或删除线）的空白文本 run。
            # 有可见样式的空白文本（如带下划线的空格）在视觉上是可见的，应予保留，
            # 因此跳过提前返回，交由后续完整 run 处理流程处理。
            has_visible_style_run = any(
                isinstance(c, Run) and c.text and self._has_visible_style(self._get_format_from_run(c)) for c in inner_contents
            )
            if not has_visible_style_run:
                return [("", None, None)]

        paragraph_elements: list[_ParagraphElement] = []
        group_text = ""
        previous_format: Optional[Formatting] = None

        # 遍历已经展开的普通 run、超链接与复杂字段结果，并按格式分组。
        flattened_elements = self._flatten_paragraph_elements(paragraph, inner_contents)
        for text, format_obj, hyperlink in flattened_elements:
            # 当新 run 有可见内容（非空或带可见样式的空白）且格式变化时触发分组
            has_visible_content = len(text.strip()) > 0 or self._has_visible_style(format_obj)
            is_blank_text = bool(text) and not text.strip()
            format_changed = format_obj != previous_format
            has_visible_boundary = self._has_visible_style(previous_format) or self._has_visible_style(format_obj)
            should_split_blank_boundary = is_blank_text and bool(group_text) and format_changed and has_visible_boundary
            if (has_visible_content and format_changed) or should_split_blank_boundary or (hyperlink is not None):
                # 前一组有实质内容（非空或带可见样式的空白）时才保存
                preserve_plain_blank = (
                    bool(group_text)
                    and not group_text.strip()
                    and (self._has_visible_style(previous_format) or self._has_visible_style(format_obj))
                )
                prev_has_visible = self._should_keep_group_text(
                    group_text,
                    previous_format,
                    preserve_plain_blank=preserve_plain_blank,
                )
                if prev_has_visible:
                    paragraph_elements.append((group_text, previous_format, None))
                group_text = ""

                # 如果有超链接，则立即添加
                if hyperlink is not None:
                    self._append_paragraph_element(paragraph_elements, text, format_obj, hyperlink)
                    text = ""
                else:
                    previous_format = format_obj

            group_text += text

        # 格式化最后一个组
        # 注意：使用 previous_format（当前累积组的格式），而非 format（最后一次循环迭代的格式）。
        # 最后一次迭代可能是无样式的空 run，若使用 format 会导致样式丢失。
        last_has_visible = self._should_keep_group_text(
            group_text,
            previous_format,
        )
        if last_has_visible:
            paragraph_elements.append((group_text, previous_format, None))

        return self._normalize_hyperlink_group_boundaries(paragraph_elements)

    def _iter_paragraph_inner_content(
        self,
        paragraph: Paragraph,
        container: Optional[BaseOxmlElement] = None,
    ) -> Iterator[Union[Run, Hyperlink]]:
        """Yield visible paragraph inline containers in document order.

        python-docx only walks direct ``w:r`` and ``w:hyperlink`` children of ``w:p``.
        Inline ``w:sdt`` content controls are skipped entirely, which drops their text
        from both ``paragraph.text`` and ``paragraph.iter_inner_content()``. This walker
        treats ``w:sdt`` and a few transparent wrapper nodes as pass-through containers
        and reuses the existing Run/Hyperlink wrappers for the actual visible content.
        """
        if container is None:
            container = paragraph._element

        _W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        for child in container:
            tag_name = self._local_name(child)
            if tag_name is None:
                continue

            if tag_name == "r":
                yield Run(child, paragraph)
            elif tag_name == "hyperlink":
                yield Hyperlink(child, paragraph)
            elif tag_name == "sdt":
                sdt_content = child.find(f"{{{_W_NS}}}sdtContent")
                if sdt_content is not None:
                    yield from self._iter_paragraph_inner_content(paragraph, sdt_content)
            elif tag_name in self._PARAGRAPH_TRANSPARENT_INLINE_CONTAINERS:
                yield from self._iter_paragraph_inner_content(paragraph, child)

    def _is_toc_sdt(self, element: BaseOxmlElement) -> bool:
        """
        检测SDT元素是否为目录(Table of Contents)。

        检测策略：
        1. 检查 w:sdtPr 中的 docPartGallery 或 tag 元素
        2. 回退到检查内容中的段落样式是否为 "TOC N" 格式

        Args:
            element: SDT XML元素

        Returns:
            bool: 如果是目录SDT返回 True，否则返回 False
        """
        # 方法1: 检查 w:sdtPr 中的 docPartGallery
        sdt_pr = element.find("w:sdtPr", namespaces=_DocxConstants._BLIP_NAMESPACES)
        if sdt_pr is not None:
            doc_part_gallery = sdt_pr.find(".//w:docPartGallery", namespaces=_DocxConstants._BLIP_NAMESPACES)
            if doc_part_gallery is not None:
                val = doc_part_gallery.get(self.XML_KEY, "")
                if "Table of Contents" in val or "toc" in val.lower():
                    return True

            # 检查 tag 元素的值
            tag_elem = sdt_pr.find("w:tag", namespaces=_DocxConstants._BLIP_NAMESPACES)
            if tag_elem is not None:
                val = tag_elem.get(self.XML_KEY, "").lower().replace(" ", "")
                if "toc" in val or "contents" in val or "tableofcontents" in val:
                    return True

        # 方法2: 检查内容段落的样式是否为 "TOC N" 格式
        sdt_content = element.find("w:sdtContent", namespaces=_DocxConstants._BLIP_NAMESPACES)
        if sdt_content is not None:
            paragraphs = sdt_content.findall("w:p", namespaces=_DocxConstants._BLIP_NAMESPACES)
            for p in paragraphs[:5]:  # 只检查前5个段落即可判断
                try:
                    p_obj = Paragraph(p, self.docx_obj)
                    paragraph_style = self._get_paragraph_style(p_obj)
                    if paragraph_style and paragraph_style.name:
                        style_name = paragraph_style.name
                        if re.match(r"^TOC\s*\d+$", style_name, re.IGNORECASE) or re.match(r"^目录\s*\d+$", style_name):
                            return True
                except Exception:
                    continue

        return False

    def _get_toc_item_level(self, paragraph: Paragraph) -> Optional[int]:
        """
        从段落样式中获取目录项的层级（0-based）。

        "TOC 1" -> 0
        "TOC 2" -> 1
        "目录 1" -> 0

        Args:
            paragraph: 段落对象

        Returns:
            Optional[int]: 层级（0-based），如果不是目录样式则返回 None
        """
        paragraph_style = self._get_paragraph_style(paragraph)
        if paragraph_style is None:
            return None
        style_name = paragraph_style.name
        if style_name:
            match = re.match(r"^(?:TOC|目录)\s*(\d+)$", style_name, re.IGNORECASE)
            if match:
                level = int(match.group(1))
                return level - 1  # 转换为 0-based
        return None

    def _is_flat_list_toc(self, items: list[tuple[int, str, list, list, Optional[str]]]) -> bool:
        """
        检测目录是否为扁平列表（插图清单、列表清单等），
        这类目录的所有条目应在同一层级，不应嵌套。

        策略：检查是否超过 50% 的条目以"图"或"表"开头。
        """
        match_count = 0
        total_count = 0
        for _level, text, _elements, _equations, _anchor in items:
            stripped = text.strip()
            if not stripped:
                continue
            total_count += 1
            if re.match(r"^[图表][\d\s.]", stripped) or re.match(r"^(Figure|Table)\s+\d", stripped, re.IGNORECASE):
                match_count += 1
        if total_count == 0:
            return False
        return match_count / total_count > 0.5

    def _correct_toc_level_by_text(self, toc_level: int, text: str) -> int:
        """
        通过文本中的编号深度修正目录项的层级。

        仅对 toc_level > 0 的条目进行修正，避免影响顶层章节标题。
        例如：
        - "1.1 LYSO..." (toc 3 → ilevel=2) → text depth 2 → 返回 1
        - "1.1.1 LYSO..." (toc 3 → ilevel=2) → text depth 3 → 返回 2
        - "本章小结" (toc 1 → ilevel=0) → 返回 0（不修正）
        """
        if toc_level == 0:
            return 0
        stripped = text.strip()
        match = re.match(r"^(\d+(?:\.\d+)+)(?![\d.])", stripped)
        if match:
            parts = match.group(1).split(".")
            # 只用明确的多级章节号把异常偏深的 TOC 样式修浅，避免普通列表编号被提升层级。
            text_level = len(parts) - 1
            if text_level < toc_level:
                return text_level
        return toc_level

    def _add_index_item(
        self,
        *,
        ilevel: int,
        elements: list,
        text: str = "",
        equations: list = None,
        anchor: Optional[str] = None,
    ) -> None:
        """
        添加目录项到索引块。

        生成的索引结构：
        {
            "type": "index",
            "ilevel": 0,
            "content": [
                {"type": "text", "content": "目录项文本"},
                {"type": "index", "ilevel": 1, "content": [...]},
            ]
        }

        Args:
            ilevel: 缩进等级（0-based）
            elements: 元素列表
            text: 处理后的文本（包含公式标记）
            equations: 公式列表
        """
        if equations is None:
            equations = []
        if not elements:
            return

        content_text = self._build_text_with_equations_and_hyperlinks(elements, text, equations)
        content_text = self._normalize_text_block_content(content_text)
        if not content_text:
            return

        # 情况 1: 首个目录项，创建新的顶层索引块
        if self.pre_index_ilevel == -1:
            index_block = {
                "type": BlockType.INDEX,
                "content": [],
                "ilevel": ilevel,
            }
            self.cur_page.append(index_block)
            self.index_block_stack.append(index_block)

            index_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            if anchor:
                index_item["anchor"] = anchor
            index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 2: 增加缩进，打开子索引块
        elif self.pre_index_ilevel < ilevel:
            if not self.index_block_stack:
                # 防御异常 TOC 状态：栈为空时按新的目录块恢复，避免单个坏层级阻断解析。
                logger.debug(
                    "Recovering DOCX index stack before adding TOC item at level {}",
                    ilevel,
                )
                self.pre_index_ilevel = -1
                self._add_index_item(
                    ilevel=ilevel,
                    elements=elements,
                    text=text,
                    equations=equations,
                    anchor=anchor,
                )
                return

            child_index_block = {
                "type": BlockType.INDEX,
                "content": [],
                "ilevel": ilevel,
            }
            parent_index_block = self.index_block_stack[-1]
            parent_index_block["content"].append(child_index_block)
            self.index_block_stack.append(child_index_block)

            index_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            if anchor:
                index_item["anchor"] = anchor
            child_index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 3: 减少缩进，关闭子索引块
        elif ilevel < self.pre_index_ilevel:
            while self.index_block_stack:
                top_block = self.index_block_stack[-1]
                if top_block["ilevel"] == ilevel:
                    break
                self.index_block_stack.pop()
            if self.index_block_stack:
                index_block = self.index_block_stack[-1]
                index_item = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                if anchor:
                    index_item["anchor"] = anchor
                index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 4: 同级目录项
        else:
            if self.index_block_stack:
                index_block = self.index_block_stack[-1]
                index_item = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                if anchor:
                    index_item["anchor"] = anchor
                index_block["content"].append(index_item)

    def _extract_paragraph_bookmark(self, paragraph_element: BaseOxmlElement) -> Optional[str]:
        """Extract a bookmark name from a paragraph, prioritizing TOC bookmarks."""
        names = self._paragraph_bookmark_names(paragraph_element)
        if not names:
            return None
        toc_names = [name for name in names if name.startswith("_Toc")]
        if toc_names:
            # Prefer anchors that are actually referenced by TOC hyperlinks.
            for name in toc_names:
                if name in self.toc_anchor_set:
                    return self._canonical_toc_anchor(name)
            return self._canonical_toc_anchor(toc_names[0])
        return names[0]

    def _extract_toc_target_anchor(self, paragraph_element: BaseOxmlElement) -> Optional[str]:
        """从真实超链接或复杂域中提取 TOC 段落的内部 bookmark。"""
        anchor_attr = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}anchor"
        anchors = []
        for hl in paragraph_element.findall(".//w:hyperlink", namespaces=_DocxConstants._BLIP_NAMESPACES):
            anchor = hl.get(anchor_attr, "").strip()
            if anchor:
                anchors.append(anchor)
        for anchor in anchors:
            if anchor.startswith("_Toc"):
                return self._canonical_toc_anchor(anchor)
        if anchors:
            return anchors[0]

        field_anchors: list[str] = []
        for instruction in self._complex_field_instructions(
            paragraph_element,
        ):
            target, is_internal = self._complex_field_hyperlink_target(instruction)
            anchor = target.removeprefix("#") if target and is_internal else ""
            if anchor:
                field_anchors.append(anchor)
        for anchor in field_anchors:
            if anchor.startswith("_Toc"):
                return self._canonical_toc_anchor(anchor)
        return field_anchors[0] if field_anchors else None

    def _handle_plain_toc_paragraph_as_index(
        self,
        *,
        paragraph: Paragraph,
        paragraph_element: BaseOxmlElement,
        paragraph_elements: list,
        text: str,
        equations: list,
    ) -> bool:
        """将未包裹在 SDT 中的普通目录段落转换为 INDEX 项。"""
        toc_level = self._get_toc_item_level(paragraph)
        if toc_level is None:
            return False
        if not text:
            return True

        target_anchor = self._extract_toc_target_anchor(paragraph_element)
        # 只有已经进入目录序列后才允许无锚点条目，避免误收复用 TOC 样式的封面文本。
        if not target_anchor and self.pre_index_ilevel == -1:
            return False
        if target_anchor and target_anchor.startswith("_Toc"):
            self.toc_anchor_set.add(target_anchor)

        if self.plain_toc_base_level is None:
            self.plain_toc_base_level = toc_level
        normalized_level = max(0, toc_level - self.plain_toc_base_level)
        corrected_level = self._correct_toc_level_by_text(normalized_level, text)
        self._add_index_item(
            ilevel=corrected_level,
            elements=paragraph_elements,
            text=text,
            equations=equations,
            anchor=target_anchor,
        )
        return True

    def _handle_sdt_as_index(self, sdt_content: BaseOxmlElement) -> None:
        """
        处理目录SDT内容，将其转换为层级化的INDEX块。

        两阶段处理：
        1. 收集所有段落及其层级；
        2. 检测目录类型（常规目录 vs 扁平列表），对层级进行修正后写入索引块。

        Args:
            sdt_content: w:sdtContent XML元素
        """
        paragraphs = sdt_content.findall(".//w:p", namespaces=_DocxConstants._BLIP_NAMESPACES)

        # --- 第一阶段：收集所有条目 ---
        toc_items: list[tuple[int, str, list, list, Optional[str]]] = []
        for p in paragraphs:
            try:
                p_obj = Paragraph(p, self.docx_obj)
                paragraph_elements = self._get_paragraph_elements(p_obj)
                text, equations = self._handle_equations_in_text(
                    element=p,
                    text=p_obj.text,
                    part=p_obj.part,
                )
                target_anchor = self._extract_toc_target_anchor(p)
                if target_anchor and target_anchor.startswith("_Toc"):
                    self.toc_anchor_set.add(target_anchor)
                if text is None:
                    continue
                text = text.strip()
                if not text:
                    continue

                toc_level = self._get_toc_item_level(p_obj)
                if toc_level is None:
                    toc_level = 0

                toc_items.append((toc_level, text, paragraph_elements, equations, target_anchor))
            except Exception as e:
                logger.debug(f"Error collecting TOC paragraph: {e}")
                continue

        # --- 第二阶段：修正层级并写入索引块 ---
        is_flat = self._is_flat_list_toc(toc_items)

        # 重置索引状态，开始新的目录块
        self._reset_index_state()

        for toc_level, text, elements, equations, target_anchor in toc_items:
            if is_flat:
                # 插图/列表清单：强制全部扁平（层级 0）
                corrected_level = 0
            else:
                # 常规目录：依据文本编号深度修正层级，解决 docx 跳级问题
                corrected_level = self._correct_toc_level_by_text(toc_level, text)

            self._add_index_item(
                ilevel=corrected_level,
                elements=elements,
                text=text,
                equations=equations,
                anchor=target_anchor,
            )

        # 处理完成后重置索引状态
        self._reset_index_state()

    def _get_heading_and_level(self, style_label: str) -> tuple[str, Optional[int]]:
        """
        从样式标签获取标题和层级。

        Args:
            style_label: 样式标签

        Returns:
            tuple[str, Optional[int]]: (标签字符串, 层级) 元组
        """
        parts = self._split_text_and_number(style_label)

        if len(parts) == 2:
            parts.sort()
            label_str: str = ""
            label_level: Optional[int] = 0
            if parts[0].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[1], None)
            if parts[1].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[0], None)
            return label_str, label_level

        return style_label, None

    def _split_text_and_number(self, input_string: str) -> list[str]:
        """
        分割字符串中的文本和数字部分。

        Args:
            input_string: 输入字符串

        Returns:
            list[str]: 分割后的部分列表
        """
        match = re.match(r"(\D+)(\d+)$|^(\d+)(\D+)", input_string)
        if match:
            parts = list(filter(None, match.groups()))
            return parts
        else:
            return [input_string]

    def _str_to_int(self, s: Optional[str], default: Optional[int] = 0) -> Optional[int]:
        """
        将字符串转换为整数。

        Args:
            s: 要转换的字符串
            default: 默认值，转换失败时返回

        Returns:
            Optional[int]: 转换后的整数，转换失败时返回默认值
        """
        if s is None:
            return None
        try:
            return int(s)
        except ValueError:
            return default
