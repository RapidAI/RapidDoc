"""DOCX 列表与编号处理；共享当前 Converter 的单文档状态。"""

from typing import Optional
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.paragraph import Paragraph
from loguru import logger
from .....schema import BlockType

from .context import _DocxConstants


class _DocxNumbering:
    """集中维护列表与编号，不自行创建文档或持有跨文档缓存。"""

    def _get_numId_and_ilvl(self, paragraph: Paragraph) -> tuple[Optional[int], Optional[int]]:
        """
        获取段落的列表编号ID和层级。

        Args:
            paragraph: 段落对象

        Returns:
            tuple[Optional[int], Optional[int]]: (numId, ilvl) 元组
        """
        numPr = self._get_effective_numPr(paragraph)

        if numPr is not None:
            # 获取 numId 元素并提取值
            namespaces = getattr(numPr, "nsmap", None) or _DocxConstants._BLIP_NAMESPACES
            numId_elem = numPr.find("w:numId", namespaces=namespaces)
            ilvl_elem = numPr.find("w:ilvl", namespaces=namespaces)
            numId = numId_elem.get(self.XML_KEY) if numId_elem is not None else None
            ilvl = ilvl_elem.get(self.XML_KEY) if ilvl_elem is not None else None

            numId_int = self._str_to_int(numId, None)
            ilvl_int = self._str_to_int(ilvl, None)
            if numId_int == 0:
                # numId=0 是 Word 中显式取消编号的信号，不能继续从样式继承编号层级。
                return numId_int, ilvl_int
            if numId_int is not None and ilvl_int is None:
                ilvl_int = self._infer_numbering_ilvl_from_style(numId_int, paragraph)

            return numId_int, ilvl_int

        return None, None  # 如果段落不是列表的一部分

    def _get_numbering_num_element(self, numId: int) -> Optional[BaseOxmlElement]:
        """根据 numId 获取 word/numbering.xml 中的 num 定义。"""
        numbering_root = self._get_numbering_root()
        if numbering_root is None:
            return None

        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return numbering_root.find(
            f".//w:num[@w:numId='{numId}']",
            namespaces=namespaces,
        )

    def _get_abstract_numbering_element(self, numId: int) -> Optional[BaseOxmlElement]:
        """根据 numId 获取对应的 abstractNum 定义，用于复用编号层级解析逻辑。"""
        numbering_root = self._get_numbering_root()
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        if numbering_root is None:
            return None

        num_element = self._get_numbering_num_element(numId)
        if num_element is None:
            return None

        abstract_num_id_elem = num_element.find(".//w:abstractNumId", namespaces=namespaces)
        if abstract_num_id_elem is None:
            return None

        abstract_num_id = abstract_num_id_elem.get(self.XML_KEY)
        if abstract_num_id is None:
            return None

        abstract_num_xpath = f".//w:abstractNum[@w:abstractNumId='{abstract_num_id}']"
        return numbering_root.find(abstract_num_xpath, namespaces=namespaces)

    def _infer_numbering_ilvl_from_style(self, numId: int, paragraph: Paragraph) -> Optional[int]:
        """当 numPr 只有 numId 时，根据 numbering.xml 中的 pStyle 反查编号层级。"""
        abstract_num_element = self._get_abstract_numbering_element(numId)
        if abstract_num_element is None:
            return None

        style_ids = {
            str(getattr(style, "style_id", "") or "") for style in self._iter_style_chain(self._get_paragraph_style(paragraph))
        }
        style_ids.discard("")
        if not style_ids:
            return None

        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        ilvl_attr = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl"
        for lvl_element in abstract_num_element.findall(".//w:lvl", namespaces=namespaces):
            p_style = lvl_element.find("w:pStyle", namespaces=namespaces)
            if p_style is None:
                continue
            if p_style.get(self.XML_KEY) in style_ids:
                return self._str_to_int(lvl_element.get(ilvl_attr), None)
        return None

    def _get_numbering_root(self) -> Optional[BaseOxmlElement]:
        """Load and cache word/numbering.xml once per conversion."""
        if self._numbering_root_loaded:
            return self._numbering_root

        self._numbering_root_loaded = True

        if not hasattr(self.docx_obj, "part") or not hasattr(self.docx_obj.part, "package"):
            return None

        for part in self.docx_obj.part.package.parts:
            if "numbering" in part.partname:
                self._numbering_root = part.element
                break

        return self._numbering_root

    def _get_numbering_level_definition(self, numId: int, ilvl: int) -> Optional[BaseOxmlElement]:
        """Resolve and cache the numbering level definition for a numId/ilvl pair."""
        cache_key = (numId, ilvl)
        if cache_key in self._numbering_level_cache:
            return self._numbering_level_cache[cache_key]

        numbering_root = self._get_numbering_root()
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        lvl_element: Optional[BaseOxmlElement] = None

        abstract_num_element = self._get_abstract_numbering_element(numId)
        if numbering_root is not None and abstract_num_element is not None:
            lvl_xpath = f".//w:lvl[@w:ilvl='{ilvl}']"
            lvl_element = abstract_num_element.find(lvl_xpath, namespaces=namespaces)

        self._numbering_level_cache[cache_key] = lvl_element
        return lvl_element

    def _get_numbering_level_start(self, numId: int, ilvl: int) -> int:
        """解析编号层级的起始值，优先使用 num/lvlOverride，其次使用 abstractNum/lvl/start。"""
        cache_key = (numId, ilvl)
        if cache_key in self._numbering_start_cache:
            return self._numbering_start_cache[cache_key]

        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        start = 1
        num_element = self._get_numbering_num_element(numId)
        if num_element is not None:
            override = num_element.find(
                f"w:lvlOverride[@w:ilvl='{ilvl}']",
                namespaces=namespaces,
            )
            if override is not None:
                start_override = override.find(
                    "w:startOverride",
                    namespaces=namespaces,
                )
                if start_override is not None:
                    start = self._str_to_int(start_override.get(self.XML_KEY), start)
                    self._numbering_start_cache[cache_key] = start
                    return start

        lvl_element = self._get_numbering_level_definition(numId, ilvl)
        if lvl_element is not None:
            start_element = lvl_element.find("w:start", namespaces=namespaces)
            if start_element is not None:
                start = self._str_to_int(start_element.get(self.XML_KEY), start)

        self._numbering_start_cache[cache_key] = start
        return start

    def _advance_list_counter(self, numId: int, ilvl: int) -> int:
        """推进 Word 编号计数，并返回当前列表项应显示的真实序号。"""
        counter_key = (numId, ilvl)
        if counter_key not in self.list_counters:
            current_number = self._get_numbering_level_start(numId, ilvl)
        else:
            current_number = self.list_counters[counter_key] + 1
        self.list_counters[counter_key] = current_number

        # 父级编号前进后，子级编号应在下次出现时重新从定义的起始值开始。
        for key in list(self.list_counters.keys()):
            counter_num_id, counter_ilevel = key
            if counter_num_id == numId and counter_ilevel > ilvl:
                self.list_counters.pop(key, None)

        return current_number

    def _is_numbered_list(self, numId: int, ilvl: int) -> bool:
        """
        根据 numFmt 值检查列表是否为编号列表。

        Args:
            numId: 列表编号ID
            ilvl: 列表层级

        Returns:
            bool: 如果是编号列表返回 True，否则返回 False
        """
        try:
            lvl_element = self._get_numbering_level_definition(numId, ilvl)
            if lvl_element is None:
                return False
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            # 获取 numFmt 元素
            num_fmt_element = lvl_element.find(".//w:numFmt", namespaces=namespaces)
            if num_fmt_element is None:
                return False

            num_fmt = num_fmt_element.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")

            # 编号格式包括: decimal, lowerRoman, upperRoman, lowerLetter, upperLetter
            # 项目符号格式包括: bullet
            numbered_formats = {
                "decimal",
                "lowerRoman",
                "upperRoman",
                "lowerLetter",
                "upperLetter",
                "decimalZero",
            }

            return num_fmt in numbered_formats

        except Exception as e:
            logger.debug(f"Error determining if list is numbered: {e}")
            return False

    def _add_list_item(
        self,
        *,
        numid: int,
        ilevel: int,
        elements: list,
        is_numbered: bool = False,
        text: str = "",
        equations: list = None,
    ) -> list:
        """
        添加列表项。

        生成的列表结构：
        {
            "type": "list",
            "attribute": "ordered" / "unordered",
            "ilevel": 0,
            "content": [
                {"type": "text", "content": "列表项文本"},
                {"type": "list", "attribute": "...", "ilevel": 1, "content": [...]},
                {"type": "text", "content": "另一个列表项"}
            ]
        }

        Args:
            numid: 列表ID
            ilevel: 缩进等级
            elements: 元素列表
            is_numbered: 是否编号
            text: 处理后的文本（包含公式标记）
            equations: 公式列表

        Returns:
            list[RefItem]: 元素引用列表
        """
        if equations is None:
            equations = []
        if not elements:
            return None

        # 构建 content_text，处理行内公式和超链接
        content_text = self._build_text_with_equations_and_hyperlinks(elements, text, equations)
        content_text = self._normalize_text_block_content(content_text)
        if not content_text:
            return None

        # 确定列表属性
        list_attribute = "ordered" if is_numbered else "unordered"
        list_start = self._advance_list_counter(numid, ilevel) if is_numbered else None

        # 情况 1: 不存在上一个列表ID，或遇到了不同 numId 的新列表，创建新的顶层列表
        if self.pre_num_id == -1 or self.pre_num_id != numid:
            # 切换到不同的列表时，先重置旧列表状态
            if self.pre_num_id != -1:
                self._close_active_list()

            list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "content": [],
                "ilevel": ilevel,
            }
            if list_start is not None:
                list_block["start"] = list_start
            self.cur_page.append(list_block)
            # 入栈, 记录当前的列表块
            self.list_block_stack.append(list_block)

            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }

            list_block["content"].append(list_item)
            self.pre_num_id = numid
            self.pre_ilevel = ilevel

        # 情况 2: 增加缩进，打开子列表
        elif (
            self.pre_num_id == numid  # 同一个列表
            and self.pre_ilevel != -1  # 上一个缩进级别已知
            and self.pre_ilevel < ilevel  # 当前层级比之前更缩进
        ):
            # 创建新的子列表块
            child_list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "content": [],
                "ilevel": ilevel,
            }
            if list_start is not None:
                child_list_block["start"] = list_start

            if not self.list_block_stack:
                logger.warning(
                    f"Missing DOCX list parent for increased indent; numid={numid}, ilevel={ilevel}. Starting a new list block."
                )
                self.cur_page.append(child_list_block)
                self.list_block_stack.append(child_list_block)
                child_list_block["content"].append(
                    {
                        "type": BlockType.TEXT,
                        "content": content_text,
                    }
                )
                self.pre_ilevel = ilevel
                return None

            # 获取栈顶的列表块，将子列表直接添加到其content中
            parent_list_block = self.list_block_stack[-1]
            parent_list_block["content"].append(child_list_block)

            # 入栈, 记录当前的列表块
            self.list_block_stack.append(child_list_block)

            # 添加当前列表项到子列表
            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            child_list_block["content"].append(list_item)

            # 更新目前缩进
            self.pre_ilevel = ilevel

        # 情况3: 减少缩进，关闭子列表
        elif (
            self.pre_num_id == numid  # 同一个列表
            and self.pre_ilevel != -1  # 上一个缩进级别已知
            and ilevel < self.pre_ilevel  # 当前层级比之前更少缩进
        ):
            # 出栈，直到找到匹配的 ilevel
            while self.list_block_stack:
                top_list_block = self.list_block_stack[-1]
                if top_list_block["ilevel"] == ilevel:
                    break
                self.list_block_stack.pop()
            if not self.list_block_stack:
                logger.warning(f"Malformed DOCX list nesting; numid={numid}, ilevel={ilevel}. Starting a new list block.")
                list_block = {
                    "type": BlockType.LIST,
                    "attribute": list_attribute,
                    "content": [],
                    "ilevel": ilevel,
                }
                if list_start is not None:
                    list_block["start"] = list_start
                self.cur_page.append(list_block)
                self.list_block_stack.append(list_block)
            else:
                list_block = self.list_block_stack[-1]

            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            list_block["content"].append(list_item)
            self.pre_ilevel = ilevel

        # 情况 4: 同级列表项（相同缩进）
        elif self.pre_num_id == numid and self.pre_ilevel == ilevel:
            if not self.list_block_stack:
                logger.warning(
                    f"Missing DOCX list block for same indent; numid={numid}, ilevel={ilevel}. Starting a new list block."
                )
                list_block = {
                    "type": BlockType.LIST,
                    "attribute": list_attribute,
                    "content": [],
                    "ilevel": ilevel,
                }
                if list_start is not None:
                    list_block["start"] = list_start
                self.cur_page.append(list_block)
                self.list_block_stack.append(list_block)
            else:
                # 获取栈顶的列表块
                list_block = self.list_block_stack[-1]

            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            list_block["content"].append(list_item)

        else:
            logger.warning(
                "Unexpected DOCX list state in _add_list_item: "
                f"pre_num_id={self.pre_num_id}, numid={numid}, "
                f"pre_ilevel={self.pre_ilevel}, ilevel={ilevel}, "
                f"stack_depth={len(self.list_block_stack)}. "
            )

    def _detect_heading_list_numids(self) -> set:
        """
        预扫描文档，检测用作章节标题的列表numId。

        判断依据（需同时满足两个条件）：
        1. 该numId的列表项之间穿插了非列表的正文内容（段落/表格等）；
        2. 该numId的列表项出现在**多个不同的缩进层级**（ilevel > 1种），
           即为真正的多级列表结构，而非普通的单级内容条目列表。

        这样可以避免将"多段内容条目之间穿插了小标签"的单级列表误判为标题列表。

        Returns:
            set: 应当转换为标题块的列表numId集合
        """
        heading_numids = set()
        # 收集文档元素序列：("list", numid, ilevel) 或 ("content",)
        items = []
        # 记录每个numId出现过的所有ilevel，用于判断是否为真正的多级列表
        numid_ilvels: dict[int, set] = {}

        for element in self.docx_obj.element.body:
            tag_name = self._local_name(element)
            if tag_name is None:
                continue
            if tag_name == "p":
                try:
                    paragraph = Paragraph(element, self.docx_obj)
                    p_style_id, _ = self._get_label_and_level(paragraph)
                    numid, ilevel = self._get_numId_and_ilvl(paragraph)
                    if numid == 0:
                        numid = None
                    text = self._get_paragraph_text(paragraph).strip()
                except Exception:
                    continue

                if numid is not None and ilevel is not None and p_style_id not in ["Title", "Heading"] and text:
                    items.append(("list", numid, ilevel))
                    if numid not in numid_ilvels:
                        numid_ilvels[numid] = set()
                    numid_ilvels[numid].add(ilevel)
                elif p_style_id not in ["Title", "Heading"] and text:
                    items.append(("content", None, None))
            elif tag_name == "tbl":
                items.append(("content", None, None))

        # 对每个numId，检测其列表项之间是否有正文内容穿插
        # seen_numids[numid] = True 表示该numId的最后一个列表项之后出现了正文内容
        seen_numids: dict[int, bool] = {}

        for item_type, numid, ilevel in items:
            if item_type == "list":
                if numid in seen_numids and seen_numids[numid]:
                    # 上次列表项之后出现了正文内容，满足条件1
                    heading_numids.add(numid)
                seen_numids[numid] = False  # 重置：记录该numId出现了新列表项
            elif item_type == "content":
                # 将所有已见numId标记为"之后出现了正文内容"
                for nid in seen_numids:
                    seen_numids[nid] = True

        # 条件2：只保留真正的多级列表（出现过多于1种ilevel的numId）
        # 单级列表（如只有ilevel=0的内容条目列表）即使有正文段落穿插也不应转换为标题
        heading_numids = {nid for nid in heading_numids if len(numid_ilvels.get(nid, set())) > 1}

        if heading_numids:
            logger.debug(f"Detected heading-style list numIds (will convert to title blocks): {heading_numids}")

        return heading_numids
