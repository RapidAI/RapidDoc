"""PPTX 列表标记与编号，复用当前转换器的单文档状态。"""

from typing import Optional
from lxml import etree
from pptx.enum.shapes import PP_PLACEHOLDER
from .....schema import BlockType


class _PptxLists:
    """集中维护列表标记与编号，不改变文档生命周期和公开入口。"""

    def _get_paragraph_list_info(self, shape, paragraph) -> dict:
        """基于段落->文本框->布局->母版继承链解析段落列表属性。"""
        marker_info = self._get_effective_list_marker(shape, paragraph)
        p = paragraph._element
        level = marker_info.get("level", self._get_paragraph_level(p))
        kind = marker_info.get("kind")

        if marker_info.get("is_list") is False:
            return {
                "is_list": False,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
                "start": None,
                "start_is_explicit_restart": False,
            }

        if kind == "buAutoNum":
            return {
                "is_list": True,
                "attribute": "ordered",
                "level": level,
                "kind": kind,
                "start": marker_info.get("start"),
                "start_is_explicit_restart": marker_info.get(
                    "start_is_explicit_restart",
                    False,
                ),
            }

        if kind in ("buChar", "buBlip"):
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
                "start": None,
                "start_is_explicit_restart": False,
            }

        if marker_info.get("is_list") is True:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
                "start": None,
                "start_is_explicit_restart": False,
            }

        # 兜底：段落级标记 + 缩进层级判断
        bu_auto_num = p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]})
        if bu_auto_num is not None:
            start = self._parse_positive_int(bu_auto_num.get("startAt"))
            return {
                "is_list": True,
                "attribute": "ordered",
                "level": paragraph.level,
                "kind": "buAutoNum",
                "start": start,
                "start_is_explicit_restart": start is not None,
            }

        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": paragraph.level,
                "kind": "buChar",
                "start": None,
                "start_is_explicit_restart": False,
            }

        if paragraph.level > 0:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": paragraph.level,
                "kind": None,
                "start": None,
                "start_is_explicit_restart": False,
            }

        return {
            "is_list": False,
            "attribute": "unordered",
            "level": 0,
            "kind": None,
            "start": None,
            "start_is_explicit_restart": False,
        }

    def _ensure_list_level(
        self,
        list_stack: list,
        level: int,
        attribute: str,
        start: Optional[int] = None,
        start_is_explicit_restart: bool = False,
    ):
        """将列表栈调整到目标层级，并在必要时创建同级/子级列表块。"""
        while len(list_stack) > level + 1:
            list_stack.pop()

        if len(list_stack) == level + 1 and list_stack[level].get("attribute") != attribute:
            list_stack.pop()

        if self._should_restart_ordered_list(
            list_stack,
            level,
            attribute,
            start,
            start_is_explicit_restart,
        ):
            list_stack.pop()

        while len(list_stack) < level + 1:
            ilevel = len(list_stack)
            new_list_block = {
                "type": BlockType.LIST,
                "attribute": attribute,
                "ilevel": ilevel,
                "content": [],
            }
            if attribute == "ordered" and start is not None and ilevel == level:
                new_list_block["start"] = start

            if list_stack:
                list_stack[-1]["content"].append(new_list_block)
            else:
                self.cur_page.append(new_list_block)

            list_stack.append(new_list_block)

    @staticmethod
    def _get_ordered_list_next_number(list_block: dict) -> int:
        """计算当前有序列表继续追加同级文本项时应使用的下一个编号。"""
        try:
            start = int(list_block.get("start", 1))
        except (TypeError, ValueError):
            start = 1
        direct_item_count = sum(1 for item in list_block.get("content", []) if item.get("type") != BlockType.LIST)
        return start + direct_item_count

    def _should_restart_ordered_list(
        self,
        list_stack: list,
        level: int,
        attribute: str,
        start: Optional[int],
        start_is_explicit_restart: bool,
    ) -> bool:
        """判断 startAt 是否表示当前同级有序列表需要重新开始。"""
        if not start_is_explicit_restart:
            return False
        if attribute != "ordered" or start is None:
            return False
        if len(list_stack) != level + 1:
            return False
        current_list = list_stack[level]
        if current_list.get("attribute") != "ordered":
            return False
        if not current_list.get("content"):
            return False
        return start != self._get_ordered_list_next_number(current_list)

    def _append_list_item(
        self,
        list_stack: list,
        level: int,
        attribute: str,
        content: str,
        start: Optional[int] = None,
        start_is_explicit_restart: bool = False,
    ):
        """向目标层级列表追加文本项。"""
        self._ensure_list_level(
            list_stack,
            level,
            attribute,
            start,
            start_is_explicit_restart,
        )
        list_stack[-1]["content"].append(
            {
                "type": BlockType.TEXT,
                "content": content,
            }
        )

    @staticmethod
    def _normalize_contiguous_list_level(
        raw_level: int,
        base_level: int,
    ) -> int:
        """
        将连续列表段的首个可见层级归一化为0级，避免缺失父级时输出代码块缩进。
        """
        return max(0, raw_level - base_level)

    def _is_list_item(self, paragraph) -> tuple[bool, str]:
        """
        判断段落是否应被视为列表项。
        该方法首先尝试通过拥有该段落的形状来解析列表样式信息。
        如果无法做到，则回退到基于段落属性和级别的更简单检查。
        Args:
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回一个2元组(`is_list`, `bullet_type`)，其中：
            `is_list` - 若段落被视为列表项，为True，否则为False；
            `bullet_type` - 为以下之一：'Bullet'(项目符号)、'Numbered'(编号)或'None'，
            描述列表标记类型。
        """
        # 尝试从段落获取形状（包含该段落的对象），如果可能的话
        shape = None
        try:
            # 这个路径适用于python-pptx段落对象
            # 首先获取文本框架(段落的父对象)
            text_frame = paragraph._parent
            # 然后获取形状(文本框架的父对象)
            shape = text_frame._parent
        except AttributeError:
            pass

        if shape is not None:
            list_info = self._get_paragraph_list_info(shape, paragraph)
            if not list_info["is_list"]:
                return (False, "None")

            if list_info["attribute"] == "ordered":
                return (True, "Numbered")
            return (True, "Bullet")

        # 如果无法获取形状，使用更简单的检查方式
        p = paragraph._element
        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return (True, "Bullet")
        elif p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]}) is not None:
            return (True, "Numbered")
        elif paragraph.level > 0:
            # 很可能是子列表项(缩进表示嵌套)
            return (True, "None")
        else:
            return (False, "None")

    def _get_effective_list_marker(self, shape, paragraph) -> dict:
        """
        返回描述段落的有效列表标记的字典。
        列表标记信息可以来自多个来源：直接段落属性、形状级别的列表样式、
        布局占位符或主幻灯片文本样式。此辅助方法解析所有这些层，并返回
        有效标记的统一视图。

        Args:
            shape: 包含段落的形状对象。
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回列表标记信息的字典，其中：
            `is_list` - True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None，描述标记类型；
            `detail` - 项目符号字符或编号类型字符串，或如果不适用则为None；
            `start_is_explicit_restart` - True 表示 start 来自段落级显式 startAt；
            `level` - 段落级别，范围在(0, 8)内。
        """
        p = paragraph._element
        lvl = self._get_paragraph_level(p)

        # 1) 直接段落属性
        pPr = p.find("a:pPr", namespaces=self.namespaces)
        is_list, kind, detail, start = self._parse_bullet_from_paragraph_properties(pPr)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
                "start": start,
                # 只有段落自身声明的 startAt 才表示一次显式重启。
                "start_is_explicit_restart": kind == "buAutoNum" and start is not None,
            }

        # 2) 形状级别的列表样式(txBody/a:lstStyle)
        txBody = shape._element.find(".//p:txBody", namespaces=self.namespaces)
        is_list, kind, detail, start = self._parse_bullet_from_text_body_list_style(txBody, lvl)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
                "start": start,
                "start_is_explicit_restart": False,
            }

        # 3) 布局占位符列表样式(如果这是一个占位符)
        layout_result = None
        if shape.is_placeholder:
            layout_ph = self._resolve_layout_placeholder(shape)

            if layout_ph is not None:
                layout_tx = layout_ph._element.find(".//p:txBody", namespaces=self.namespaces)
                (
                    is_list,
                    kind,
                    detail,
                    start,
                ) = self._parse_bullet_from_text_body_list_style(layout_tx, lvl)

                # 仅在is_list明确为True/False时使用布局结果
                if is_list is not None:
                    layout_result = {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                        "start": start,
                        "start_is_explicit_restart": False,
                    }

                # 4) 解析主文本样式
                ph_type = shape.placeholder_format.type
                master = shape.part.slide.slide_layout.slide_master
                (
                    is_list,
                    kind,
                    detail,
                    start,
                ) = self._parse_bullet_from_master_text_styles(master, ph_type, lvl)

                # 检查主样式是否有标记信息
                if kind in ("buChar", "buAutoNum", "buBlip"):
                    return {
                        "is_list": True,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                        "start": start,
                        "start_is_explicit_restart": False,
                    }
                elif is_list is not None:
                    return {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                        "start": start,
                        "start_is_explicit_restart": False,
                    }

            # If layout has explicit is_list value but master didn't override it, use layout
            # 如果布局有显式的is_list值但主样式没有覆盖它，则使用布局结果
            if layout_result is not None:
                return layout_result

        return {
            "is_list": None,
            "kind": None,
            "detail": None,
            "level": lvl,
            "start": None,
            "start_is_explicit_restart": False,
        }

    def _get_paragraph_level(self, paragraph) -> int:
        """
        返回段落XML元素的缩进级别。
        段落可以有不同的缩进级别(0-8)。级别存储在段落属性XML元素的'lvl'属性中。

        Args:
            paragraph: 需要提取级别的段落XML元素。

        Returns:
            返回范围在(0, 8)内的段落级别。当找不到'a:pPr'元素、没有'lvl'属性
            或'lvl'属性值无效时，返回0。
        """
        pPr = paragraph.find("a:pPr", namespaces=self.namespaces)
        if pPr is not None and "lvl" in pPr.attrib:
            try:
                return int(pPr.get("lvl"))
            except ValueError:
                pass
        return 0

    @staticmethod
    def _parse_positive_int(value: Optional[str]) -> Optional[int]:
        """解析正整数属性，非法或缺失时返回 None。"""
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _parse_bullet_from_paragraph_properties(
        self, pPr
    ) -> tuple[Optional[bool], Optional[str], Optional[str], Optional[int]]:
        """
        从段落属性节点解析项目符号或编号信息。
        检查'a:pPr'或'a:lvlXpPr'元素，并提取关于项目符号字符、自动编号、
        图片项目符号或显式'buNone'标记的信息。

        Args:
            pPr: 段落属性XML元素('a:pPr'或'a:lvlXpPr')。

        Returns:
            返回一个4元组(`is_list`, `kind`, `detail`, `start`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`(项目符号字符)、`buAutoNum`(自动编号)、
            `buBlip`(图片项目符号)、`buNone`(无标记)或None，描述标记类型；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
            `start` - 自动编号起始值；未声明时为None。
        """
        if pPr is None:
            return (None, None, None, None)

        # 显式指定无项目符号
        if pPr.find("a:buNone", namespaces=self.namespaces) is not None:
            return (False, "buNone", None, None)

        # 项目符号字符
        buChar = pPr.find("a:buChar", namespaces=self.namespaces)
        if buChar is not None:
            return (True, "buChar", buChar.get("char"), None)

        # 自动编号
        buAuto = pPr.find("a:buAutoNum", namespaces=self.namespaces)
        if buAuto is not None:
            return (
                True,
                "buAutoNum",
                buAuto.get("type"),
                self._parse_positive_int(buAuto.get("startAt")),
            )

        # 图片项目符号
        buBlip = pPr.find("a:buBlip", namespaces=self.namespaces)
        if buBlip is not None:
            return (True, "buBlip", "image", None)

        return (None, None, None, None)

    def _parse_bullet_from_text_body_list_style(
        self, txBody, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str], Optional[int]]:
        """
        从文本体的列表样式中解析项目符号或编号信息。
        在'txBody'下搜索'a:lstStyle/a:lvl{lvl+1}pPr'，并使用级别特定的段落属性
        推断项目符号或编号信息。

        Args:
            txBody: 文本体XML元素'p:txBody'。
            lvl: 段落级别，范围在(0, 8)内。
        Returns:
            返回一个4元组(`is_list`, `kind`, `detail`, `start`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
            `start` - 自动编号起始值；未声明时为None。
        """
        if txBody is None:
            return (None, None, None, None)
        lstStyle = txBody.find("a:lstStyle", namespaces=self.namespaces)
        lvl_pPr = self._find_level_properties_in_list_style(lstStyle, lvl)
        return self._parse_bullet_from_paragraph_properties(lvl_pPr)

    def _parse_bullet_from_master_text_styles(
        self, slide_master, placeholder_type, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str], Optional[int]]:
        """
        从主幻灯片的文本样式中解析项目符号或编号信息。
        在主幻灯片的'p:txStyles'中查找相应的样式bucket('titleStyle'、'bodyStyle'或
        'otherStyle')，并为给定的级别提取项目符号或编号信息。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            返回一个4元组(`is_list`, `kind`, `detail`, `start`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
            `start` - 自动编号起始值；未声明时为None。
        """
        style = self._get_master_text_style_node(slide_master, placeholder_type)
        if style is None:
            return (None, None, None, None)

        lvl_pPr = style.find(f".//a:lvl{lvl + 1}pPr", namespaces=self.namespaces)
        return self._parse_bullet_from_paragraph_properties(lvl_pPr)

    def _find_level_properties_in_list_style(self, lstStyle, lvl: int):
        """Find the level-specific paragraph properties node from a list style.
        从列表样式中查找指定级别的段落属性节点。

        This looks for an `a:lvl{lvl+1}pPr` node inside an `a:lstStyle` element, where
        在'a:lstStyle'元素内查找'a:lvl{lvl+1}pPr'节点，其中'a:lvl1pPr'对应级别0，
        `a:lvl1pPr` corresponds to level 0, `a:lvl2pPr` to level 1, and so on.
        'a:lvl2pPr'对应级别1，依此类推。

        Args:
            lstStyle: List style XML element `a:lstStyle`.
            lstStyle: 列表样式XML元素'a:lstStyle'。
            lvl: Paragraph level in the range (0, 8).
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            Matching `a:lvl{lvl+1}pPr` XML element, or None if no matching element is
            匹配的'a:lvl{lvl+1}pPr'XML元素，如果未找到匹配元素则返回None。
                found.
        """
        if lstStyle is None:
            return None
        tag = f"a:lvl{lvl + 1}pPr"
        return lstStyle.find(tag, namespaces=self.namespaces)

    def _get_master_text_style_node(self, slide_master, placeholder_type) -> Optional[etree._Element]:
        """
        获取占位符的相应主文本样式节点。
        大多数内容占位符(BODY/OBJECT)使用'p:bodyStyle'，而标题使用'p:titleStyle'。
        所有其他占位符默认使用'p:otherStyle'。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。

        Returns:
            从主幻灯片的'p:txStyles'中匹配的样式节点('p:bodyStyle'、'p:titleStyle'或'p:otherStyle')，或当未定义样式时返回None。
        """
        txStyles = slide_master._element.find(".//p:txStyles", namespaces=self.namespaces)
        if txStyles is None:
            return None

        if placeholder_type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
            return txStyles.find("p:bodyStyle", namespaces=self.namespaces)

        if placeholder_type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE,
            PP_PLACEHOLDER.SUBTITLE,
        ):
            return txStyles.find("p:titleStyle", namespaces=self.namespaces)

        return txStyles.find("p:otherStyle", namespaces=self.namespaces)
