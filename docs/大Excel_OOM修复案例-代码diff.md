# 附录：相对官方 v0.9.9 的完整代码改动

本附录包含所有修改文件的完整 diff，可直接用于 patch 或代码审查。

---

## 1. rapid_doc/model/xlsx/xlsx_converter.py（核心改动，+631 -100）

```diff
diff --git a/rapid_doc/model/xlsx/xlsx_converter.py b/rapid_doc/model/xlsx/xlsx_converter.py
index c769cc4..7e17f0c 100644
--- a/rapid_doc/model/xlsx/xlsx_converter.py
+++ b/rapid_doc/model/xlsx/xlsx_converter.py
@@ -1,5 +1,6 @@
 # Copyright (c) Opendatalab. All rights reserved.
 import collections
+import gc
 import html
 import posixpath
 import zipfile
@@ -34,6 +35,9 @@ AUTO_GAP_TOLERANCE_CANDIDATES = (0, 1, 2)
 AUTO_GAP_TOLERANCE_PREFERENCE = {1: 0, 0: 1, 2: 2}
 AUTO_GAP_TOLERANCE_PREFERENCE_MARGIN = 0.15
 
+# 大 sheet XML 阈值：超过此大小使用轻量级解析（避免 OOM）
+SHEET_XML_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5MB
+
 
 @dataclass
 class DataRegion:
@@ -164,6 +168,63 @@ class _MergedCellLookup:
         return self._anchor_spans.get((row, col), (1, 1))
 
 
+class _LightCell:
+    """轻量级单元格，仅存储值，替代 openpyxl Cell 以降低内存占用。"""
+    __slots__ = ("row", "column", "value", "font", "alignment", "fill", "hyperlink")
+
+    def __init__(self, row: int, column: int, value):
+        self.row = row
+        self.column = column
+        self.value = value
+        self.font = None
+        self.alignment = None
+        self.fill = None
+        self.hyperlink = None
+
+
+class _LightMergedRange:
+    """轻量级合并区域，提供 min_row/max_row/min_col/max_col 属性（1-based）。"""
+    __slots__ = ("min_row", "max_row", "min_col", "max_col")
+
+    def __init__(self, min_row: int, max_row: int, min_col: int, max_col: int):
+        self.min_row = min_row
+        self.max_row = max_row
+        self.min_col = min_col
+        self.max_col = max_col
+
+
+class _LightMergedCells:
+    """轻量级合并单元格容器，兼容 openpyxl merged_cells 接口。"""
+
+    def __init__(self, ranges: list):
+        self.ranges = ranges
+
+
+class _LightweightSheet:
+    """轻量级工作表，用 dict 存储单元格值，替代 openpyxl Worksheet。
+
+    用于大 sheet XML（>5MB）的内存高效解析，避免 openpyxl 创建百万级 Cell 对象。
+    """
+
+    SHEETSTATE_VISIBLE = "visible"
+
+    def __init__(self, title: str, cells: dict, merged_ranges: list):
+        self.title = title
+        self.sheet_state = self.SHEETSTATE_VISIBLE
+        self._cells = cells  # {(row, col): _LightCell}
+        self.merged_cells = _LightMergedCells(merged_ranges)
+        self._images = []
+        self._charts = []
+        self._rels = []
+
+    def cell(self, row: int, column: int):
+        """按 1-based 行列号获取单元格，不存在则返回空单元格。"""
+        return self._cells.get(
+            (row, column),
+            _LightCell(row, column, None),
+        )
+
+
 class XlsxConverter:
     def __init__(
         self,
@@ -220,12 +281,337 @@ class XlsxConverter:
         self.math_map = {}
         self._merged_cell_lookup_cache = {}
 
+    # ---- 大 sheet 轻量级解析（SAX / iterparse）----
+
+    @staticmethod
+    def _parse_shared_strings(zf: zipfile.ZipFile) -> list[str]:
+        """流式解析 sharedStrings.xml，返回共享字符串列表。
+
+        仅在 <si> end 事件时合并所有 <t> 子元素为一条字符串，
+        确保富文本（含多个 <r><t>）只产生一个 sst 项，索引与 cell 引用一致。
+        注意：不能对 <t> 等子元素调用 clear()，否则 <si> end 时 text 已丢失。
+        """
+        path = "xl/sharedStrings.xml"
+        if path not in zf.namelist():
+            return []
+        NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+        sst: list[str] = []
+        with zf.open(path) as f:
+            for _, el in ET.iterparse(f, events=("end",)):
+                if el.tag != f"{{{NS}}}si":
+                    continue
+                # 合并 si 下所有 <t> 文本（普通字符串 1 个 <t>，富文本多个 <r><t>）
+                texts = "".join(
+                    t.text or "" for t in el.findall(f".//{{{NS}}}t")
+                )
+                sst.append(texts)
+                el.clear()
+        return sst
+
+    @staticmethod
+    def _col_letters_to_num(letters: str) -> int:
+        """将列字母（如 'A', 'BC'）转为 1-based 列号。"""
+        num = 0
+        for ch in letters.upper():
+            num = num * 26 + (ord(ch) - ord("A") + 1)
+        return num
+
+    def _parse_sheet_lightweight(
+        self, sheet_xml_path: str, sheet_name: str
+    ) -> _LightweightSheet:
+        """用 iterparse 流式解析 sheet XML，构建轻量级 Worksheet。
+
+        仅提取单元格值和合并区域，不创建 openpyxl Cell 对象，
+        内存占用从 O(百万级 Python 对象) 降至 O(非空单元格 dict)。
+        """
+        NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+        sst = self._parse_shared_strings(self.zf)
+        cells: dict[tuple[int, int], _LightCell] = {}
+        merged_ranges: list = []
+
+        # 连续空行截断：遇到连续 EMPTY_ROW_CUTOFF 行无非空 cell 则停止解析，
+        # 避免异常文件（如 18 万行空行）浪费 SAX 遍历时间
+        EMPTY_ROW_CUTOFF = 10
+        last_nonempty_row = 0
+        current_row = 0
+        row_has_value = False
+        stop_parsing = False
+
+        with self.zf.open(sheet_xml_path) as f:
+            prev_row = None
+            for event, elem in ET.iterparse(f, events=("start", "end")):
+                if stop_parsing:
+                    elem.clear()
+                    continue
+
+                tag = elem.tag
+
+                if tag == f"{{{NS}}}row" and event == "end":
+                    if prev_row is not None:
+                        prev_row.clear()
+                    prev_row = elem
+                    # 行结束时检查连续空行
+                    if current_row > 0 and not row_has_value:
+                        if current_row - last_nonempty_row >= EMPTY_ROW_CUTOFF:
+                            stop_parsing = True
+                    row_has_value = False
+                    continue
+
+                if tag == f"{{{NS}}}row" and event == "start":
+                    # 更新当前行号
+                    r_attr = elem.get("r", "")
+                    if r_attr.isdigit():
+                        current_row = int(r_attr)
+                    continue
+
+                if tag != f"{{{NS}}}c" or event != "end":
+                    continue
+
+                ref = elem.get("r", "")
+                if not ref:
+                    elem.clear()
+                    continue
+
+                col_str = "".join(ch for ch in ref if ch.isalpha())
+                row_str = "".join(ch for ch in ref if ch.isdigit())
+                try:
+                    row_num = int(row_str)
+                    col_num = 0
+                    for ch in col_str.upper():
+                        col_num = col_num * 26 + (ord(ch) - ord("A") + 1)
+                except (ValueError, OverflowError):
+                    elem.clear()
+                    continue
+
+                cell_type = elem.get("t", "")
+                value = None
+                v_el = elem.find(f"{{{NS}}}v")
+
+                if cell_type == "s" and v_el is not None and v_el.text:
+                    idx = int(v_el.text)
+                    value = sst[idx] if idx < len(sst) else ""
+                elif cell_type == "inlineStr":
+                    is_el = elem.find(f"{{{NS}}}is")
+                    if is_el is not None:
+                        t_el = is_el.find(f"{{{NS}}}t")
+                        if t_el is not None:
+                            value = t_el.text or ""
+                        else:
+                            value = "".join(
+                                t.text or ""
+                                for t in is_el.findall(f".//{{{NS}}}t")
+                            )
+                elif cell_type == "str" and v_el is not None:
+                    value = v_el.text or ""
+                elif v_el is not None and v_el.text:
+                    value = v_el.text
+
+                if value is not None:
+                    cells[(row_num, col_num)] = _LightCell(row_num, col_num, value)
+                    last_nonempty_row = max(last_nonempty_row, row_num)
+                    row_has_value = True
+
+                elem.clear()
+
+        # 合并区域按有效数据边界过滤：丢弃远离实际数据的合并区域
+        # （异常文件可能在 18 万行处有大量空合并区域，全加载会浪费内存）
+        if cells:
+            data_max_row = max(k[0] for k in cells.keys())
+            data_max_col = max(k[1] for k in cells.keys())
+            merged_row_limit = data_max_row + EMPTY_ROW_CUTOFF
+            merged_col_limit = data_max_col + EMPTY_ROW_CUTOFF
+        else:
+            merged_row_limit = merged_col_limit = 0
+
+        # 解析合并单元格
+        import re as _re
+        with self.zf.open(sheet_xml_path) as f:
+            for _, elem in ET.iterparse(f, events=("end",)):
+                if elem.tag == f"{{{NS}}}mergeCell":
+                    ref = elem.get("ref", "")
+                    if ref:
+                        # 解析 "A1:C3" -> _LightMergedRange
+                        m = _re.match(
+                            r"([A-Z]+)(\d+):([A-Z]+)(\d+)", ref
+                        )
+                        if m:
+                            min_row = int(m.group(2))
+                            max_row = int(m.group(4))
+                            min_col = self._col_letters_to_num(m.group(1))
+                            max_col = self._col_letters_to_num(m.group(3))
+                            # 过滤远离数据边界的合并区域
+                            if (
+                                min_row > merged_row_limit
+                                or min_col > merged_col_limit
+                            ):
+                                elem.clear()
+                                continue
+                            merged_ranges.append(
+                                _LightMergedRange(
+                                    min_row, max_row, min_col, max_col,
+                                )
+                            )
+                    elem.clear()
+
+        logger.info(
+            "轻量级解析 %s: %d 单元格, %d 合并区域",
+            sheet_name, len(cells), len(merged_ranges),
+        )
+        return _LightweightSheet(sheet_name, cells, merged_ranges)
+
+    def _get_sheet_xml_path(self, sheet_title: str) -> str | None:
+        """从 ZIP 中查找 sheet 标题对应的 XML 路径。"""
+        if not self.zf:
+            return None
+        NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+        NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
+        NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
+        try:
+            wb_xml = self.zf.read("xl/workbook.xml")
+            wb_root = ET.fromstring(wb_xml)
+            target = None
+            for s in wb_root.findall(f".//{{{NS_MAIN}}}sheet"):
+                if s.get("name") == sheet_title:
+                    rid = s.get(f"{{{NS_R}}}id")
+                    break
+            else:
+                return None
+            rid = s.get(f"{{{NS_R}}}id")
+            rels_xml = self.zf.read("xl/_rels/workbook.xml.rels")
+            rels_root = ET.fromstring(rels_xml)
+            for rel in rels_root.findall(f"{{{NS_REL}}}Relationship"):
+                if rel.get("Id") == rid:
+                    target = rel.get("Target", "")
+                    break
+            if target:
+                return f"xl/{target}" if not target.startswith("/") else target.lstrip("/")
+        except Exception as e:
+            logger.warning("查找 sheet XML 路径失败 %s: %s", sheet_title, e)
+        return None
+
+    def _get_sheet_xml_size(self, sheet_title: str) -> int:
+        """获取 sheet XML 解压后大小。"""
+        path = self._get_sheet_xml_path(sheet_title)
+        if path and self.zf and path in self.zf.namelist():
+            return self.zf.getinfo(path).file_size
+        return 0
+
+    def _collect_lightweight_sheet_images(
+        self, sheet: _LightweightSheet
+    ) -> list[dict]:
+        """从 ZIP 直接解析大 sheet 的 drawing 图片。"""
+        if not self.zf:
+            return []
+        NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+        NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
+        NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
+        NS_XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
+        NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
+
+        sheet_path = self._get_sheet_xml_path(sheet.title)
+        if not sheet_path:
+            return []
+
+        sheet_name_base = sheet_path.rsplit("/", 1)[-1]
+        rels_path = f"xl/worksheets/_rels/{sheet_name_base}.rels"
+        if rels_path not in self.zf.namelist():
+            return []
+
+        try:
+            rels_xml = self.zf.read(rels_path)
+            rels_root = ET.fromstring(rels_xml)
+            drawing_target = None
+            for rel in rels_root.findall(f"{{{NS_REL}}}Relationship"):
+                if "drawing" in rel.get("Type", ""):
+                    drawing_target = rel.get("Target", "")
+                    break
+            if not drawing_target:
+                return []
+
+            if drawing_target.startswith("../"):
+                drawing_path = "xl/" + drawing_target.replace("../", "")
+            elif drawing_target.startswith("/"):
+                drawing_path = drawing_target.lstrip("/")
+            else:
+                drawing_path = f"xl/worksheets/{drawing_target}"
+
+            if drawing_path not in self.zf.namelist():
+                return []
+
+            drawing_xml = self.zf.read(drawing_path)
+            drawing_root = ET.fromstring(drawing_xml)
+
+            drawing_dir = drawing_path.rsplit("/", 1)[0]
+            drawing_name = drawing_path.rsplit("/", 1)[-1]
+            drawing_rels_path = f"{drawing_dir}/_rels/{drawing_name}.rels"
+
+            img_rid_to_target = {}
+            if drawing_rels_path in self.zf.namelist():
+                d_rels = ET.fromstring(self.zf.read(drawing_rels_path))
+                for rel in d_rels.findall(f"{{{NS_REL}}}Relationship"):
+                    if "image" in rel.get("Type", ""):
+                        img_rid_to_target[rel.get("Id")] = rel.get("Target", "")
+
+            images = []
+            for anchor_tag in ["twoCellAnchor", "oneCellAnchor"]:
+                for anchor in drawing_root.findall(f".//{{{NS_XDR}}}{anchor_tag}"):
+                    from_node = anchor.find(f"{{{NS_XDR}}}from")
+                    if from_node is None:
+                        continue
+                    col_node = from_node.find(f"{{{NS_XDR}}}col")
+                    row_node = from_node.find(f"{{{NS_XDR}}}row")
+                    if col_node is None or row_node is None:
+                        continue
+                    col_idx = int(col_node.text or "0")
+                    row_idx = int(row_node.text or "0")
+
+                    blip = anchor.find(f".//{{{NS_A}}}blip")
+                    if blip is None:
+                        continue
+                    embed_id = blip.get(f"{{{NS_R}}}embed")
+                    if not embed_id or embed_id not in img_rid_to_target:
+                        continue
+
+                    img_rel_path = img_rid_to_target[embed_id]
+                    if img_rel_path.startswith("../"):
+                        img_path = f"xl/{img_rel_path.replace('../', '')}"
+                    else:
+                        img_path = f"{drawing_dir}/{img_rel_path}"
+
+                    if img_path not in self.zf.namelist():
+                        continue
+
+                    try:
+                        with self.zf.open(img_path) as img_f:
+                            pil_img = Image.open(img_f)
+                            pil_img.load()
+                            if pil_img.mode != "RGB":
+                                b64 = image_to_b64str(pil_img, image_format="PNG")
+                            else:
+                                b64 = image_to_b64str(pil_img, image_format="JPEG")
+                            images.append({
+                                "anchor": (row_idx, col_idx),
+                                "base64": b64,
+                            })
+                    except Exception as e:
+                        logger.warning("轻量级图片提取失败 %s: %s", img_path, e)
+
+            return images
+        except Exception as e:
+            logger.warning("轻量级 drawing 解析失败 %s: %s", sheet.title, e)
+            return []
+
+    # ---- 常规解析路径 ----
+
     def _convert_package_bytes(self, file_bytes: bytes) -> None:
         """用独立字节流解析 XLSX 包，便于原始包失败后用规范化包重试。"""
         self._convert_package_stream(BytesIO(file_bytes))
 
     def _convert_package_stream(self, file_stream: BinaryIO) -> None:
-        """直接使用可复位的 XLSX 流解析正常路径，避免提前复制完整包字节。"""
+        """直接使用可复位的 XLSX 流解析正常路径，避免提前复制完整包字节。
+
+        对大 sheet XML（>5MB）使用纯 ZIP 解析（完全跳过 openpyxl），避免 OOM。
+        """
         self._reset_state()
         try:
             self.zf = zipfile.ZipFile(file_stream)
@@ -233,33 +619,118 @@ class XlsxConverter:
             logger.warning(f"Failed to open zip file: {e}")
             self.zf = None
 
+        if not self.zf:
+            return
+
         try:
-            rewind_stream(file_stream)
-            self.workbook = load_workbook(
-                filename=file_stream,
-                data_only=True,
-                rich_text=True,
-            )
-            if self.workbook is not None:
-                # 遍历需要参与转换的工作表，避免为隐藏表或尾部空页生成无效页面。
-                sheet_pages = []
-                for idx, sheet in enumerate(self._iter_sheets_to_convert(), start=1):
-                    logger.debug(f"正在处理第 {idx} 个工作表：{sheet.title}")
-                    self.cur_page = []
-                    self._convert_sheet(sheet)
-                    sheet_pages.append((sheet.title, self.cur_page))
-                if self._should_emit_sheet_titles(
-                    [page for _, page in sheet_pages]
-                ):
-                    self._prepend_sheet_titles(sheet_pages)
-                self.pages.extend(page for _, page in sheet_pages)
+            # 检查是否有大 sheet，决定走哪条解析路径
+            has_large_sheet = self._has_large_sheet()
+
+            if has_large_sheet:
+                logger.info("检测到超大 sheet，使用纯 ZIP 解析路径（跳过 openpyxl）")
+                self._convert_all_sheets_lightweight()
             else:
-                logger.error("工作簿未初始化。")
+                # 小文件走 openpyxl 完整解析（保留样式/富文本/超链接）
+                rewind_stream(file_stream)
+                self.workbook = load_workbook(
+                    filename=file_stream,
+                    data_only=True,
+                    rich_text=True,
+                )
+                if self.workbook is not None:
+                    for idx, ws in enumerate(self._iter_sheets_to_convert(), start=1):
+                        logger.debug(f"正在处理第 {idx} 个工作表：{ws.title}")
+                        self.cur_page = []
+                        self._convert_sheet(ws)
+                        if self._should_emit_sheet_titles(
+                            [(ws.title, self.cur_page)]
+                        ):
+                            pass  # 单 sheet 不加标题
+                        self.pages.append(self.cur_page)
+                else:
+                    logger.error("工作簿未初始化。")
         finally:
             if self.zf:
                 self.zf.close()
                 self.zf = None
 
+    def _has_large_sheet(self) -> bool:
+        """检查 ZIP 内是否有超过阈值的 sheet XML。"""
+        if not self.zf:
+            return False
+        for name in self.zf.namelist():
+            if "xl/worksheets/" in name and name.endswith(".xml") and "_rels" not in name:
+                if self.zf.getinfo(name).file_size > SHEET_XML_SIZE_THRESHOLD:
+                    return True
+        return False
+
+    def _get_all_sheet_infos(self) -> list[dict]:
+        """从 ZIP 解析所有 sheet 的元数据（标题、XML 路径、大小）。"""
+        NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+        NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
+        NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
+
+        wb_xml = self.zf.read("xl/workbook.xml")
+        wb_root = ET.fromstring(wb_xml)
+
+        rid_to_target = {}
+        rels_xml = self.zf.read("xl/_rels/workbook.xml.rels")
+        for rel in ET.fromstring(rels_xml).findall(f"{{{NS_REL}}}Relationship"):
+            rid_to_target[rel.get("Id")] = rel.get("Target", "")
+
+        sheets = []
+        for s in wb_root.findall(f".//{{{NS_MAIN}}}sheet"):
+            name = s.get("name")
+            rid = s.get(f"{{{NS_R}}}id")
+            state = s.get("state", "visible")
+            target = rid_to_target.get(rid, "")
+            xml_path = f"xl/{target}" if not target.startswith("/") else target.lstrip("/")
+            xml_size = 0
+            if xml_path in self.zf.namelist():
+                xml_size = self.zf.getinfo(xml_path).file_size
+            sheets.append({
+                "name": name,
+                "xml_path": xml_path,
+                "xml_size": xml_size,
+                "state": state,
+            })
+        return sheets
+
+    def _convert_all_sheets_lightweight(self):
+        """纯 ZIP 路径：所有 sheet 均用轻量级 iterparse 解析。"""
+        sheet_infos = self._get_all_sheet_infos()
+        sheet_pages = []
+
+        for idx, info in enumerate(sheet_infos, start=1):
+            if (
+                not self.include_hidden_sheets
+                and info["state"] != "visible"
+            ):
+                logger.debug(f"跳过隐藏工作表：{info['name']}")
+                continue
+
+            logger.info(
+                "处理 sheet '{}' (XML {:.1f} MB)",
+                info["name"], info["xml_size"] / 1024 / 1024,
+            )
+
+            light_sheet = self._parse_sheet_lightweight(
+                info["xml_path"], info["name"],
+            )
+            light_sheet._images = self._collect_lightweight_sheet_images(light_sheet)
+
+            self.cur_page = []
+            self._convert_sheet(light_sheet)
+            sheet_pages.append((info["name"], self.cur_page))
+
+            # 释放当前 sheet 数据再处理下一个
+            del light_sheet
+            gc.collect()
+
+        if self._should_emit_sheet_titles([page for _, page in sheet_pages]):
+            self._prepend_sheet_titles(sheet_pages)
+        self.pages.extend(page for _, page in sheet_pages)
+
     def _retry_convert_package_bytes_after_normalization(
         self,
         file_bytes: bytes,
@@ -306,27 +777,67 @@ class XlsxConverter:
             page.insert(0, self._build_sheet_title_block(sheet_title))
 
     def _convert_sheet(self, sheet):
-        if isinstance(sheet, Worksheet):
-            # Pre-calc maps
+        is_light = isinstance(sheet, _LightweightSheet)
+        is_worksheet = isinstance(sheet, Worksheet)
+
+        if is_light:
+            # 轻量级 sheet：已解析好的 _LightweightSheet
+            self.math_map = {}
+            self.sheet_images = list(getattr(sheet, "_images", []))
+        elif is_worksheet:
+            # 普通 Worksheet（不应在 read_only=True 流程中出现，但保留兼容）
             self.math_map = self._map_math_formulas_to_cells(sheet)
             self.sheet_images = self._collect_sheet_images(sheet)
-            self.table_image_map = collections.defaultdict(list)
-            for image_info in self.sheet_images:
-                anchor = image_info["anchor"]
-                if anchor[0] is None or anchor[1] is None:
-                    continue
-                self.table_image_map[anchor].append(
-                    f'<img src="{image_info["base64"]}" />'
-                )
+        else:
+            # 只读 ReadOnlyWorksheet：无 _cells，需从 iter_rows 构建
+            self.math_map = {}
+            self.sheet_images = []
+            cells = {}
+            for row in sheet.iter_rows():
+                for c in row:
+                    if c.value is not None:
+                        cells[(c.row, c.column)] = _LightCell(c.row, c.column, c.value)
+            sheet._cells = cells
+            # 只读模式下 merged_cells 可能为空，尝试从 XML 补充
+            if not list(sheet.merged_cells.ranges) and self.zf:
+                import re as _re
+                sheet_path = self._get_sheet_xml_path(sheet.title)
+                if sheet_path and sheet_path in self.zf.namelist():
+                    NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
+                    with self.zf.open(sheet_path) as f:
+                        for _, elem in ET.iterparse(f, events=("end",)):
+                            if elem.tag == f"{{{NS}}}mergeCell":
+                                ref = elem.get("ref", "")
+                                if ref:
+                                    m = _re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", ref)
+                                    if m:
+                                        sheet.merged_cells.ranges.append(
+                                            _LightMergedRange(
+                                                int(m.group(2)), int(m.group(4)),
+                                                self._col_letters_to_num(m.group(1)),
+                                                self._col_letters_to_num(m.group(3)),
+                                            )
+                                        )
+                                elem.clear()
+
+        self.table_image_map = collections.defaultdict(list)
+        for image_info in self.sheet_images:
+            anchor = image_info["anchor"]
+            if anchor[0] is None or anchor[1] is None:
+                continue
+            self.table_image_map[anchor].append(
+                f'<img src="{image_info["base64"]}" />'
+            )
 
-            used_cells, visual_artifacts = self._find_tables_in_sheet(sheet)
+        used_cells, visual_artifacts = self._find_tables_in_sheet(sheet)
+        if is_worksheet:
             visual_artifacts.extend(self._find_charts_in_sheet(sheet))
-            for _, _, block in sorted(
-                visual_artifacts,
-                key=lambda item: (item[0][0], item[0][1], item[1]),
-            ):
-                self.cur_page.append(block)
-            self._find_images_in_sheet(used_cells)  # 提取图片
+        for _, _, block in sorted(
+            visual_artifacts,
+            key=lambda item: (item[0][0], item[0][1], item[1]),
+        ):
+            self.cur_page.append(block)
+        self._find_images_in_sheet(used_cells)
 
     @staticmethod
     def _serialize_sheet_image(image: XlsImage) -> str:
@@ -477,7 +988,9 @@ class XlsxConverter:
     ) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int, dict]]]:
         used_cells = set()
         visual_artifacts = []
-        if self.workbook is not None:
+        # 轻量级 sheet（_LightweightSheet）无需 workbook 也可识别表格；
+        # openpyxl Worksheet/ReadOnlyWorksheet 需要 workbook 已加载
+        if isinstance(sheet, _LightweightSheet) or self.workbook is not None:
             content_layer = self._get_sheet_content_layer(sheet)  # 检测工作表的可见性
             tables = self._find_data_tables(sheet)  # 检测工作表中的所有数据表格
 
@@ -587,7 +1100,7 @@ class XlsxConverter:
 
     def _build_excel_cell(
         self,
-        sheet: Worksheet,
+        sheet,
         display_row: int,
         display_col: int,
         source_row: int,
@@ -600,12 +1113,18 @@ class XlsxConverter:
         cell_text = ""
         text_is_html = False
         media_content = []
-        if "DISPIMG" in raw_cell_text:
+
+        if isinstance(sheet, _LightweightSheet):
+            # 轻量级路径：纯文本，无样式/富文本/DISPIMG
+            cell_text = html.escape(raw_cell_text) if raw_cell_text else ""
+            text_is_html = bool(raw_cell_text)
+        elif "DISPIMG" in raw_cell_text:
             cell_image = self._get_cell_image(raw_cell_text)
             if cell_image:
                 media_content.append(cell_image)
         else:
             cell_text, text_is_html = self._cell_value_to_html(cell)
+
         media_content.extend(self.table_image_map.get((source_row, source_col), []))
 
         return ExcelCell(
@@ -614,7 +1133,7 @@ class XlsxConverter:
             text=cell_text,
             row_span=row_span,
             col_span=col_span,
-            styles=self._extract_cell_style(cell),
+            styles={} if isinstance(sheet, _LightweightSheet) else self._extract_cell_style(cell),
             media=media_content,
             text_is_html=text_is_html,
             source_row=source_row,
@@ -931,7 +1450,8 @@ class XlsxConverter:
     def _select_best_gap_candidate(
         self, sheet: Worksheet
     ) -> tuple[int, float, list[ExcelTable]]:
-        candidates = []
+        """逐候选值串行执行，每次 GC，峰值内存 1x 而非 Nx。"""
+        best = None
         for gap_tolerance in AUTO_GAP_TOLERANCE_CANDIDATES:
             raw_tables = self._find_data_tables_with_gap_raw(sheet, gap_tolerance)
             summary = self._summarize_candidate_tables(raw_tables)
@@ -943,36 +1463,23 @@ class XlsxConverter:
                 + 0.5 * float(summary["weighted_blank_ratio"])
                 + 1.0 * float(summary["row_overlap_excess_ratio"])
             )
-            candidates.append(
-                {
-                    "gap_tolerance": gap_tolerance,
-                    "penalty": penalty,
-                    "tables": self._filter_semantic_subset_tables(raw_tables),
-                    **summary,
-                }
-            )
-
-        min_penalty = min(float(candidate["penalty"]) for candidate in candidates)
-        near_best_candidates = [
-            candidate
-            for candidate in candidates
-            if float(candidate["penalty"])
-            <= (min_penalty + AUTO_GAP_TOLERANCE_PREFERENCE_MARGIN)
-        ]
+            tables = self._filter_semantic_subset_tables(raw_tables)
+            candidate = {
+                "gap_tolerance": gap_tolerance,
+                "penalty": penalty,
+                "tables": tables,
+                **summary,
+            }
+            if best is None or penalty < best["penalty"]:
+                best = candidate
+            # 释放当前候选值的中间结果，GC 后再跑下一个
+            del raw_tables, summary, tables, candidate
+            gc.collect()
 
-        best_candidate = min(
-            near_best_candidates,
-            key=lambda candidate: (
-                int(candidate["severe_separator_count"]),
-                AUTO_GAP_TOLERANCE_PREFERENCE[int(candidate["gap_tolerance"])],
-                float(candidate["interior_blank_line_ratio"]),
-                float(candidate["penalty"]),
-            ),
-        )
         return (
-            int(best_candidate["gap_tolerance"]),
-            float(best_candidate["penalty"]),
-            best_candidate["tables"],
+            int(best["gap_tolerance"]),
+            float(best["penalty"]),
+            best["tables"],
         )
 
     def _select_best_tables(self, sheet: Worksheet) -> list[ExcelTable]:
@@ -1062,7 +1569,9 @@ class XlsxConverter:
         return "\n".join(lines)
 
     def _find_images_in_sheet(self, used_cells: set[tuple[int, int]] = None):
-        if self.workbook is not None:
+        # 轻量级 sheet（_LightweightSheet）无需 workbook 也可输出图片；
+        # openpyxl Worksheet/ReadOnlyWorksheet 需要 workbook 已加载
+        if isinstance(self.sheet_images, list) and self.sheet_images:
             for image_info in self.sheet_images:
                 r, c = image_info["anchor"]
                 if (
@@ -1151,20 +1660,23 @@ class XlsxConverter:
     def _find_true_data_bounds(self, sheet: Worksheet) -> DataRegion:
         """查找工作表中真实的数据边界（最小/最大行列）。
 
-        该函数扫描所有单元格，找到包含所有非空单元格或合并单元格区域的
-        最小矩形范围，返回边界的行列索引。
+        该函数扫描所有单元格，找到包含所有非空单元格的最小矩形范围。
+        注意：合并单元格区域不参与整体边界计算，避免异常文件
+        （如 18 万行合并区域但实际数据仅 38 行）导致边界过大、
+        洪水填充遍历百万级空单元格而 OOM。合并区域在表格识别阶段
+        通过 merged_lookup 仍会被正确处理（只要落在数据边界内）。
 
         参数：
             sheet: 待分析的工作表。
 
         返回：
-            覆盖所有数据和合并单元格的最小矩形区域 DataRegion。
+            覆盖所有非空单元格的最小矩形区域 DataRegion。
             若工作表为空，则默认返回 (1, 1, 1, 1)。
         """
         min_row, min_col = None, None
         max_row, max_col = 0, 0
 
-        # 遍历所有有值的单元格，动态更新边界
+        # 仅遍历有值的单元格确定边界
         for cell in sheet._cells.values():
             if cell.value is not None:
                 r, c = cell.row, cell.column
@@ -1173,17 +1685,6 @@ class XlsxConverter:
                 max_row = max(max_row, r)
                 max_col = max(max_col, c)
 
-        # 将合并单元格的范围也纳入边界计算
-        for merged in sheet.merged_cells.ranges:
-            min_row = (
-                merged.min_row if min_row is None else min(min_row, merged.min_row)
-            )
-            min_col = (
-                merged.min_col if min_col is None else min(min_col, merged.min_col)
-            )
-            max_row = max(max_row, merged.max_row)
-            max_col = max(max_col, merged.max_col)
-
         # 若工作表中没有任何数据，默认返回 (1, 1, 1, 1)
         if min_row is None or min_col is None:
             min_row = min_col = max_row = max_col = 1
@@ -1554,45 +2055,53 @@ class XlsxConverter:
         return plain_text, False
 
     def _extract_cell_style(self, cell):
-        """Extract styles from an openpyxl cell."""
+        """Extract styles from an openpyxl cell. 兼容只读模式的 cell（无 font/alignment/fill）。"""
         style = {}
-        if cell.font:
-            if cell.font.b:
+        font = getattr(cell, "font", None)
+        if font:
+            if getattr(font, "b", False):
                 style["font-weight"] = "bold"
-            if cell.font.i:
+            if getattr(font, "i", False):
                 style["font-style"] = "italic"
-            if cell.font.u:
+            if getattr(font, "u", None):
                 style["text-decoration"] = "underline"
-            if cell.font.strike:
+            if getattr(font, "strike", False):
                 style["text-decoration"] = "line-through"
+            font_color = getattr(font, "color", None)
             if (
-                cell.font.color
-                and hasattr(cell.font.color, "rgb")
-                and cell.font.color.rgb
+                font_color
+                and hasattr(font_color, "rgb")
+                and font_color.rgb
             ):
-                # Color might be ARGB "FF000000"
-                color = cell.font.color.rgb
+                color = font_color.rgb
                 if isinstance(color, str) and len(color) == 8:
                     style["color"] = "#" + color[2:]
                 elif isinstance(color, str):
                     style["color"] = "#" + color
 
-        if cell.alignment:
-            if cell.alignment.horizontal:
-                style["text-align"] = cell.alignment.horizontal
-            if cell.alignment.vertical:
-                style["vertical-align"] = cell.alignment.vertical
+        alignment = getattr(cell, "alignment", None)
+        if alignment:
+            if getattr(alignment, "horizontal", None):
+                style["text-align"] = alignment.horizontal
+            if getattr(alignment, "vertical", None):
+                style["vertical-align"] = alignment.vertical
 
-        if cell.fill and cell.fill.patternType == "solid" and cell.fill.fgColor:
-            # handle bg color
-            color = cell.fill.fgColor.rgb
+        fill = getattr(cell, "fill", None)
+        if fill and getattr(fill, "patternType", None) == "solid":
+            fg_color = getattr(fill, "fgColor", None)
             if (
-                hasattr(cell.fill.fgColor, "type")
-                and cell.fill.fgColor.type == "rgb"
-                and color
+                fg_color
+                and hasattr(fg_color, "rgb")
+                and fg_color.rgb
             ):
-                if isinstance(color, str) and len(color) == 8:
-                    style["background-color"] = "#" + color[2:]
+                color = fg_color.rgb
+                if (
+                    hasattr(fg_color, "type")
+                    and fg_color.type == "rgb"
+                    and color
+                ):
+                    if isinstance(color, str) and len(color) == 8:
+                        style["background-color"] = "#" + color[2:]
         return style
 
     @staticmethod
```

---

## 2. rapid_doc/model/xlsx/main.py

```diff
diff --git a/rapid_doc/model/xlsx/main.py b/rapid_doc/model/xlsx/main.py
index aea1294..895ce98 100644
--- a/rapid_doc/model/xlsx/main.py
+++ b/rapid_doc/model/xlsx/main.py
@@ -9,8 +9,8 @@ def convert_path(file_path: str):
         return convert_binary(fh)
 
 
-def convert_binary(file_binary: BinaryIO):
-    converter = XlsxConverter()
+def convert_binary(file_binary: BinaryIO, gap_tolerance: int | None = None):
+    converter = XlsxConverter(gap_tolerance=gap_tolerance)
     converter.convert(file_binary)
     return converter.pages
 
```

## 3. rapid_doc/backend/office/office_analyze.py

```diff
diff --git a/rapid_doc/backend/office/office_analyze.py b/rapid_doc/backend/office/office_analyze.py
index 8a48f3d..580c9b3 100644
--- a/rapid_doc/backend/office/office_analyze.py
+++ b/rapid_doc/backend/office/office_analyze.py
@@ -8,7 +8,8 @@ from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes
 
 def office_analyze(
         file_bytes,
-        image_writer=None
+        image_writer=None,
+        gap_tolerance: int | None = None,
 ):
     infer_start = time.time()
     file_type = guess_suffix_by_bytes(file_bytes)
@@ -22,7 +23,11 @@ def office_analyze(
         from rapid_doc.model.xlsx.main import convert_binary
     else:
         raise ValueError(f"Unsupported or unknown office file type: {file_type}")
-    results = convert_binary(file_stream)
+
+    if file_type == "xlsx":
+        results = convert_binary(file_stream, gap_tolerance=gap_tolerance)
+    else:
+        results = convert_binary(file_stream)
 
     infer_time = round(time.time() - infer_start, 2)
     safe_time = max(infer_time, 0.01)
```

## 4. rapid_doc/cli/common.py

```diff
diff --git a/rapid_doc/cli/common.py b/rapid_doc/cli/common.py
index 4a31fd7..c68ad21 100644
--- a/rapid_doc/cli/common.py
+++ b/rapid_doc/cli/common.py
@@ -363,6 +363,7 @@ def _process_office_doc(
         f_dump_orig_file=True,
         f_dump_content_list=True,
         f_make_md_mode=MakeMode.MM_MD,
+        gap_tolerance: int | None = None,
 ):
     need_remove_index = []
     for i, file_bytes in enumerate(pdf_bytes_list):
@@ -377,6 +378,7 @@ def _process_office_doc(
             middle_json, infer_result = office_analyze(
                 file_bytes,
                 image_writer=image_writer,
+                gap_tolerance=gap_tolerance,
             )
 
             f_draw_layout_bbox = False
@@ -429,6 +431,7 @@ def do_parse(
         f_dump_orig_file=f_dump_orig_pdf,
         f_dump_content_list=f_dump_content_list,
         f_make_md_mode=f_make_md_mode,
+        gap_tolerance=kwargs.get("gap_tolerance"),
     )
     for index in sorted(need_remove_index, reverse=True):
         del pdf_bytes_list[index]
@@ -488,6 +491,7 @@ async def aio_do_parse(
         f_dump_orig_file=f_dump_orig_pdf,
         f_dump_content_list=f_dump_content_list,
         f_make_md_mode=f_make_md_mode,
+        gap_tolerance=kwargs.get("gap_tolerance"),
     )
     for index in sorted(need_remove_index, reverse=True):
         del pdf_bytes_list[index]
```

## 5. docker/app.py

```diff
diff --git a/docker/app.py b/docker/app.py
index c6d3000..24fc5c9 100644
--- a/docker/app.py
+++ b/docker/app.py
@@ -169,6 +169,7 @@ async def file_parse(
     response_format_zip: bool = Form(False),
     start_page_id: int = Form(0),
     end_page_id: int = Form(99999),
+    gap_tolerance: Optional[int] = Form(None),
 ):
     """
     Parse files using RapidDoc - Compatible with official API
@@ -277,6 +278,7 @@ async def file_parse(
                 end_page_id=end_page_id,
                 layout_config = layout_config, ocr_config = ocr_config, formula_config = formula_config,
                 table_config = table_config, checkbox_config = checkbox_config, image_config = image_config,
+                gap_tolerance = gap_tolerance,
             )
 
         # 根据 response_format_zip 决定返回类型
```

## 6. docker/Dockerfile

```diff
diff --git a/docker/Dockerfile b/docker/Dockerfile
index 589cb65..cf829fb 100644
--- a/docker/Dockerfile
+++ b/docker/Dockerfile
@@ -28,14 +28,14 @@ RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/
 WORKDIR /app
 
 # 配置 pip 国内镜像与超时，降低大包下载超时概率
-ENV PIP_INDEX_URL=https://pypi.org/simple
+ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
 ENV PIP_DEFAULT_TIMEOUT=300
 ENV PIP_RETRIES=10
 
 # 安装 Python 依赖
 RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages && \
     python3 -m pip install --no-cache-dir --prefer-binary --break-system-packages \
-        --index-url https://pypi.org/simple \
+        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
         'rapid-doc[cpu,api]==0.9.9' && \
     python3 -m pip cache purge
 
@@ -49,6 +49,9 @@ RUN sed -i 's/\r$//' /app/start_with_env.sh && chmod +x /app/start_with_env.sh
 # 复制应用代码（最后复制，避免频繁变更影响缓存）
 COPY docker/app.py docker/file_converter.py docker/download_file.py docker/download_models.py docker/models_download_utils.py /app/
 
+# 覆盖 rapid_doc 包源码（本地修改版，覆盖 pip 安装的版本）
+COPY rapid_doc/ /app/rapid_doc/
+
 # 设置基础环境变量
 ENV PYTHONPATH=/app
 
```

## 7. docker/docker-compose.yml

```diff
diff --git a/docker/docker-compose.yml b/docker/docker-compose.yml
index 20284b7..2177064 100644
--- a/docker/docker-compose.yml
+++ b/docker/docker-compose.yml
@@ -2,6 +2,9 @@ services:
   rapid-doc-server:
     container_name: rapid-doc-server
     image: hzkitty/rapid-doc:0.9.9
+    build:
+      context: ..
+      dockerfile: docker/Dockerfile
     ports:
       - "8888:8888"
     environment:
@@ -9,4 +12,7 @@ services:
       #- PADDLEOCRVL_VERSION=v1.6
       #- PADDLEOCRVL_VL_REC_BACKEND=vllm-server
       #- PADDLEOCRVL_VL_VL_REC_SERVER_URL=http://localhost:8118/v1
+    # 内存限制：物理内存 10GB，物理+swap 总上限 30GB（即 swap 上限 20GB）
+    mem_limit: 10g
+    memswap_limit: 30g
     restart: always
```

## 8. .dockerignore（新增）

```
# 排除权重文件（download_models.py 会在镜像内下载到 /app/models）
# 注意：只排除 download_models.py 会重新下载的 layout/table/formula 权重
# resources/ 下的 OCR/orientation 权重和 magika 权重不在下载列表，必须保留
rapid_doc/model/layout/rapid_layout_self/models/*.onnx
rapid_doc/model/table/rapid_table_self/models/*.onnx
rapid_doc/model/formula/rapid_formula_self/models/*.onnx

# 排除缓存和无关产物
**/__pycache__
**/*.pyc
**/*.pyo
.git
.github
.idea
.vscode
*.egg-info
*.tar
*.tar.gz
*.zip
*.log
.DS_Store
debug/
tmp/
temp/
slurm_logs/
bench_tmp.py
package_rapiddoc_output.py
demo/
docs/
tests/
```
