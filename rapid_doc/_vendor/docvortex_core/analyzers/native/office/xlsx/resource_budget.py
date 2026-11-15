"""在 openpyxl 展开合并单元格前检查实际工作表的累计资源预算。"""

from __future__ import annotations

from collections.abc import Iterator
import posixpath
import re
from typing import Final

from lxml import etree as ET

from ..errors import OoxmlPackageError, LegacyOfficeResourceLimitError as ResourceLimitError
from .....foundation.ooxml_zip import OoxmlZipFile

MAX_MERGE_RANGE_CELLS: Final = 100_000
MAX_MERGED_CELLS: Final = 500_000
MAX_MERGE_RANGES: Final = 100_000
_MAX_EXCEL_ROW: Final = 1_048_576
_MAX_EXCEL_COLUMN: Final = 16_384
_CELL_RANGE = re.compile(r"\$?([A-Za-z]{1,3})\$?([0-9]{1,7})(?::\$?([A-Za-z]{1,3})\$?([0-9]{1,7}))?")
_SHEET_RELATIONSHIP_ATTRIBUTES: Final = frozenset(
    {
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
        "{http://purl.oclc.org/ooxml/officeDocument/relationships}id",
    }
)
_WORKBOOK_CONTENT_TYPES: Final = (
    "application/vnd.ms-excel.template.macroEnabled.main+xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml",
    "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
)


def _local_name(name: object) -> str:
    """忽略 Strict 与 Transitional 的命名空间差异取得 XML 本地名称。"""
    return name.rsplit("}", 1)[-1] if isinstance(name, str) else ""


def _read_control_xml(source: OoxmlZipFile, part_name: str) -> ET._Element | None:
    """读取已受共享控制 XML 限额约束的清单或关系文件。"""
    try:
        parser = ET.XMLParser(resolve_entities=False, load_dtd=False, no_network=True, huge_tree=False)
        root = ET.fromstring(source.read(part_name), parser=parser)
        if root.getroottree().docinfo.doctype:
            raise OoxmlPackageError(f"XLSX XML part must not contain a DTD: {part_name}")
        return root
    except KeyError:
        return None
    except ET.XMLSyntaxError as exc:
        raise OoxmlPackageError(f"Invalid XLSX XML part: {part_name}") from exc


def _resolve_target(source_part: str, target: str) -> str:
    """按 OPC 包内相对路径解析关系目标，不访问文件系统。"""
    if target.startswith("/"):
        return target[1:]
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def _workbook_part(source: OoxmlZipFile) -> str | None:
    """按 openpyxl 的内容类型优先级定位工作簿，并兼容根关系声明。"""
    content_types = _read_control_xml(source, "[Content_Types].xml")
    if content_types is not None:
        for content_type in _WORKBOOK_CONTENT_TYPES:
            for part in content_types:
                if _local_name(part.tag) == "Override" and part.get("ContentType") == content_type:
                    return (part.get("PartName") or "").lstrip("/")
        if any(
            _local_name(part.tag) == "Default" and part.get("ContentType") in _WORKBOOK_CONTENT_TYPES for part in content_types
        ):
            return "xl/workbook.xml"
    root_rels = _read_control_xml(source, "_rels/.rels")
    if root_rels is not None:
        for relationship in root_rels:
            if (relationship.get("Type") or "").rstrip("/").endswith("/officeDocument"):
                return _resolve_target("", relationship.get("Target") or "")
    return None


def _iter_part_attributes(
    source: OoxmlZipFile, part_name: str, *, parent_name: str, element_name: str
) -> Iterator[dict[str, str]]:
    """流式取出指定子元素属性，及时移除已处理节点以限制 XML 树大小。"""
    try:
        with source.open(part_name) as stream:
            stack: list[ET._Element] = []
            for event, element in ET.iterparse(
                stream, events=("start", "end"), resolve_entities=False, load_dtd=False, no_network=True, huge_tree=False
            ):
                if event == "start":
                    if not stack and element.getroottree().docinfo.doctype:
                        raise OoxmlPackageError(f"XLSX XML part must not contain a DTD: {part_name}")
                    if stack and _local_name(stack[-1].tag) == parent_name and _local_name(element.tag) == element_name:
                        yield dict(element.attrib)
                    stack.append(element)
                else:
                    stack.pop()
                    element.clear()
                    if stack:
                        stack[-1].remove(element)
    except ET.XMLSyntaxError as exc:
        raise OoxmlPackageError(f"Invalid XLSX XML part: {part_name}") from exc


def _worksheet_parts(source: OoxmlZipFile) -> Iterator[str]:
    """跟随实际工作簿的 sheet 引用，保留隐藏表和重复关系加载的计费。"""
    workbook_part = _workbook_part(source)
    if workbook_part is None or workbook_part not in source.NameToInfo:
        return
    directory, basename = posixpath.split(workbook_part)
    relationships = _read_control_xml(source, posixpath.join(directory, "_rels", basename + ".rels"))
    if relationships is None:
        return
    targets: dict[str, list[str]] = {}
    for relationship in relationships:
        relationship_id = relationship.get("Id")
        target = relationship.get("Target")
        if not relationship_id or not target or "chartsheet" in (relationship.get("Type") or ""):
            continue
        # openpyxl 对外部关系保留原 target；只有目标实际属于本 ZIP 才可能被加载。
        if relationship.get("TargetMode") != "External":
            target = _resolve_target(workbook_part, target)
        if target in source.NameToInfo:
            targets.setdefault(relationship_id, []).append(target)
    for sheet in _iter_part_attributes(source, workbook_part, parent_name="sheets", element_name="sheet"):
        for attribute, relationship_id in sheet.items():
            if attribute in _SHEET_RELATIONSHIP_ATTRIBUTES:
                yield from targets.get(relationship_id, ())


def _range_area(reference: str, part_name: str) -> int:
    """只接受 Excel 合法 A1 坐标，直接计算面积而不枚举或创建单元格。"""
    match = _CELL_RANGE.fullmatch(reference)
    if match is None:
        raise OoxmlPackageError(f"Invalid XLSX merged range in {part_name}: {reference}")
    first_column, first_row, last_column, last_row = match.groups()
    columns = []
    for column in (first_column, last_column or first_column):
        number = 0
        for character in column.upper():
            number = number * 26 + ord(character) - ord("A") + 1
        columns.append(number)
    min_row, max_row = int(first_row), int(last_row or first_row)
    min_column, max_column = columns
    if not (1 <= min_row <= max_row <= _MAX_EXCEL_ROW and min_column <= max_column <= _MAX_EXCEL_COLUMN):
        raise OoxmlPackageError(f"Invalid XLSX merged range in {part_name}: {reference}")
    return (max_row - min_row + 1) * (max_column - min_column + 1)


def validate_xlsx_merge_budget(source: OoxmlZipFile) -> None:
    """在第三方加载前累计全部工作表的合并面积和原始记录数，拒绝放大输入。"""
    total_cells = 0
    range_count = 0
    for part_name in _worksheet_parts(source):
        for attributes in _iter_part_attributes(source, part_name, parent_name="mergeCells", element_name="mergeCell"):
            range_count += 1
            if range_count > MAX_MERGE_RANGES:
                raise ResourceLimitError(f"XLSX exceeds max_merge_ranges={MAX_MERGE_RANGES}")
            area = _range_area(attributes.get("ref", ""), part_name)
            if area > MAX_MERGE_RANGE_CELLS:
                raise ResourceLimitError(f"XLSX merged range exceeds max_merge_range_cells={MAX_MERGE_RANGE_CELLS}")
            total_cells += area
            if total_cells > MAX_MERGED_CELLS:
                raise ResourceLimitError(f"XLSX exceeds max_merged_cells={MAX_MERGED_CELLS}")


__all__ = ["validate_xlsx_merge_budget"]
