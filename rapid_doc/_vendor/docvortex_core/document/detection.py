"""Structural OOXML/OLE detection only; no model or other format parsers."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

OLE2_SIG_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

OOXML_ROOT_RELS = "_rels/.rels"

OOXML_CONTENT_TYPES = "[Content_Types].xml"

OOXML_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

OOXML_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

OOXML_OFFICE_DOCUMENT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"

OOXML_OFFICE_DOCUMENT_RELS = frozenset(
    {
        OOXML_OFFICE_DOCUMENT_REL,
        "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument",
    }
)

OOXML_MAIN_CONTENT_TYPES = {
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"): "docx",
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"): "pptx",
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"): "xlsx",
}

OOXML_MAIN_PART_ROOT_TAG_SUFFIXES = {
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document": "docx",
    "{http://purl.oclc.org/ooxml/wordprocessingml/main}document": "docx",
    "{http://schemas.openxmlformats.org/presentationml/2006/main}presentation": "pptx",
    "{http://purl.oclc.org/ooxml/presentationml/main}presentation": "pptx",
    "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}workbook": "xlsx",
    "{http://purl.oclc.org/ooxml/spreadsheetml/main}workbook": "xlsx",
}

OOXML_CONVENTIONAL_MAIN_PARTS = ("word/document.xml", "ppt/presentation.xml", "xl/workbook.xml")

OLE2_STREAM_SUFFIX_MAP: dict[str, str] = {
    "WordDocument": "doc",
    "Workbook": "xls",
    "Book": "xls",
    "PowerPoint Document": "ppt",
}

def _strip_package_part_name(part_name: str | None) -> str:
    """规范化 OPC part 路径，方便匹配 Content_Types 中的 PartName。"""
    if not part_name:
        return ""
    return part_name.replace("\\", "/").lstrip("/")

def _ooxml_relationship_targets(root: ElementTree.Element) -> list[str]:
    """从根关系文件中提取 Office 主文档关系目标，兼容 Strict 与 Transitional 关系类型。"""
    targets = []
    for relationship in root:
        if relationship.tag not in {
            f"{{{OOXML_PACKAGE_REL_NS}}}Relationship",
            "Relationship",
        }:
            continue
        if relationship.get("TargetMode") == "External":
            continue
        if relationship.get("Type") not in OOXML_OFFICE_DOCUMENT_RELS:
            continue
        target = _strip_package_part_name(relationship.get("Target"))
        if target:
            targets.append(target)
    return targets

def _ooxml_content_type_overrides(root: ElementTree.Element) -> dict[str, str]:
    """读取 Content_Types 中每个显式 part 的 ContentType 映射。"""
    overrides = {}
    for override in root:
        if override.tag not in {
            f"{{{OOXML_CONTENT_TYPES_NS}}}Override",
            "Override",
        }:
            continue
        part_name = _strip_package_part_name(override.get("PartName"))
        content_type = override.get("ContentType")
        if part_name and content_type:
            overrides[part_name] = content_type
    return overrides

def _ooxml_main_part_suffix(package: ZipFile, part_name: str) -> str | None:
    """解析主文档 part 的实际根标签判型，兼容 Strict/Transitional 命名空间。"""
    try:
        with package.open(part_name) as stream:
            for _, root in ElementTree.iterparse(stream, events=("start",)):
                return OOXML_MAIN_PART_ROOT_TAG_SUFFIXES.get(root.tag)
    except (KeyError, ElementTree.ParseError, RuntimeError, OSError, ValueError):
        return None
    return None

def _guess_ooxml_suffix_from_zip(package: ZipFile) -> str | None:
    """根据 OOXML 包内标准主文档关系和主内容类型判断 Office 子类型。

    非标准包的 Content_Types Override 可能缺失或指向不存在的 part，关系类型也可能
    是 ISO Strict 变体；Override 判定失败时按主文档 part 根元素兜底，再退到惯例路径。
    """
    try:
        rels_root = ElementTree.fromstring(package.read(OOXML_ROOT_RELS))
    except (KeyError, ElementTree.ParseError):
        rels_root = None
    try:
        content_types_root = ElementTree.fromstring(package.read(OOXML_CONTENT_TYPES))
        overrides = _ooxml_content_type_overrides(content_types_root)
    except (KeyError, ElementTree.ParseError):
        overrides = {}

    targets = _ooxml_relationship_targets(rels_root) if rels_root is not None else []
    for target in targets:
        suffix = OOXML_MAIN_CONTENT_TYPES.get(overrides.get(target, ""))
        if suffix:
            return suffix
    for target in targets:
        suffix = _ooxml_main_part_suffix(package, target)
        if suffix:
            return suffix
    for part_name in OOXML_CONVENTIONAL_MAIN_PARTS:
        try:
            package.getinfo(part_name)
        except KeyError:
            continue
        suffix = _ooxml_main_part_suffix(package, part_name)
        if suffix:
            return suffix
    return None

def _guess_ooxml_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """优先用 OOXML 包结构识别 docx/pptx/xlsx，避免 Magika 被内嵌对象误导。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as package:
            return _guess_ooxml_suffix_from_zip(package)
    except (
        BadZipFile,
        KeyError,
        ElementTree.ParseError,
        RuntimeError,
        OSError,
        ValueError,
    ):
        return None

def _guess_ooxml_suffix_by_path(file_path: Path) -> str | None:
    """从文件路径读取 OOXML 包结构；失败时交给 Magika 原有逻辑兜底。"""
    try:
        with ZipFile(file_path) as package:
            return _guess_ooxml_suffix_from_zip(package)
    except (
        BadZipFile,
        KeyError,
        ElementTree.ParseError,
        RuntimeError,
        OSError,
        ValueError,
    ):
        return None

def _guess_ole2_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """用 OLE2 magic + olefile 内部 stream 区分 doc/xls/ppt。

    olefile 是纯 Python 库且已是核心依赖（mineru.model.flash.office.legacy 使用）。
    在 OOXML 识别失败后、Magika 兜底前插入此层，避免 Magika 对 OLE2 返回 unknown。
    """
    if len(file_bytes) < 8 or file_bytes[:8] != OLE2_SIG_BYTES:
        return None
    try:
        import olefile  # type: ignore[import-untyped]

        with olefile.OleFileIO(BytesIO(file_bytes)) as ole:
            for stream_name in ole.listdir(streams=True):
                name = "/".join(stream_name)
                suffix = OLE2_STREAM_SUFFIX_MAP.get(name)
                if suffix:
                    return suffix
    except Exception:
        return None
    return None

def _guess_ole2_suffix_by_path(file_path: Path) -> str | None:
    """从文件路径读取 OLE2 容器并识别旧 Office 格式。"""
    try:
        with open(file_path, "rb") as f:
            return _guess_ole2_suffix_by_bytes(f.read())
    except OSError:
        return None
