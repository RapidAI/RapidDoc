"""OOXML 格式复用的 Open Packaging Conventions 基础能力。"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
import posixpath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree

CONTENT_TYPES_MEMBER = "[Content_Types].xml"
ROOT_RELS_MEMBER = "_rels/.rels"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PACKAGE_RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
RELATIONSHIP_TAG = f"{{{PACKAGE_RELATIONSHIPS_NS}}}Relationship"
OFFICE_DOCUMENT_REL_TAIL = "officeDocument"


WORDPROCESSINGML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STRICT_WORDPROCESSINGML_NS = "http://purl.oclc.org/ooxml/wordprocessingml/main"

# Strict OOXML 与 Transitional 共用的 URI 映射；格式专属命名空间（如 presentationml）由各
# normalizer 追加。前缀类映射必须放在更短的同前缀映射之前。
STRICT_OOXML_COMMON_REPLACEMENTS: tuple[tuple[bytes, bytes], ...] = (
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/metadata/thumbnail",
        b"http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/customProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/extendedProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    ),
    (
        b"http://purl.oclc.org/ooxml/drawingml/main",
        b"http://schemas.openxmlformats.org/drawingml/2006/main",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/math",
        b"http://schemas.openxmlformats.org/officeDocument/2006/math",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/customProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/extendedProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes",
        b"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/oleObject",
        b"http://schemas.openxmlformats.org/officeDocument/2006/oleObject",
    ),
)


def translate_strict_ooxml_uris(
    xml_bytes: bytes,
    replacements: Sequence[tuple[bytes, bytes]],
) -> bytes:
    """按给定映射把 Strict OOXML URI 字节替换为 Transitional URI。"""
    normalized = xml_bytes
    for strict_uri, transitional_uri in replacements:
        normalized = normalized.replace(strict_uri, transitional_uri)
    return normalized


def repair_content_type_overrides(
    member_data: dict[str, bytes],
    rel_tail_content_types: dict[str, str],
) -> bytes | None:
    """按关系图为缺失或错误的 Office part 补 [Content_Types].xml Override。

    非标准包的主文档与子 part 常没有 Override（有效内容类型退化为 application/xml），
    python-docx/python-pptx 无法识别。关系类型比 Content_Types 更可信，这里从根
    officeDocument 关系出发沿关系图补全；rel_tail_content_types 以关系类型尾段（如
    "officeDocument"、"slide"、"styles"）映射到期望内容类型。返回新的
    [Content_Types].xml 字节，无需修改时返回 None。
    """
    content_types_xml = member_data.get(CONTENT_TYPES_MEMBER)
    root_rels_xml = member_data.get(ROOT_RELS_MEMBER)
    if content_types_xml is None or root_rels_xml is None:
        return None

    try:
        parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
        content_types_root = etree.fromstring(content_types_xml, parser)
        root_rels_root = etree.fromstring(root_rels_xml, parser)
    except etree.XMLSyntaxError:
        return None

    overrides = _content_type_override_elements(content_types_root)
    try:
        pending = _collect_parts_needing_overrides(member_data, overrides, root_rels_root, rel_tail_content_types)
    except (etree.XMLSyntaxError, KeyError, RuntimeError):
        return None
    if not pending:
        return None

    for part_name, expected_ct in pending:
        override = overrides.get(part_name)
        if override is not None:
            override.set("ContentType", expected_ct)
        else:
            override = etree.SubElement(content_types_root, f"{{{CONTENT_TYPES_NS}}}Override")
            override.set("PartName", "/" + part_name)
            override.set("ContentType", expected_ct)
            overrides[part_name] = override

    return etree.tostring(
        content_types_root,
        xml_declaration=content_types_xml.lstrip().startswith(b"<?xml"),
        encoding="UTF-8",
        standalone=True,
    )


def _content_type_override_elements(content_types_root: etree._Element) -> dict[str, etree._Element]:
    """按去除前导斜杠的 PartName 建立已存在 Override 的索引。"""
    overrides: dict[str, etree._Element] = {}
    for override in content_types_root:
        if override.tag != f"{{{CONTENT_TYPES_NS}}}Override":
            continue
        part_name = (override.get("PartName") or "").replace("\\", "/").lstrip("/")
        if part_name:
            overrides[part_name] = override
    return overrides


def _collect_parts_needing_overrides(
    member_data: dict[str, bytes],
    overrides: dict[str, etree._Element],
    root_rels_root: etree._Element,
    rel_tail_content_types: dict[str, str],
) -> list[tuple[str, str]]:
    """从根 officeDocument 关系出发沿关系图收集需要补 Override 的 Office part。"""
    parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
    queue: list[tuple[str, str]] = []
    for relationship in root_rels_root:
        if not _is_relationship_element(relationship):
            continue
        rel_tail = relationship.get("Type", "").rsplit("/", 1)[-1]
        expected_ct = rel_tail_content_types.get(OFFICE_DOCUMENT_REL_TAIL)
        if rel_tail != OFFICE_DOCUMENT_REL_TAIL or expected_ct is None:
            continue
        target = _resolve_relationship_target(ROOT_RELS_MEMBER, relationship.get("Target"))
        if target and target in member_data:
            queue.append((target, expected_ct))

    pending: list[tuple[str, str]] = []
    visited: set[str] = set()
    while queue:
        part_name, expected_ct = queue.pop()
        if part_name in visited:
            continue
        visited.add(part_name)

        override = overrides.get(part_name)
        if override is None or override.get("ContentType") != expected_ct:
            pending.append((part_name, expected_ct))

        rels_name = _part_relationships_member(part_name)
        rels_xml = member_data.get(rels_name) if rels_name else None
        if rels_xml is None:
            continue
        rels_root = etree.fromstring(rels_xml, parser)
        for relationship in rels_root:
            if not _is_relationship_element(relationship):
                continue
            rel_tail = relationship.get("Type", "").rsplit("/", 1)[-1]
            expected_part_ct = rel_tail_content_types.get(rel_tail)
            if expected_part_ct is None:
                continue
            target = _resolve_relationship_target(rels_name, relationship.get("Target"))
            if target and target in member_data:
                queue.append((target, expected_part_ct))
    return pending


def _part_relationships_member(part_name: str) -> str | None:
    """根据 part 路径推导其关系成员路径（如 word/document.xml → word/_rels/document.xml.rels）。"""
    normalized = part_name.replace("\\", "/")
    if normalized in {"", "."}:
        return None
    part_dir, part_basename = posixpath.split(normalized)
    if not part_basename:
        return None
    if part_dir:
        return f"{part_dir}/_rels/{part_basename}.rels"
    return f"_rels/{part_basename}.rels"


def _is_relationship_element(element: etree._Element) -> bool:
    """判断元素是否为 Relationship，兼容缺省命名空间写法。"""
    if element.tag == RELATIONSHIP_TAG:
        return True
    try:
        return etree.QName(element).localname == "Relationship"
    except ValueError:
        return False


def _resolve_relationship_target(rels_filename: str, target: str | None) -> str | None:
    """把关系文件中的 Target 解析成 ZIP 包内的规范成员路径。"""
    if not target:
        return None

    target = target.replace("\\", "/")
    if target.startswith("/"):
        resolved = posixpath.normpath(target.lstrip("/"))
    else:
        base_dir = relationship_source_base_dir(rels_filename.replace("\\", "/"))
        if base_dir is None:
            return None
        resolved = posixpath.normpath(posixpath.join(base_dir, target))

    if resolved in {"", "."} or resolved.startswith("../"):
        return None
    return resolved


def write_zip_package(members: Sequence[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 ZIP 包，并由标准库重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()


def relationship_source_base_dir(rels_filename: str) -> str | None:
    """从规范 OPC relationship 成员路径推导源 part 所在目录。"""
    if rels_filename == "_rels/.rels":
        return ""

    marker = "/_rels/"
    if marker not in rels_filename:
        return None

    prefix, rels_basename = rels_filename.rsplit(marker, 1)
    if not rels_basename.endswith(".rels"):
        return None

    source_part_name = rels_basename[: -len(".rels")]
    source_part_path = posixpath.normpath(posixpath.join(prefix, source_part_name))
    return posixpath.dirname(source_part_path)


__all__ = [
    "CONTENT_TYPES_MEMBER",
    "OFFICE_DOCUMENT_REL_TAIL",
    "ROOT_RELS_MEMBER",
    "STRICT_OOXML_COMMON_REPLACEMENTS",
    "STRICT_WORDPROCESSINGML_NS",
    "WORDPROCESSINGML_NS",
    "relationship_source_base_dir",
    "repair_content_type_overrides",
    "translate_strict_ooxml_uris",
    "write_zip_package",
]
