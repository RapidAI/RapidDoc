"""在第三方 Office 解析器打开原包前验证 ZIP 的声明与实际解压预算。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct
from typing import BinaryIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo
import zlib

from ..analyzers.native.office.errors import (
    OoxmlPackageError,
    LegacyOfficeResourceLimitError as ResourceLimitError,
)

MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 4096 * 1024 * 1024
MAX_MEMBERS = 1_000_000
MAX_CENTRAL_DIRECTORY_BYTES = 32 * 1024 * 1024
MAX_CONTROL_XML_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_OOXML_MARKERS = frozenset(
    {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "xl/workbook.xml", "ppt/presentation.xml"}
)


class _NotZipFile(BadZipFile):
    """输入没有 ZIP 结束记录，允许文件类型识别继续尝试其他格式。"""


def _read_at(stream: BinaryIO, offset: int, size: int) -> bytes:
    """从明确位置读取小块结构，截断或负偏移按损坏 ZIP 处理。"""
    if offset < 0:
        raise BadZipFile("Invalid ZIP structure offset")
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise BadZipFile("Truncated ZIP structure")
    return data


def _zip64_record(stream: BinaryIO, locator_offset: int, declared_offset: int) -> tuple[int, bytes]:
    """解析 ZIP64 结束记录，并兼容自解压前缀导致的相对偏移。"""
    candidates = [declared_offset, locator_offset - 56]
    for offset in candidates:
        if offset < 0 or offset + 56 > locator_offset:
            continue
        record = _read_at(stream, offset, 56)
        if record[:4] != b"PK\x06\x06":
            continue
        size = struct.unpack_from("<Q", record, 4)[0]
        if size < 44 or offset + 12 + size != locator_offset:
            continue
        if size > MAX_CENTRAL_DIRECTORY_BYTES:
            raise ResourceLimitError("OOXML ZIP64 metadata exceeds the central-directory budget")
        return offset, record

    # 扩展 ZIP64 记录可带附加字段；分块向前寻找，不把整段元数据放进内存。
    end = locator_offset
    lower = max(0, locator_offset - MAX_CENTRAL_DIRECTORY_BYTES - 12)
    suffix = b""
    while end > lower:
        start = max(lower, end - _CHUNK_BYTES)
        data = _read_at(stream, start, end - start) + suffix
        cursor = len(data)
        while (index := data.rfind(b"PK\x06\x06", 0, cursor)) >= 0:
            cursor = index
            offset = start + index
            if offset + 56 > locator_offset:
                continue
            record = _read_at(stream, offset, 56)
            size = struct.unpack_from("<Q", record, 4)[0]
            if size >= 44 and offset + 12 + size == locator_offset:
                return offset, record
        suffix = data[:3]
        end = start
    raise BadZipFile("Missing ZIP64 end record")


def _directory_bounds(stream: BinaryIO) -> tuple[int, int, int]:
    """先读取有界 EOCD/ZIP64，再决定中央目录的位置和预算。"""
    stream.seek(0, 2)
    length = stream.tell()
    tail_size = min(length, 65535 + 22)
    tail = _read_at(stream, length - tail_size, tail_size)
    index = tail.rfind(b"PK\x05\x06")
    if index < 0 or index + 22 > len(tail):
        raise _NotZipFile("File is not a zip file")
    _, disk, directory_disk, disk_count, count, size, offset, comment_size = struct.unpack_from("<4s4H2IH", tail, index)
    if index + 22 + comment_size > len(tail):
        raise BadZipFile("Truncated ZIP comment")
    end = length - tail_size + index
    if disk or directory_disk:
        raise BadZipFile("Multi-disk ZIP packages are not supported")
    locator_offset = end - 20
    locator = _read_at(stream, locator_offset, 20) if locator_offset >= 0 else b""
    if locator[:4] == b"PK\x06\x07":
        _, zip64_disk, zip64_offset, disk_total = struct.unpack("<4sIQI", locator)
        if zip64_disk or disk_total != 1:
            raise BadZipFile("Multi-disk ZIP64 packages are not supported")
        end, record = _zip64_record(stream, locator_offset, zip64_offset)
        _, _, _, _, disk, directory_disk, disk_count, count, size, offset = struct.unpack("<4sQ2H2I4Q", record)
        if disk or directory_disk:
            raise BadZipFile("Multi-disk ZIP64 packages are not supported")
    elif size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        raise BadZipFile("Missing ZIP64 locator")
    if disk_count != count:
        raise BadZipFile("Inconsistent ZIP member counts")
    if size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ResourceLimitError(f"OOXML central directory exceeds {MAX_CENTRAL_DIRECTORY_BYTES} bytes")
    if count > MAX_MEMBERS:
        raise ResourceLimitError(f"OOXML package exceeds {MAX_MEMBERS} members")
    start = end - size
    if start < 0 or offset > start:
        raise BadZipFile("Invalid ZIP central-directory bounds")
    return start, end, count


def _check_directory_before_open(stream: BinaryIO) -> None:
    """在 ZipFile 分配目录对象前逐条计数，防止伪造 EOCD 数量绕过上限。"""
    position, end, declared_count = _directory_bounds(stream)
    count = 0
    while position < end:
        header = _read_at(stream, position, 46)
        if header[:4] != b"PK\x01\x02":
            raise BadZipFile("Invalid central-directory record")
        name_size, extra_size, comment_size = struct.unpack_from("<3H", header, 28)
        position += 46 + name_size + extra_size + comment_size
        if position > end:
            raise BadZipFile("Truncated central-directory record")
        count += 1
        if count > MAX_MEMBERS:
            raise ResourceLimitError(f"OOXML package exceeds {MAX_MEMBERS} members")
    if count != declared_count:
        raise BadZipFile("Inconsistent ZIP central-directory count")


def _member_limit(info: ZipInfo) -> int:
    """关系和内容类型 XML 使用较小上限，正文与工作表保留普通成员预算。"""
    if info.filename == "[Content_Types].xml" or info.filename.endswith(".rels"):
        return min(MAX_ENTRY_BYTES, MAX_CONTROL_XML_BYTES)
    return MAX_ENTRY_BYTES


def _compressed_limit(expanded_limit: int) -> int:
    """为压缩输入保留少量编码开销，同时限制第三方 read 的一次性分配。"""
    return expanded_limit + expanded_limit // 100 + _CHUNK_BYTES


class OoxmlZipFile(ZipFile):
    """只读 ZIP，在任何成员读取前验证声明和实际解压量。"""

    def __init__(
        self, file: str | Path | BinaryIO, *, ooxml_only: bool = False, validate_payloads: bool = True
    ) -> None:
        """保留原包供第三方读取，验证期间只保留固定大小的解压块。"""
        self._owned_stream: BinaryIO | None = None
        self.is_ooxml = False
        stream = file
        if isinstance(file, (str, Path)):
            stream = open(file, "rb")
            self._owned_stream = stream
        try:
            try:
                _check_directory_before_open(stream)
            except _NotZipFile:
                raise
            except BadZipFile as exc:
                # 已找到 ZIP 结构却未通过预检时，禁止其他判型器重新无约束地打开原包。
                raise OoxmlPackageError(f"Invalid ZIP directory: {exc}") from exc
            super().__init__(stream, mode="r")
            self.is_ooxml = any(name in self.NameToInfo for name in _OOXML_MARKERS)
            if not ooxml_only or self.is_ooxml:
                self._validate_members(validate_payloads=validate_payloads)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """关闭 ZIP 及自行打开的文件，同时保留调用者传入的流。"""
        if hasattr(self, "fp"):
            super().close()
        if self._owned_stream is not None:
            self._owned_stream.close()
            self._owned_stream = None

    def _validate_members(self, *, validate_payloads: bool) -> None:
        """先拒绝声明超限，再核对每个成员实际解压长度及整包累计量。"""
        infos = self.infolist()
        if len({info.filename for info in infos}) != len(infos):
            raise OoxmlPackageError("Duplicate OOXML ZIP member names")
        declared_total = 0
        compressed_total = 0
        for info in infos:
            if info.file_size > _member_limit(info):
                raise ResourceLimitError(f"OOXML member {info.filename!r} exceeds {_member_limit(info)} bytes")
            if info.compress_size > _compressed_limit(_member_limit(info)):
                raise ResourceLimitError(f"OOXML compressed member {info.filename!r} exceeds the input budget")
            declared_total += info.file_size
            if declared_total > MAX_TOTAL_BYTES:
                raise ResourceLimitError(f"OOXML expanded package exceeds {MAX_TOTAL_BYTES} bytes")
            compressed_total += info.compress_size
            if compressed_total > _compressed_limit(MAX_TOTAL_BYTES):
                raise ResourceLimitError("OOXML compressed package exceeds the input budget")
            if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                raise OoxmlPackageError("OOXML ZIP supports only Stored and Deflate compression")
        if not validate_payloads:
            # 判型只通过有界的 ZipFile 接口读控制 XML，实际解压校验在转换入口执行。
            return
        actual_total = 0
        for info in infos:
            actual_total += self._validate_member_stream(info, MAX_TOTAL_BYTES - actual_total)

    def _validate_member_stream(self, info: ZipInfo, remaining_total: int) -> int:
        """直接扫描原始压缩流，避免 ZipExtFile 被伪造 file_size 提前截断。"""
        stream = self.fp
        assert stream is not None
        try:
            header = _read_at(stream, info.header_offset, 30)
            if header[:4] != b"PK\x03\x04":
                return 0  # 保留现有 normalizer 对坏可选成员的容错。
            if info.flag_bits & 1:
                return 0  # 加密成员由原解析器报告口令/格式错误。
            name_size, extra_size = struct.unpack_from("<2H", header, 26)
            position = info.header_offset + 30 + name_size + extra_size
            if position < 0 or position + info.compress_size > self.start_dir:
                return 0
            stream.seek(position)
        except (BadZipFile, OSError):
            return 0
        remaining_compressed = info.compress_size
        actual_size = 0
        limit = min(_member_limit(info), remaining_total)
        inflater = zlib.decompressobj(-15) if info.compress_type == ZIP_DEFLATED else None
        while remaining_compressed:
            chunk = stream.read(min(_CHUNK_BYTES, remaining_compressed))
            if not chunk:
                return actual_size
            remaining_compressed -= len(chunk)
            while chunk:
                if inflater is None:
                    output = chunk
                    chunk = b""
                else:
                    try:
                        output = inflater.decompress(chunk, min(_CHUNK_BYTES, limit - actual_size + 1))
                    except zlib.error:
                        return actual_size
                    chunk = inflater.unconsumed_tail
                actual_size += len(output)
                if actual_size > limit:
                    raise ResourceLimitError(f"OOXML actual decompression exceeds the budget at {info.filename!r}")
                if inflater is not None and inflater.eof:
                    if inflater.unused_data or remaining_compressed:
                        raise OoxmlPackageError(f"Unexpected data after OOXML Deflate stream: {info.filename}")
                    break
        if inflater is not None and not inflater.eof:
            return actual_size  # 截断流由现有修复路径处理，但已展开部分仍计入预算。
        if actual_size != info.file_size:
            raise OoxmlPackageError(f"OOXML member size disagrees with decompressed data: {info.filename}")
        return actual_size


def validate_ooxml_package(source: bytes | str | Path | BinaryIO) -> None:
    """验证原始包并恢复调用者流位置，供第三方解析器随后重新打开。"""
    if isinstance(source, bytes):
        with OoxmlZipFile(BytesIO(source)):
            return
    if isinstance(source, (str, Path)):
        with OoxmlZipFile(source):
            return
    position = source.tell()
    try:
        with OoxmlZipFile(source):
            pass
    finally:
        source.seek(position)


__all__ = ["OoxmlZipFile", "validate_ooxml_package"]
