from pathlib import Path
from loguru import logger
import io
import json
import re
import zipfile

DEFAULT_LANG = "txt"
PDF_SIG_BYTES = b"%PDF"
ZIP_SIG_BYTES = b"PK\x03\x04"


def guess_language_by_text(code: str) -> str:
    s = code.strip()
    if not s:
        return DEFAULT_LANG

    try:
        json.loads(s)
        return "json"
    except Exception:
        pass

    lower = s.lower()

    if lower.startswith("<!doctype html") or "<html" in lower:
        return "html"

    if s.startswith("<?xml"):
        return "xml"

    if re.search(r"^#{1,6}\s", s, re.M) or re.search(r"\[.+?\]\(.+?\)", s):
        return "markdown"

    if re.search(r"^\s*(import |from .+ import |def |class )", s, re.M):
        return "python"

    if re.search(r"^\s*(select|insert|update|delete|create|drop|alter)\b", s, re.I):
        return "sql"

    return DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes: bytes, file_path=None) -> str:
    if file_bytes.startswith(PDF_SIG_BYTES):
        return "pdf"

    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg"

    if file_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"

    if file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        return "webp"

    if file_bytes.startswith(ZIP_SIG_BYTES):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = set(zf.namelist())
                if "word/document.xml" in names:
                    return "docx"
                if "xl/workbook.xml" in names:
                    return "xlsx"
                if "ppt/presentation.xml" in names:
                    return "pptx"
            return "zip"
        except Exception:
            return "zip"

    text = file_bytes[:4096].decode("utf-8", errors="ignore").strip()
    if text:
        lang = guess_language_by_text(text)
        return lang if lang != DEFAULT_LANG else "txt"

    if file_path:
        ext = Path(file_path).suffix.lower().lstrip(".")
        if ext:
            return ext

    return "bin"


def guess_suffix_by_path(file_path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    ext = file_path.suffix.lower().lstrip(".")
    try:
        with open(file_path, "rb") as f:
            head = f.read(8192)
        suffix = guess_suffix_by_bytes(head, file_path=file_path)

        # 保留你原先对 pdf 的保护逻辑
        if ext == "pdf" and head[:4] == PDF_SIG_BYTES:
            return "pdf"

        return suffix or ext or "unknown"
    except Exception as e:
        logger.warning(f"Failed to identify file {file_path}: {e}")
        return ext or "unknown"