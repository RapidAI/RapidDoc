"""Structural file detection and lazy code-language classification."""
from functools import lru_cache
from pathlib import Path
from io import BytesIO

from rapid_doc._vendor.docvortex_core.document.detection import (
    _guess_ooxml_suffix_by_bytes, _guess_ooxml_suffix_by_path,
    _guess_ole2_suffix_by_bytes, _guess_ole2_suffix_by_path,
)

DEFAULT_LANG = "txt"
PDF_SIG_BYTES = b"%PDF"


def reject_encrypted_ooxml(file_bytes: bytes) -> None:
    """Encrypted modern Office uses OLE storage rather than a ZIP package."""
    if not file_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return
    import olefile
    try:
        with olefile.OleFileIO(BytesIO(file_bytes)) as ole:
            encrypted = ole.exists("EncryptedPackage") or ole.exists("EncryptionInfo")
    except OSError:
        return  # The selected parser reports malformed containers.
    if encrypted:
        from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeEncryptedError
        raise LegacyOfficeEncryptedError("Encrypted OOXML packages are not supported")


@lru_cache(maxsize=1)
def _get_magika():
    from rapid_doc.model.magika.magika import Magika
    return Magika()


def _normalize_text_for_language_guess(code: str) -> str:
    # Restore explicit surrogate pairs and drop lone halves.
    return code.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="ignore")


def guess_language_by_text(code: str) -> str:
    normalized = _normalize_text_for_language_guess(code)
    if not normalized:
        return DEFAULT_LANG
    try:
        label = _get_magika().identify_bytes(normalized.encode("utf-8")).prediction.output.label
    except Exception:
        return DEFAULT_LANG
    return label if label != "unknown" else DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes: bytes, file_path=None) -> str:
    if file_bytes.startswith(PDF_SIG_BYTES):
        return "pdf"
    reject_encrypted_ooxml(file_bytes)
    office = _guess_ooxml_suffix_by_bytes(file_bytes) or _guess_ole2_suffix_by_bytes(file_bytes)
    if office:
        return office
    return _get_magika().identify_bytes(file_bytes).prediction.output.label


def guess_suffix_by_path(file_path) -> str:
    path = Path(file_path)
    with path.open("rb") as stream:
        if stream.read(4) == PDF_SIG_BYTES:
            return "pdf"
    office = _guess_ooxml_suffix_by_path(path) or _guess_ole2_suffix_by_path(path)
    if office:
        return office
    with path.open("rb") as stream:
        if stream.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            reject_encrypted_ooxml(path.read_bytes())
    return _get_magika().identify_path(path).prediction.output.label
