"""Keep PDFium lifecycle and font callbacks private to the Rust engine.

The packaged library already has a distinct name and file. Explicit libraries
are copied to a content-addressed private file before loading, regardless of
whether pypdfium2 has been imported yet. No pypdfium2 initialization hooks change.
"""

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

_NAMES = {"win32": "rapiddoc_pdfium.dll", "darwin": "librapiddoc_pdfium.dylib"}
_NAME = _NAMES.get(sys.platform, "librapiddoc_pdfium.so")
_PACKAGED = Path(__file__).parent / "runtime" / _NAME
_LOCK = threading.Lock()
_COPIES = {}


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _cache_root():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "rapid_doc_rust" / "pdfium"


def resolve_pdfium_library(override=None):
    selected = override or os.environ.get("RAPIDDOC_PDFIUM_LIBRARY")
    if not selected:
        return str(_PACKAGED)
    source = Path(selected).expanduser().resolve(strict=True)
    if source == _PACKAGED.resolve():
        return str(source)
    # A pre-import check is insufficient: pypdfium2 may be imported later.
    # Always isolate an external library before its first Rust initialization.
    with _LOCK:
        if source in _COPIES:
            return str(_COPIES[source])
        checksum = _digest(source)
        directory = _cache_root() / checksum
        directory.mkdir(parents=True, exist_ok=True)
        stem, suffix = Path(_NAME).stem, Path(_NAME).suffix
        target = directory / (stem + "-" + checksum[:16] + suffix)
        if (
            not target.is_file()
            or target.is_symlink()
            or target.samefile(source)
            or _digest(target) != checksum
        ):
            descriptor, temporary = tempfile.mkstemp(prefix="copy-", suffix=suffix, dir=directory)
            os.close(descriptor)
            temporary = Path(temporary)
            try:
                shutil.copyfile(source, temporary)
                if _digest(temporary) != checksum:
                    raise RuntimeError("PDFium library changed while preparing an isolated copy")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        _COPIES[source] = target
        return str(target)
