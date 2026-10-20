"""核验当前 ctypes ABI 后借用函数地址；不重载运行库，不缓存文档或原生地址。"""

import ctypes as ct
from itertools import chain

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from ...._compute_backend import get_native
from ..pdfium import pdfium_guard

_CALLS = 0
_EMPTY_PAGES = 0
_UNAVAILABLE_REASON = "not probed"
_RECORD_BATCH_SIZE = None


def bridge_info():
    """报告真实完成的桥接次数和最近的能力探测结果。"""
    return {
        "pdfium_bridge_calls": _CALLS,
        "pdfium_empty_pages": _EMPTY_PAGES,
        "pdfium_bridge_unavailable_reason": _UNAVAILABLE_REASON,
        "pdfium_record_batch_size": _RECORD_BATCH_SIZE,
    }


def read_native_chars(textpage, extended, *, frame=None, rotation=0):
    """在同一 textpage 和锁内完成原始读取，特殊输入及非标准函数留给参考实现。"""
    global _CALLS, _EMPTY_PAGES, _UNAVAILABLE_REASON, _RECORD_BATCH_SIZE
    native = get_native()
    if native is None:
        _UNAVAILABLE_REASON = "python backend"
        return None
    if not hasattr(native, "read_pdfium_chars"):
        _UNAVAILABLE_REASON = "batch reader unavailable; rebuild extension"
        return None
    if type(textpage) is not pdfium.PdfTextPage or not textpage.raw or type(extended) is not bool:
        _UNAVAILABLE_REASON = "nonstandard or closed text page"
        return None
    # 零字符页没有需要借用的字符函数地址；在原锁内确认后直接返回空快照，
    # 单独计数，不能把未执行的 Rust FFI 调用记为桥接成功。
    with pdfium_guard():
        if textpage.count_chars() == 0:
            _EMPTY_PAGES += 1
            _UNAVAILABLE_REASON = None
            return (), ()
    args = (raw.FPDF_TEXTPAGE, ct.c_int)
    double_pointer = ct.POINTER(ct.c_double)
    specs = (
        ("FPDFText_GetUnicode", ct.c_uint, args),
        ("FPDFText_GetCharAngle", ct.c_float, args),
        ("FPDFText_GetLooseCharBox", ct.c_int, (*args, ct.POINTER(raw.FS_RECTF))),
        ("FPDFText_GetCharBox", ct.c_int, (*args, double_pointer, double_pointer, double_pointer, double_pointer)),
        ("FPDFText_GetFontInfo", ct.c_ulong, (*args, ct.c_void_p, ct.c_ulong, ct.POINTER(ct.c_int))),
        ("FPDFText_GetFontSize", ct.c_double, args),
        ("FPDFText_GetFontWeight", ct.c_int, args),
        ("FPDFText_GetTextObject", raw.FPDF_PAGEOBJECT, args),
        ("FPDFTextObj_GetTextRenderMode", ct.c_int, (raw.FPDF_PAGEOBJECT,)),
        ("FPDFText_GetCharOrigin", ct.c_int, (*args, double_pointer, double_pointer)),
        ("FPDFText_IsGenerated", ct.c_int, args),
    )
    if ct.sizeof(raw.FS_RECTF) != 16 or any(
        getattr(raw.FS_RECTF, name).offset != offset for name, offset in (("left", 0), ("top", 4), ("right", 8), ("bottom", 12))
    ):
        _UNAVAILABLE_REASON = "FS_RECTF ABI mismatch"
        return None
    functions, addresses = [], []
    for name, result, arguments in specs:
        function = getattr(raw, name, None)
        if (
            not isinstance(function, ct._CFuncPtr)
            or function.restype is not result
            or tuple(function.argtypes or ()) != arguments
            or getattr(function, "errcheck", None) is not None
        ):
            _UNAVAILABLE_REASON = f"unsupported symbol or ABI: {name}"
            return None
        functions.append(function)
        addresses.append(ct.cast(function, ct.c_void_p).value)
    # 本地列表在原生调用结束前保留函数强引用；textpage 参数保留页面及运行库的生命周期。
    with pdfium_guard():
        count = textpage.count_chars()
        if count < 0:
            _UNAVAILABLE_REASON = "negative character count"
            return None
        try:
            if frame is not None:
                result = native.read_pdfium_visual_batches(
                    addresses, ct.cast(textpage.raw, ct.c_void_p).value, count, extended, frame, rotation
                )
            else:
                reader = getattr(native, "read_pdfium_char_batches", native.read_pdfium_chars)
                result = reader(addresses, ct.cast(textpage.raw, ct.c_void_p).value, count, extended)
        except native.PdfiumReadError as exc:
            raise pdfium.PdfiumError(str(exc)) from exc
    _CALLS += 1
    _UNAVAILABLE_REASON = None
    records, fonts = result
    batches_types = tuple(
        kind
        for name in ("PdfiumCharacterBatches", "PdfiumVisualCharacterBatches")
        if (kind := getattr(native, name, None)) is not None
    )
    if isinstance(records, batches_types):
        _RECORD_BATCH_SIZE = native.PDFIUM_RECORD_BATCH_SIZE
        return chain.from_iterable(records), fonts
    _RECORD_BATCH_SIZE = None
    return result
