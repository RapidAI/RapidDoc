"""The host and vendored wrappers must serialize the same PDFium runtime."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from rapid_doc.utils import PyPDFium2Parser
from rapid_doc.utils.pdfium_guard import pdfium_guard as host_guard
from rapid_doc._vendor.docvortex_core.document.pdf import pdfium as vendor


def test_both_wrappers_share_one_lock():
    assert vendor._pdfium_lock is PyPDFium2Parser.lock


def test_vendor_cannot_enter_while_host_is_using_pdfium():
    started, entered, release = Event(), Event(), Event()

    def vendor_call():
        started.set()
        with vendor.pdfium_guard():
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with host_guard():
                future = pool.submit(vendor_call)
                assert started.wait(5)
                assert not entered.wait(0.1), "Both wrappers entered PDFium simultaneously"
            assert entered.wait(5), "Vendor route stayed blocked after host released the lock"
        finally:
            release.set()
        future.result(timeout=5)


def test_nested_font_initialization_and_exception_release_the_lock():
    with host_guard():
        with vendor.pdfium_guard():
            assert vendor.initialize_pdfium_runtime() is not None
    with pytest.raises(RuntimeError, match="test failure"):
        with vendor.pdfium_guard():
            with host_guard():
                raise RuntimeError("test failure")

    def next_call():
        with vendor.pdfium_guard():
            return True

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(next_call).result(timeout=5)
