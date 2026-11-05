"""Real same-process PDFium lifecycle tests, isolated per import order."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import argparse
import concurrent.futures
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
MODES = ['python-first', 'rust-first', 'raw-first', 'external-python-first',
         'external-rust-first', 'environment', 'concurrent', 'python-shutdown']


def worker(mode, sdk_root):
    if sdk_root:
        sys.path.insert(0, str(sdk_root))
    source = ROOT/'demo/pdfs/demo1.pdf'
    scan = ROOT/'demo/pdfs/small_ocr.pdf'
    chinese = ROOT/'demo/pdfs/示例1-论文模板.pdf'
    import rapid_doc_rust as sdk
    from PIL import Image
    raw_location = Path(importlib.util.find_spec('pypdfium2_raw').origin).parent
    shared = raw_location/('pdfium.dll' if sys.platform == 'win32' else
                            ('libpdfium.dylib' if sys.platform == 'darwin' else 'libpdfium.so'))
    override = str(shared) if mode.startswith('external-') else None
    if mode == 'environment':
        os.environ['RAPIDDOC_PDFIUM_LIBRARY'] = str(shared)

    def rust(path=source):
        parsed = sdk.parse_pdf_text(path, end_page=1, allow_partial=True, pdfium_library=override)
        rendered = sdk.render_pdf_page(path, dpi=72, color_mode='RGB', pdfium_library=override)
        with Image.open(io.BytesIO(rendered.png)) as image:
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
        return dict(document=hashlib.sha256(json.dumps(parsed.document, sort_keys=True).encode()).hexdigest(),
                    text=parsed.text, width=rendered.width, height=rendered.height, pixels=pixels)

    def python(path=source):
        with pdfium.PdfDocument(path) as document:
            page = document[0]
            textpage = page.get_textpage()
            bitmap = page.render(scale=1)
            pil = bitmap.to_pil()
            result = dict(count=len(document), text=textpage.get_text_range(), size=pil.size,
                          pixels=hashlib.sha256(pil.convert('RGB').tobytes()).hexdigest())
            pil.close()
            bitmap.close()
            textpage.close()
            page.close()
            return result

    python_first = mode in ('python-first', 'external-python-first', 'environment', 'python-shutdown')
    if python_first:
        import pypdfium2 as pdfium
        before_python = python()
    elif mode == 'raw-first':
        import pypdfium2_raw
    before_rust = rust()
    import pypdfium2 as pdfium
    if not python_first:
        before_python = python()
    assert rust() == before_rust
    assert python() == before_python
    # Keep Python document/page/text/bitmap handles alive across Rust calls.
    with pdfium.PdfDocument(source) as document:
        page = document[0]
        textpage = page.get_textpage()
        bitmap = page.render(scale=1)
        before_text = textpage.get_text_range()
        image = bitmap.to_pil()
        before_pixels = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        image.close()
        for _ in range(3):
            assert rust() == before_rust
            assert textpage.get_text_range() == before_text
            image = bitmap.to_pil()
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() == before_pixels
            image.close()
        bitmap.close()
        textpage.close()
        page.close()
    scan_rust = rust(scan)
    scan_python = python(scan)
    assert rust(scan) == scan_rust and python(scan) == scan_python
    chinese_rust = rust(chinese)
    chinese_python = python(chinese)
    assert rust(chinese) == chinese_rust and python(chinese) == chinese_python

    if mode == 'concurrent':
        barrier = threading.Barrier(2)
        def repeated(function, expected):
            barrier.wait(timeout=20)
            for _ in range(5):
                assert function() == expected
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(repeated, rust, before_rust), pool.submit(repeated, python, before_python)]
            for future in futures:
                future.result(timeout=90)
    if mode == 'python-shutdown':
        from pypdfium2 import _library_scope
        _library_scope.destroy_lib()
        try:
            assert rust() == before_rust
        finally:
            _library_scope.init_lib()
        assert python() == before_python
        assert rust() == before_rust

    from rapid_doc_rust._pdfium_runtime import resolve_pdfium_library
    library = Path(resolve_pdfium_library(override)).resolve()
    import pypdfium2_raw as raw
    own = ctypes.CDLL(str(library))
    own_address = ctypes.cast(own.FPDF_InitLibraryWithConfig, ctypes.c_void_p).value
    python_address = ctypes.cast(raw.FPDF_InitLibraryWithConfig, ctypes.c_void_p).value
    assert own_address != python_address, 'Both wrappers mapped the same PDFium instance'
    assert not library.samefile(shared)
    info = sdk.pdfium_font_runtime_info(pdfium_library=override)
    assert info['font_policy']
    return dict(mode=mode, sdk=sdk.__file__, rust_library=str(library), python_library=str(shared),
                distinct_function_addresses=True, live_handles_unchanged=True,
                native_and_scan_render_checks=True, repeated_outputs_identical=True,
                chinese_document_checked=True,
                concurrent_test=mode == 'concurrent', python_shutdown_test=mode == 'python-shutdown',
                font_policy=info['font_policy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=MODES)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists()
    if args.worker:
        result = worker(args.worker, args.sdk_root.resolve() if args.sdk_root else None)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in MODES:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                   '--output', str(output/(mode+'.json'))]
        if args.sdk_root:
            command.extend(['--sdk-root', str(args.sdk_root.resolve())])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=150)
        (output/(mode+'.stdout.txt')).write_bytes(result.stdout)
        (output/(mode+'.stderr.txt')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{mode} failed ({result.returncode}); see {output/(mode+".stderr.txt")}')
        results.append(json.loads((output/(mode+'.json')).read_text()))
        print('Passed:', mode, flush=True)
    (output/'summary.json').write_text(json.dumps(dict(cases=results, all_passed=True), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
