"""Installed shared-engine concurrency; bounded fixtures, exact output comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from loguru import logger
from rapid_doc_rust import RapidDoc

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--record", action="store_true")
parser.add_argument("--report", type=Path)
args = parser.parse_args()
root = args.root.resolve()
inputs = root / "tests/migration/inputs"
names = [f"synthetic.{ext}" for ext in ("doc", "docx", "xls", "xlsx", "ppt", "pptx")]
names += ["pdf_native.pdf", "pdf_scan.pdf", "pdf_mixed.pdf"]
logger.remove()


def signature(result):
    return dict(markdown=result.markdown, middle=result.middle_json,
                content=result.content_list_json,
                images={k: hashlib.sha256(v).hexdigest() for k, v in result.images.items()})


def rss():
    if os.name != "nt":
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak", "rss", "peak_paged", "paged",
                "peak_nonpaged", "nonpaged", "commit", "peak_commit", "private")]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    assert memory(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb)
    return values.rss


with RapidDoc(layout_config={"model_type": "docvortex_rule"},
              ocr_config={"seal_enable": False},
              formula_enable=False, table_enable=False) as engine:
    if args.record:
        expected = {name: signature(engine(inputs / name)) for name in names}
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
        print("Recorded pre-change output for", len(expected), "fixtures")
    else:
        expected = json.loads(args.reference.read_text(encoding="utf-8"))
        # Exclude model initialization from all throughput comparisons.
        for name in names:
            assert signature(engine(inputs / name)) == expected[name], name
        report = []
        for count in (1, 2, 4):
            samples = [rss()]
            stop = threading.Event()
            def sample():
                while not stop.wait(0.01):
                    samples.append(rss())
            monitor = threading.Thread(target=sample)
            monitor.start()
            started = time.perf_counter()
            try:
                with ThreadPoolExecutor(count) as pool:
                    for name, output in zip(names, pool.map(lambda name: signature(engine(inputs / name)), names)):
                        assert output == expected[name], name
            finally:
                seconds = time.perf_counter() - started
                stop.set()
                monitor.join()
            report.append(dict(workers=count, seconds=seconds, sampled_peak_rss_bytes=max(samples)))

        # Force both calls to be active before either can continue parsing.
        native = engine._get_native_engine()
        data = (inputs / "pdf_native.pdf").read_bytes()
        barrier = threading.Barrier(2, timeout=20)

        def parse_blocked():
            def progress(event):
                if event["stage"] == "document_opened":
                    barrier.wait()
            return native.parse(data, "{}", progress, lambda device: None)[0]

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(parse_blocked) for _ in range(2)]
            for future in futures:
                assert json.loads(future.result(timeout=60))["markdown"] == expected["pdf_native.pdf"]["markdown"]

        # Both requests must enter OCR preparation without holding the recognizer lock.
        barrier = threading.Barrier(2, timeout=20)
        scan = (inputs / "pdf_scan.pdf").read_bytes()

        def parse_scan():
            waited = False
            def progress(event):
                nonlocal waited
                if event["stage"] == "ocr_recognition" and event.get("completed") == 0 and not waited:
                    waited = True
                    barrier.wait()
            metadata = native.parse(scan, "{}", progress, lambda device: None)[0]
            assert waited, "Scan fixture did not exercise the recognizer"
            return json.loads(metadata)

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(parse_scan) for _ in range(2)]
            for future in futures:
                assert future.result(timeout=60)["markdown"] == expected["pdf_scan.pdf"]["markdown"]

        # Even a configuration snapshot swap must not hide an active request from close().
        entered, resume = threading.Event(), threading.Event()

        def held_parse():
            def progress(event):
                if event["stage"] == "document_opened":
                    entered.set()
                    assert resume.wait(20)
            return native.parse(data, "{}", progress, lambda device: None)

        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(held_parse)
            try:
                assert entered.wait(20)
                native.set_config(engine._engine_config())
                try:
                    engine.close()
                except ValueError as error:
                    assert "already in use" in str(error)
                else:
                    raise AssertionError("Active request lost its lifetime guard")
            finally:
                resume.set()
            pending.result(timeout=60)
        summary = dict(fixtures=len(names), warm_models=True, concurrency=report, overlap=True,
                       loaded_roles=native.loaded_model_roles())
        if args.report:
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
