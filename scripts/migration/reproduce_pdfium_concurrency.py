"""Reproduce PDFium concurrency failures without modifying either parser or locks.

Run each case in a separate subprocess: unsafe PDFium races can crash Python.
Inputs are read once and compared against each route's own serial baseline.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BUSINESS = Path("D:/sdgh/python/windows/agent/gpt-server")


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def load_rust():
    # Load only the abi3 package from the SDK environment; keep all other
    # dependencies in the selected interpreter (including business Python 3.12).
    package = ROOT / ".venv/Lib/site-packages/rapid_doc_rust"
    spec = importlib.util.spec_from_file_location(
        "rapid_doc_rust", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def child(args):
    faulthandler.enable(all_threads=True)
    os.environ["MINERU_PDF_CONCURRENCY_ENABLED"] = "false"
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    os.environ["RAPIDDOC_RUNTIME_DIR"] = str(ROOT / "build/go-sdk/windows-x86_64-cgo")
    # Business cases import its installed Python implementation, not our cwd.
    if args.implementation == "workspace":
        sys.path.insert(0, str(ROOT))
    files = [Path(p) for p in args.files]
    inputs = [(p.name, p.read_bytes()) for p in files]
    routes = {}
    identity = {
        "python": sys.version,
        "inputs": [
            {"file": str(p), "bytes": len(data), "sha256": checksum(data)}
            for p, (_, data) in zip(files, inputs)
        ],
    }
    if args.case in ("python", "mixed", "pipeline-python", "pipeline-mixed"):
        import pypdfium2 as pdfium
        from rapid_doc.utils import PyPDFium2Parser
        from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
        from rapid_doc._vendor.docvortex_core.document.pdf import pdfium as vendor

        # The installed code is intentionally untouched. Record both lock
        # identities, then initialize once before the baseline/warm stress run.
        identity.update(
            python_package=PyPDFium2Parser.__file__,
            same_pdfium_lock=PyPDFium2Parser.lock is vendor._pdfium_lock,
            pypdfium2=str(pdfium.PYPDFIUM_INFO),
            pdfium=str(pdfium.PDFIUM_INFO),
        )
        vendor.initialize_pdfium_runtime()

        def host(data):
            with PyPDFium2Parser.lock:
                with pdfium.PdfDocument(data) as doc:
                    with closing(doc[0]) as page:
                        with closing(page.get_textpage()) as text:
                            value = text.get_text_bounded()
                        with closing(page.render(scale=0.5)) as bitmap:
                            image = bitmap.to_pil().copy()
                        try:
                            return [len(doc), value, image.size, checksum(image.tobytes())]
                        finally:
                            image.close()

        def docvortex(data):
            with PDFDocument(data) as doc:
                chars = doc[0].get_chars()
                vectors = doc[0].get_vector_geometry()
                image = doc.render_page(0, scale=0.5).pil_image
                try:
                    return [len(doc), chars, repr(vectors), image.size, checksum(image.tobytes())]
                finally:
                    image.close()

        routes.update(host=host, docvortex=docvortex)
        if args.python_route != "both":
            routes = {args.python_route: routes[args.python_route]}
    if args.case in ("rust", "mixed", "pipeline-rust", "pipeline-mixed"):
        rust = load_rust()
        from rapid_doc_rust import _native
        from rapid_doc_rust._pdfium_runtime import resolve_pdfium_library

        library = resolve_pdfium_library()
        identity.update(rust_package=rust.__file__, rust_pdfium=library)

        def native(data):
            text = rust.parse_pdf_text(data, end_page=1, allow_partial=True)
            width, height, pixels = _native.render_pdf_page_bytes(library, data, 0, 36.0)
            return [text.document, width, height, checksum(pixels)]

        routes["rust"] = native
    if args.case.startswith("pipeline-"):
        routes.clear()
        local = threading.local()
        shared_engines = {}
        common = dict(
            layout_config={"model_type": "docvortex_rule"},
            formula_enable=False,
            table_enable=True,
            ocr_config={
                "seal_enable": False,
                "Det.model_path": str(ROOT / "rapid_doc/resources/PP-OCRv6_det_small.onnx"),
                "Rec.model_path": str(ROOT / "rapid_doc/resources/PP-OCRv6_rec_small.onnx"),
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            },
            # Both default to false. The installed Rust preview rejects the
            # explicit key before parsing; keep that separate from race tests.
            table_config={},
        )
        os.environ["RAPID_MODELS_DIR"] = str(ROOT.parent / "models")
        os.environ["USE_DOC_ORIENTATION_CLASSIFY"] = "false"

        def run_engine(kind, data):
            cache = shared_engines if args.shared_engine else getattr(local, "engines", None)
            if cache is None:
                cache = local.engines = {}
            if kind not in cache:
                if kind == "python":
                    from rapid_doc import RapidDoc
                else:
                    from rapid_doc_rust import RapidDoc
                cache[kind] = RapidDoc(**common)
            output = cache[kind](
                data, end_page_id=0, f_draw_layout_bbox=False, f_draw_span_bbox=False
            )
            return [
                output.markdown,
                output.content_list_json,
                {name: checksum(value) for name, value in output.images.items()},
            ]

        if args.case != "pipeline-rust":
            routes["python"] = lambda data: run_engine("python", data)
        if args.case != "pipeline-python":
            routes["rust"] = lambda data: run_engine("rust", data)

    # Timestamp/UUID-free evidence only. Different implementations need not
    # produce identical output; each must match its own serial reference.
    def digest(value):
        return checksum(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode())

    result = {
        "identity": identity,
        "case": args.case,
        "workers": args.workers,
        "shared_engine": args.shared_engine,
        "requested": args.tasks,
        "completed": 0,
        "failed": [],
        "mismatches": [],
        "baselines": {},
    }
    target = Path(args.result)

    def save():
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    for route, function in routes.items():
        for name, data in inputs:
            baseline = function(data)
            result["baselines"][route + ":" + name] = digest(baseline)
            target.with_name(target.stem + f".baseline-{route}-{name}.json").write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            save()
    names = list(routes)
    barrier = threading.Barrier(args.workers)
    count = 0
    counter_lock = threading.Lock()

    def work(index):
        nonlocal count
        if index < args.workers:
            barrier.wait(timeout=30)
        route = names[index % len(names)]
        name, data = inputs[(index // len(names)) % len(inputs)]
        print(json.dumps({"start": index, "route": route, "file": name}), flush=True)
        output = routes[route](data)
        value = digest(output)
        with counter_lock:
            count += 1
            if value != result["baselines"][route + ":" + name] and count < 50:
                target.with_name(target.stem + f".actual-{index}-{route}-{name}.json").write_text(
                    json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
                )
        return index, route, name, value

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(work, i): i for i in range(args.tasks)}
        for future in as_completed(pending):
            try:
                index, route, name, value = future.result()
                result["completed"] += 1
                if value != result["baselines"][route + ":" + name]:
                    result["mismatches"].append(
                        {"index": index, "route": route, "file": name, "actual": value}
                    )
            except BaseException as error:
                result["failed"].append(
                    {
                        "index": pending[future],
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            save()
    result["seconds"] = time.perf_counter() - started
    save()
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "failed": len(result["failed"]),
                "mismatches": len(result["mismatches"]),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--implementation", choices=["business", "workspace"], default="business")
    parser.add_argument(
        "--case",
        choices=["python", "rust", "mixed", "pipeline-python", "pipeline-rust", "pipeline-mixed"],
        default="python",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--shared-engine", action="store_true")
    parser.add_argument("--python-route", choices=["both", "host", "docvortex"], default="both")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            str(ROOT / "tests/migration/inputs" / name)
            for name in ("pdf_native.pdf", "pdf_mixed.pdf", "pdf_scan.pdf")
        ],
    )
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    result = Path(args.result).resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    python = (
        BUSINESS / ".venv/Scripts/python.exe"
        if args.implementation == "business"
        else ROOT / ".venv/Scripts/python.exe"
    )
    command = [
        str(python),
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--child",
        "--implementation",
        args.implementation,
        "--case",
        args.case,
        "--workers",
        str(args.workers),
        "--tasks",
        str(args.tasks),
        "--result",
        str(result),
        "--files",
        *args.files,
    ]
    if args.shared_engine:
        command.append("--shared-engine")
    command.extend(["--python-route", args.python_route])
    started = time.perf_counter()
    with result.with_suffix(".log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command, cwd=BUSINESS, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
            )
            exit_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            exit_code, timed_out = None, True
    receipt = {
        "exit_code": exit_code,
        "timeout": timed_out,
        "seconds": time.perf_counter() - started,
        "command": command,
    }
    result.with_suffix(".process.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
