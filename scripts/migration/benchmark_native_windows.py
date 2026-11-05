"""Isolated CPU native-mode batch lifecycle and output/memory verification."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.migration.accept_pdf_delivery import Sampling, memory, snapshot, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("python", "rust"), required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "tests/fixtures/upstream_068/deep_structure_catalog.pdf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--start-page-id", type=int)
    parser.add_argument("--end-page-id", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = []

    def event(stage, **values):
        row = dict(stage=stage, seconds=time.perf_counter() - started, memory=memory(), **values)
        events.append(row)
        (args.output / "events.json").write_bytes(json.dumps(events, indent=2).encode())
        print("WINDOW", json.dumps(row), flush=True)

    started = time.perf_counter()
    if args.engine == "rust":
        from rapid_doc_rust import RapidDoc
        from rapid_doc_rust import _native
        original = _native.PdfEngine

        class Engine:
            def __init__(self):
                self.inner = original()

            def parse(self, library, data, request, resolve, progress):
                def record(raw):
                    value = json.loads(raw)
                    if value["stage"] in ("batch", "native_analysis", "layout_fallback"):
                        event(value["stage"], detail=value.get("detail"), page_index=value.get("page_index"))
                    return progress(raw)
                return self.inner.parse(library, data, request, resolve, record)

        _native.PdfEngine = Engine
    else:
        from rapid_doc import RapidDoc
        import rapid_doc.main as module
        original = module.pipeline_doc_analyze

        def record(*pos, **kwargs):
            event("batch", start_page=kwargs.get("start_page_id"), window=kwargs.get("pdf_pages_batch"))
            result = original(*pos, **kwargs)
            event("native_and_models_complete", start_page=kwargs.get("start_page_id"))
            return result

        module.pipeline_doc_analyze = record
    # Native pipeline comparison, without formula or seal inference, using the
    # public default 64-page window; table/OCR fallbacks remain enabled.
    engine = RapidDoc(layout_config={"model_type": "docvortex_rule"}, parse_method="txt",
                      formula_enable=False, ocr_config={"seal_enable": False}, checkbox_config={})
    assert engine.pdf_pages_batch == 64
    event("before_parse", window=engine.pdf_pages_batch)
    samples = Sampling()
    try:
        with samples:
            selection = {name: value for name, value in
                         (("start_page_id", args.start_page_id), ("end_page_id", args.end_page_id))
                         if value is not None}
            result = engine(args.source, **selection)
    except Exception as error:
        failure = dict(engine=args.engine, source=str(args.source.resolve()), window=64,
                       success=False, error_type=type(error).__name__, error=str(error),
                       seconds=time.perf_counter() - started, memory=samples.summary(), events=events)
        (args.output / "report.json").write_bytes(json.dumps(failure, ensure_ascii=False, indent=2).encode())
        raise
    parsed = time.perf_counter() - started
    event("after_parse")
    data = snapshot(result)
    (args.output / "snapshot.json").write_bytes(json.dumps(data, ensure_ascii=False).encode())
    report = dict(engine=args.engine, source=str(args.source.resolve()), window=64,
                  pages=len(result.middle_json["pdf_info"]), seconds_including_initialization=parsed,
                  memory=samples.summary(), events=events)
    if args.engine == "rust":
        batches = [e for e in events if e["stage"] == "batch"]
        native = [e for e in events if e["stage"] == "native_analysis"]
        assert len(batches) == len(native) == (report["pages"] + 63) // 64
        first = args.start_page_id or 0
        assert [e["page_index"] for e in native] == list(range(first, first + report["pages"], 64))
    if args.reference:
        expected = json.loads((args.reference / "snapshot.json").read_bytes())
        report["comparison"] = comparison(expected, data)
    (args.output / "report.json").write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode())
    print("COMPLETE", args.engine, report["pages"], "pages; peak working set GiB:",
          round(report["memory"]["observed_peak_working_set"] / 1024 ** 3, 3), flush=True)


if __name__ == "__main__":
    main()
