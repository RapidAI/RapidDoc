"""User four-file run, full 03 OCR regression and native diagnostic events."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import hashlib, json, sys, time
from loguru import logger
import rapid_doc_rust as sdk
from rapid_doc_rust._runtime import LazySession, load_models

root = Path(sys.argv[1]).resolve()
reference_name = sys.argv[3] if len(sys.argv) > 3 else "user-four-pdfs-20261028.json"
assert Path(reference_name).name == reference_name
reference = json.loads((root / "build/migration" / reference_name).read_text(encoding="utf-8"))
baseline = None
if len(sys.argv) > 4:
    assert Path(sys.argv[4]).name == sys.argv[4]
    baseline = {
        c["source"]: c
        for c in json.loads((root / "build/migration" / sys.argv[4]).read_text(encoding="utf-8"))[
            "cases"
        ]
    }
models = load_models({}, {}, {}, {})
assert [
    models[k].threads
    for k in ["layout_model", "detector", "recognizer", "formula_model", "table_model"]
] == [0, 0, 0, 0, 4]


def forbidden(*args, **kwargs):
    raise AssertionError("Reported documents used Python model scheduling")


sdk.parse_pdf_models = forbidden
LazySession.session = forbidden
logs = []
token = logger.add(lambda message: logs.append(str(message)), level="DEBUG")
engine = sdk.RapidDoc(ocr_config={"seal_enable": False})
summaries = []
from rapid_doc_rust._native import PdfEngine

events = []


class ObservedEngine:
    def __init__(self):
        self.native = PdfEngine()

    def parse(self, library, data, request, resolver, progress):
        def observe(event):
            events.append(json.loads(event))
            return progress(event)

        return self.native.parse(library, data, request, resolver, observe)


engine._native_engine = ObservedEngine()
namespace = sys.argv[2] if len(sys.argv) > 2 else "reported-four"
assert namespace and all(c.isalnum() or c in "-_" for c in namespace)
try:
    for case in reference["cases"]:
        source = root / case["source"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        started = time.perf_counter()
        result = engine(source)
        elapsed = time.perf_counter() - started
        assert (
            len(result.middle_json["pdf_info"]) == len(case["expected"]["middle_json"]["pdf_info"])
            and result.markdown.strip()
        )
        destination = root / "build" / namespace / source.stem
        destination.mkdir(parents=True, exist_ok=True)
        actual = dict(
            markdown=result.markdown,
            middle_json=result.middle_json,
            content_list_json=result.content_list_json,
        )
        (destination / "actual.json").write_text(
            json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (destination / "document.md").write_text(result.markdown, encoding="utf-8")
        for name, png in result.images.items():
            (destination / name).write_bytes(png)
        if "assets" in case:
            import io
            from PIL import Image

            assets = []
            for name, png in sorted(result.images.items()):
                with Image.open(io.BytesIO(png)) as im:
                    assets.append(
                        dict(
                            name=name,
                            mode=im.mode,
                            size=list(im.size),
                            pixels_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                        )
                    )
            assert assets == case["assets"], source
        summary = dict(
            source=case["source"],
            pages=len(result.middle_json["pdf_info"]),
            seconds=elapsed,
            markdown_characters=len(result.markdown),
            assets=len(result.images),
            matches_user_python_markdown=result.markdown == case["expected"]["markdown"],
            matches_user_python_middle_json=result.middle_json == case["expected"]["middle_json"],
            matches_user_python_content_list=result.content_list_json
            == case["expected"]["content_list_json"],
        )
        if baseline is not None:
            prior = baseline[case["source"]]
            assert prior["sha256"] == case["sha256"]
            summary["matches_pre_change_rust_output"] = actual == prior["expected"]
            assert summary["matches_pre_change_rust_output"], source
        summaries.append(summary)
        (root / "build" / f"{namespace}-summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Reported document completed:", json.dumps(summary, ensure_ascii=False), flush=True)
finally:
    logger.remove(token)
assert "preparing pages 1-10" in "".join(logs)
assert "page 10/10 prepared" in "".join(logs)
assert "OCR recognition (crops 0/" in "".join(logs)
assert "ocr_recognition:" in "".join(logs)
assert "intra_op_threads=auto" in "".join(logs)
assert "intra_op_threads=4" in "".join(logs)
assert not any(n in sys.modules for n in ["numpy", "cv2", "shapely", "pypdfium2"])
assert all(
    s["matches_user_python_markdown"]
    and s["matches_user_python_content_list"]
    and (
        s["matches_user_python_middle_json"]
        or baseline is not None
        and s["matches_pre_change_rust_output"]
    )
    for s in summaries
), summaries
if baseline is not None:
    print(
        "Existing Python middle JSON differences retained:",
        [s["source"] for s in summaries if not s["matches_user_python_middle_json"]],
        flush=True,
    )
groups = []
for event in events:
    if event["stage"] == "batch":
        groups.append([])
    if groups:
        groups[-1].append(event["stage"])
for stages in groups:
    assert max(i for i, s in enumerate(stages) if s == "layout") < min(
        i for i, s in enumerate(stages) if s == "formula"
    )
    assert max(i for i, s in enumerate(stages) if s == "formula") < min(
        i for i, s in enumerate(stages) if s == "ocr_detection"
    )
    assert max(i for i, s in enumerate(stages) if s == "ocr_detection") < min(
        i for i, s in enumerate(stages) if s == "ocr_recognition"
    )
print(
    f"{len(summaries)} original user document outputs, full OCR ordering, original thread defaults, native logs and Rust-only pipeline passed"
)
