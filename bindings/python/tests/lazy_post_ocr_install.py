"""Regression: nonempty document post-OCR must accept the API's lazy session."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import json
import sys
from pathlib import Path

from rapid_doc_rust import post_process_model_ocr
from rapid_doc_rust._host_context import host_call
from rapid_doc_rust._runtime import LazySession

root = Path(sys.argv[1]).resolve()
case = json.loads(
    (root / "build/migration/legacy-post-ocr-chain.json").read_text(encoding="utf-8")
)["cases"][0]
images = {key: root / path for key, path in case["images"].items()}
recognizer = LazySession("rec")


@host_call
def run(pages, recognizer):
    return post_process_model_ocr(
        pages, recognizer=recognizer, images=images, batch_size=case["batch_size"]
    )


# An empty plan must not force model loading.
assert run([], recognizer) == []
assert recognizer._session is None
lazy = run(case["pages"], recognizer)
assert recognizer._session is not None
assert lazy == run(case["pages"], recognizer.session())
assert any(
    span.get("content")
    for page in lazy
    for block in page["preproc_blocks"]
    for line in block["lines"]
    for span in line["spans"]
)
try:
    run(case["pages"], object())
except TypeError as error:
    assert str(error) == "recognizer must be OnnxSession"
else:
    raise AssertionError("Invalid recognizer was accepted")
print("Lazy/eager post-OCR equivalence, empty-plan laziness and invalid-session validation passed")
