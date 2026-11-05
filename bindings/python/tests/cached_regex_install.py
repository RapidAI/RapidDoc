"""Original Python title contracts plus repeatable native rule timing."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import json
import os
import sys
import time
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
cases = json.loads(
    (root / "build/regex-reference-20261029/native-title-contracts.json").read_text(
        encoding="utf-8"
    )
)["cases"]


def same(actual, expected):
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            same(left, right)
    else:
        assert actual == expected, (actual, expected)


def run(case):
    q = case["request"]
    try:
        lines = sdk.classify_native_page_titles(
            q["lines"],
            q["page_size"],
            page_index=q["page_index"],
            container_bboxes=q["container_bboxes"],
            caption_container_bboxes=q.get("caption_container_bboxes"),
            body_profile=q.get("document_body_profile"),
            title_profile=q.get("document_title_profile"),
        )
    except sdk.RapidDocError as exc:
        assert "expected" not in case and exc.code == "invalid_input", case["label"]
    else:
        assert "expected" in case, case["label"]
        same(dict(lines=lines), case["expected"])


for case in cases:
    run(case)
samples = []
for _ in range(3):
    cpu = time.process_time()
    started = time.perf_counter()
    for case in cases:
        run(case)
    samples.append(
        dict(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
    )
result = dict(
    module=sdk.__file__, cases=len(cases), outputs_and_errors_match_reference=True, samples=samples
)
if os.environ.get("RAPIDDOC_REGEX_RESULT"):
    destination = Path(os.environ["RAPIDDOC_REGEX_RESULT"])
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
