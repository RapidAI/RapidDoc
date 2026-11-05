"""Host resource overrides preserve isolation and do not become character limits."""

import json
import os
import hashlib
import sys
from pathlib import Path
from rapid_doc_rust._host_context import host_call
from rapid_doc_rust import legacy_pipeline_json


@host_call
def run():
    return json.loads(
        legacy_pipeline_json("filter_layout", '{"layout_dets":[],"use_custom_ocr":false}')
    )


saved = os.environ.get("RAPIDDOC_RESOURCE_LIMITS")
try:
    os.environ["RAPIDDOC_RESOURCE_LIMITS"] = '{"max_json_nodes":1}'
    try:
        run()
    except Exception as error:
        assert "预算" in str(error), error
    else:
        raise AssertionError("Explicit node budget was ignored")
    os.environ["RAPIDDOC_RESOURCE_LIMITS"] = '{"max_json_bytes":2}'
    try:
        run()
    except Exception as error:
        assert "预算" in str(error), error
    else:
        raise AssertionError("Explicit byte budget was ignored")
    os.environ["RAPIDDOC_RESOURCE_LIMITS"] = '{"max_chars":100000}'
    try:
        run()
    except ValueError as error:
        assert "Invalid RAPIDDOC_RESOURCE_LIMITS" in str(error)
    else:
        raise AssertionError("A character limit was accepted as a resource option")
    os.environ.pop("RAPIDDOC_RESOURCE_LIMITS", None)
    assert run() == []
finally:
    if saved is None:
        os.environ.pop("RAPIDDOC_RESOURCE_LIMITS", None)
    else:
        os.environ["RAPIDDOC_RESOURCE_LIMITS"] = saved
from rapid_doc_rust._classification import collect_samples

root = Path(sys.argv[1]).resolve()
reference = json.loads(
    (root / "build/migration/native-character-capacity-multiple-objects.json").read_text(
        encoding="utf-8"
    )
)
data = (root / reference["source"]).read_bytes()
assert hashlib.sha256(data).hexdigest() == reference["sha256"]
sample = collect_samples(data, [0])[0]
assert sample["char_count"] == reference["char_count"] and sample["char_count"] > 100000, sample[
    "char_count"
]
assert sample["font_non_generated_char_counts"]["Helvetica"] == 100020
print(
    "Explicit resource limits, scope restoration and real PDF sampling above 100,000 characters passed"
)
from rapid_doc_rust._native import PdfEngine
from rapid_doc_rust import _prepare_pdf

library, pdf = _prepare_pdf(root / "demo/pdfs/示例1-论文模板.pdf", None, 256 * 1024 * 1024)


def no_model(role):
    raise AssertionError("Resource budget must reject before loading a model")


try:
    os.environ["RAPIDDOC_RESOURCE_LIMITS"] = '{"max_json_nodes":1}'
    # Native layout conversion validates the actual JSON tree before OCR/models.
    try:
        PdfEngine().parse(
            library,
            pdf,
            '{"end_page":1,"native_only":true,"docvortex_mode":"rule"}',
            no_model,
            lambda event: None,
        )
    except Exception as error:
        assert "预算" in str(error), error
    else:
        raise AssertionError("Native production engine ignored resource configuration")
finally:
    if saved is None:
        os.environ.pop("RAPIDDOC_RESOURCE_LIMITS", None)
    else:
        os.environ["RAPIDDOC_RESOURCE_LIMITS"] = saved
print("Native production engine enforces explicit resources before model resolution")
