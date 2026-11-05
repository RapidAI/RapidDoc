"""Installed SDK lifecycle, result ownership and callback-safe close checks."""

import json
from pathlib import Path
import sys

from rapid_doc_rust import RapidDoc, RapidDocError
from rapid_doc_rust._native import DocumentEngine


ROOT = Path(sys.argv[1]).resolve()
SOURCE = ROOT / "tests/migration/inputs/synthetic.docx"


def expect_closed(operation):
    try:
        operation()
    except (RuntimeError, ValueError, RapidDocError) as error:
        assert "closed" in str(error), error
    else:
        raise AssertionError("Closed engine accepted an operation")


# Closing a lazy instance must not load models or read its supplied input.
engine = RapidDoc()
engine.close()
engine.close()
assert engine.closed
expect_closed(lambda: engine("missing.pdf"))
expect_closed(engine.warmup)
expect_closed(engine.__enter__)

# Results own their data and survive both normal and exceptional scope exits.
with RapidDoc() as engine:
    assert not engine.closed
    first = engine(SOURCE)
    second = engine(SOURCE)
    assert first.markdown == second.markdown
assert engine.closed
assert first.markdown and first.middle_json["pdf_info"]
failure = RuntimeError("caller failed")
try:
    with RapidDoc() as engine:
        engine(SOURCE)
        raise failure
except RuntimeError as error:
    assert error is failure
assert engine.closed

# A retained reference to the native object cannot revive its closed session.
native = DocumentEngine("{}", "{}")
native.close()
native.close()
expect_closed(native.loaded_model_roles)
expect_closed(lambda: native.set_config("{}"))
expect_closed(lambda: native.set_resources("{}"))
expect_closed(lambda: native.warmup("{}", lambda event: None, lambda device: None))
expect_closed(lambda: native.parse(b"", "{}", lambda event: None, lambda device: None))

# A progress callback cannot destroy the engine currently executing that callback.
with RapidDoc(
    layout_config={"model_type": "docvortex_rule"},
    formula_enable=False,
    table_enable=False,
) as engine:
    native = engine._get_native_engine()
    callbacks = []

    def progress(event):
        if callbacks:
            return
        try:
            engine.close()
        except ValueError as error:
            assert "already in use" in str(error), error
        else:
            raise AssertionError("An active engine was closed")
        assert not engine.closed
        callbacks.append(event["stage"])

    metadata, images, drawings = native.parse(
        (ROOT / "tests/migration/inputs/pdf_native.pdf").read_bytes(),
        "{}",
        progress,
        lambda device: None,
    )
    assert callbacks
    assert json.loads(metadata)["markdown"]
assert engine.closed
expect_closed(native.loaded_model_roles)

print("Python lifecycle passed: lazy/repeated close, context exits, owned results and busy callbacks")
