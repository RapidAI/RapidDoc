"""Byte/range adapters for native classification. No classification rules here."""

import json
from loguru import logger
from ._host_context import host_call


@host_call
def classify(source, *, start_page=0, end_page=None):
    from . import _prepare_pdf
    from ._native import classify_pdf_json

    library, data = _prepare_pdf(source, None, None)
    result = json.loads(classify_pdf_json(library, data, start_page, end_page))
    if result.get("error"):
        logger.warning(
            "PDF classification selected OCR after an extraction failure: {}", result["error"]
        )
    return result["method"]


def get_sample_page_indices(count, max_pages=10):
    from ._native import classification_indices_json

    return json.loads(classification_indices_json(count, max_pages))


@host_call
def collect_samples(source, page_indices):
    from . import _prepare_pdf
    from ._native import classification_sample_json

    library, data = _prepare_pdf(source, None, None)
    return [json.loads(classification_sample_json(library, data, index)) for index in page_indices]
