"""Unicode tools and lazy code-language classification; no natural-language model."""
import re
from contextlib import contextmanager

DEFAULT_CODE_LANGUAGE = "txt"
_INVALID_SURROGATES = re.compile("[\ud800-\udfff]")

def remove_invalid_surrogates(text: str) -> str:
    return _INVALID_SURROGATES.sub("", text)

def guess_code_language(code: str) -> str:
    from rapid_doc.utils.guess_suffix_or_lang import guess_language_by_text
    return guess_language_by_text(code)

@contextmanager
def _code_language_model_scope():
    yield
