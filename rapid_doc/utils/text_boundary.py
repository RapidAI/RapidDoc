"""Text spacing and paragraph script rules without a language model."""
from rapid_doc._vendor.docvortex_core.foundation._text import (
    _is_unspaced_character,
    merge_text_line_contents,
    resolve_text_line_boundary,
)
from rapid_doc._vendor.docvortex_core.foundation.language import remove_invalid_surrogates


def uses_unspaced_script(text: str) -> bool:
    """Choose the CJK paragraph threshold by letters, not punctuation or Hangul.

    Mixed paragraphs are classified by the majority of their alphabetic glyphs;
    individual text joins still use their actual boundary characters.
    """
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(_is_unspaced_character(char) for char in letters) * 2 >= len(letters)


__all__ = ["merge_text_line_contents", "resolve_text_line_boundary", "uses_unspaced_script", "remove_invalid_surrogates"]
