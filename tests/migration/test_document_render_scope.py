"""A production document must not re-enter the diagnostic renderer capacity."""
import os
from unittest.mock import patch
import pytest
import rapid_doc_rust as sdk


def test_document_render_scope_and_restoration():
    from rapid_doc_rust._host_context import host_call
    scoped_render = host_call(sdk.render_model_pages)
    pages = [dict(page_idx=0, page_size=[600,800], para_blocks=[], preproc_blocks=[],
                  discarded_blocks=[], metadata=[0]*200_001)]
    with pytest.raises(sdk.RapidDocError):
        sdk.render_model_pages(pages)
    assert scoped_render(pages) == ''
    assert scoped_render(pages, make_mode='content_list') == []
    with pytest.raises(sdk.RapidDocError):
        sdk.render_model_pages(pages)
    with patch.dict(os.environ, {'RAPIDDOC_RESOURCE_LIMITS':'{"max_json_bytes":8}'}):
        with pytest.raises(sdk.RapidDocError):
            scoped_render(pages)
