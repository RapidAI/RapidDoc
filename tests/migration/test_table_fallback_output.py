"""Installed native package exports recovered prose without pretending it is a table."""
import json
import rapid_doc_rust as sdk
from rapid_doc_rust._native import legacy_pipeline_json


def test_prediction_keeps_prose_and_page_hints():
    result=json.loads(legacy_pipeline_json('apply_table_prediction',json.dumps(dict(
        table={'category_id':5,'layout_image_list':[]},html_code='<p>A &amp; B</p>',text_fallback=True,
        formulas=[{'bbox':[0,0,2,2]}],fill_images=[],scale=2))))
    assert result['text_fallback'] is True and result['html']=='<p>A &amp; B</p>'
    assert 'layout_image_list' not in result
    assert result['formula_boxes']==[[0,0,1,1]]


def test_native_markdown_and_content_list_preserve_recovered_text():
    span=dict(type='table',html='<p>A &amp; B</p>\n<p>$x$</p>\n<p>image-id</p>\n',text_fallback=True)
    block=dict(type='table',bbox=[0,0,100,100],blocks=[dict(type='table_body',lines=[dict(spans=[span])])])
    pages=[dict(page_idx=0,page_size=[100,100],para_blocks=[block],discarded_blocks=[])]
    for mode in ['mm_markdown','nlp_markdown']:
        text=sdk.render_model_pages(pages,make_mode=mode)
        assert 'A &amp; B' in text and '$x$' in text and 'image-id' in text
        assert '<table' not in text
    content=sdk.render_model_pages(pages,make_mode='content_list')
    assert content[0]['type']=='text' and 'A & B' in content[0]['text']
    # Turning off the marker preserves the existing NLP behavior for tables.
    span.pop('text_fallback');span['html']='<table><tr><td>A &amp; B</td></tr></table>'
    assert sdk.render_model_pages(pages,make_mode='nlp_markdown')==''
    assert '<table>' in sdk.render_model_pages(pages)


def test_native_fenced_text_matches_independent_python():
    from rapid_doc.model.table.fallback import markdown_prose
    for html in ['<p>A &amp; B\u200312</p>\n<p>C  34</p>\n', '<p>```</p>',
                 '<p>&lt;img&gt;</p>', '<p>$x$</p>', '<p><img src="images/a.png" /></p>']:
        span=dict(type='table',html=html,text_fallback=True)
        block=dict(type='table',bbox=[0,0,100,100],blocks=[dict(type='table_body',lines=[dict(spans=[span])])])
        pages=[dict(page_idx=0,page_size=[100,100],para_blocks=[block],discarded_blocks=[])]
        for mode in ['mm_markdown','nlp_markdown']:
            assert sdk.render_model_pages(pages,make_mode=mode)==markdown_prose(html)
        text=sdk.render_model_pages(pages,make_mode='content_list')[0]['text']
        assert not text.startswith('```text\n')
        assert '\u2003' not in text
