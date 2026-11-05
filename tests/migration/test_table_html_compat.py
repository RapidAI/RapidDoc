"""Malformed model markup follows Python's real BeautifulSoup behavior."""
import json
import pytest
from rapid_doc_rust import legacy_pipeline_json
from rapid_doc.model.table.utils import normalize_table_html_cell_text, select_best_table_model


CASES = [
    '<table><tr><td</table>',
    '<table><tr><td>中 文</td><td</table>',
    '<table><tr><td>中 文</td</tr></table>',
    '<table><tr><td>中 文</td> <3 test> </tr></table>',
    '<table><tr><td>中 文</td> < 2 </tr></table>',
    '<table><tr><td>中 文</td><',
    '<table><tr><td>中 文</td><td',
    '<table><tr><td>中 文</td><td title="unfinished',
    '<table><tr><td>中 文</td><td/>',
    '<table><tr><td>中 文</td><td data-key="a/b">A &amp; B</td></tr></table>',
    '<table><tr><td>中 文</td><td/foo=bar>value</td></tr></table>',
]


def op(name, value):
    return json.loads(legacy_pipeline_json(name,json.dumps(value,ensure_ascii=False)))


@pytest.mark.parametrize('html', CASES)
def test_normalization_matches_python_without_inventing_grid(html):
    assert op('normalize_legacy_table_html',html)==normalize_table_html_cell_text(html)


@pytest.mark.parametrize('html', CASES)
def test_candidate_selection_matches_python_on_incomplete_tags(html):
    other='<table><tr><td>A &amp; B</td><td>中 文</td></tr></table>'
    rows=[[],['A &amp; B','中 文'],[1,1]]
    for wired, wireless in [(html,other),('',html),(html,'')]:
        actual=op('select_legacy_table',dict(texts=rows[1],wired_html=wired,wireless_html=wireless))
        assert actual['html']==select_best_table_model(rows,wired,wireless)
