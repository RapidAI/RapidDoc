"""The new features have explicit expectations, not just Rust/Python agreement."""
import json
from pathlib import Path

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.fib import parse_fib
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.parser import parse_doc_document
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make

ROOT=Path(__file__).resolve().parents[2]
CASES=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/doc_extensions.json').read_text(encoding='utf8'))['cases']

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_binary_extensions_full_contract(case):
    word=bytes.fromhex(case['word_hex']);table=bytes.fromhex(case['table_hex'])
    pages=DocConverter._document_pages(parse_doc_document(word,table,b'',parse_fib(word)))
    assert pages==case['pages']
    middle=result_to_middle_json(pages,None)
    assert middle==case['middle']
    for mode,expected in case['exports'].items():
        assert union_make(middle['pdf_info'],mode,'images')==expected
    if case['name']=='code-literal':
        assert pages[0][0]['content']=='  print(```)\n\treturn <&>'
        assert case['exports']['mm_markdown']=='````\n  print(```)\n\treturn <&>\n````'
    if case['name']=='linked-textboxes-comments':
        body=[b for b in pages[0] if b['type']=='text']
        assert [b['content'][0]['content'] for b in body[1:]]==['Box α.','Linked continuation.']
        assert any(b['type']=='page_footnote' and b['content'][0]['content']=='[comment 1] Review ' for b in pages[0])
        assert not any(b['type']=='page_footnote' for b in pages[1])
