"""Installed DOC stage acceptance: extensions, compatibility and pinned real corpus."""
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
from _mtef_test_utils import _build_nested_cfb
from rapid_doc import RapidDoc as PythonDoc
from rapid_doc_rust import RapidDoc as RustDoc, RapidDocError
from rapid_doc_rust._native import parse_office_bytes
from accept_docx_delivery import snapshot

def fixture(name):
    data=(ROOT/'tests/fixtures/office_rust_reference_068'/name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)

def binary_input(case):
    """Rebuild streams without depending on a previous Cargo test run."""
    streams={'WordDocument':case['word_hex'],'1Table':case['table_hex'],**case.get('pool',{})}
    if case.get('native_hex'):
        streams['ObjectPool/_100/Equation Native']=case['native_hex']
    entries=[dict(name='Root Entry',kind=5)]
    children={0:[]};storages={'':0}
    for path,payload in sorted(streams.items()):
        parts=path.split('/');parent=0
        for i,name in enumerate(parts[:-1]):
            prefix='/'.join(parts[:i+1])
            if prefix not in storages:
                index=len(entries);storages[prefix]=index
                entries.append(dict(name=name,kind=1));children[index]=[]
                children[parent].append(index)
            parent=storages[prefix]
        children[parent].append(len(entries))
        entries.append(dict(name=parts[-1],kind=2,data=bytes.fromhex(payload)))
    for parent,items in children.items():
        if items:entries[parent]['child']=items[0]
        for left,right in zip(items,items[1:]):entries[left]['right']=right
    return _build_nested_cfb(entries)


def main():
    from loguru import logger
    logger.remove()
    rows=[]
    for name in ['doc_stories.json','doc_extensions.json','doc_compat.json.gz']:
        for case in fixture(name)['cases']:
            data=binary_input(case)
            try:
                parsed=RustDoc()(data)
            except RapidDocError as error:
                assert 'error' in case and error.code=='corrupt_document',(case['name'],str(error))
                rows.append(dict(case=case['name'],expected_error=error.code));continue
            assert 'error' not in case,case['name']
            assert parsed.middle_json['pdf_info']==case['middle']['pdf_info'],case['name']
            assert parsed.markdown==case['exports']['mm_markdown'],case['name']
            for mode,expected in case['exports'].items():
                metadata,assets=parse_office_bytes(data,make_mode=mode)
                assert json.loads(metadata)['markdown']==expected,(case['name'],mode)
                assert not assets,case['name']
            rows.append(dict(case=case['name'],public_equal=True,exports=4))
    real=[]
    for case in fixture('doc_external.json.gz')['cases']:
        data=bytes.fromhex(case['source_hex'])
        python=PythonDoc()(data);rust=RustDoc()(data)
        assert snapshot(python)==snapshot(rust),case['name']
        real.append(dict(case=case['name'],middle_markdown_content_assets_equal=True))
    data=(ROOT/'demo/ms_office_docs/docx_01.doc').read_bytes()
    assert snapshot(PythonDoc()(data))==snapshot(RustDoc()(data)),'project DOC'
    real.append(dict(case='project docx_01.doc',middle_markdown_content_assets_equal=True))
    report=dict(binary_cases=rows,real_documents=real,all_passed=True)
    destination=ROOT/'build/doc-completion-installed-20261101.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'Installed acceptance: {len(rows)} binary cases, {len(real)} independent real DOCs; all passed')

if __name__=='__main__':main()
