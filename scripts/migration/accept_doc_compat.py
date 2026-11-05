"""Verify installed DOC compatibility, including stable malformed-document errors."""
import gzip
import json
from pathlib import Path
from rapid_doc_rust import RapidDoc,RapidDocError

ROOT=Path(__file__).resolve().parents[2]


def main():
    fixture=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/doc_compat.json.gz').read_bytes()))
    rows=[]
    for case in fixture['cases']:
        path=ROOT/'build/doc-compat-inputs'/f"{case['name']}.doc"
        assert path.is_file(),path
        try:
            result=RapidDoc()(path)
        except RapidDocError as error:
            assert 'error' in case and error.code=='corrupt_document',(case['name'],str(error))
            rows.append({'case':case['name'],'expected_error':error.code})
            continue
        assert 'error' not in case,case['name']
        assert result.middle_json['pdf_info']==case['middle']['pdf_info'],case['name']
        assert result.markdown==case['exports']['mm_markdown'],case['name']
        assert not result.images,case['name']
        rows.append({'case':case['name'],'pages':len(result.middle_json['pdf_info']),'middle_equal':True,'markdown_equal':True})
    (ROOT/'build/doc-compat-installed-20261101.json').write_text(json.dumps({'cases':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Installed DOC compatibility: {len(rows)} passed')


if __name__=='__main__':main()
