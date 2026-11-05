"""Check installed Rust against Python-frozen story results and four exports."""
import json
from pathlib import Path
from rapid_doc_rust import RapidDoc

ROOT=Path(__file__).resolve().parents[2]


def main():
    rows=[]
    cases=[]
    for filename in ['doc_stories.json','doc_extensions.json']:
        cases.extend(json.loads((ROOT/'tests/fixtures/office_rust_reference_068'/filename).read_text(encoding='utf-8'))['cases'])
    for case in cases:
        path=ROOT/'build/doc-story-inputs'/f"{case['name']}.doc"
        parsed=RapidDoc()(path)
        assert parsed.middle_json['pdf_info']==case['middle']['pdf_info'],case['name']
        assert parsed.markdown==case['exports']['mm_markdown'],case['name']
        assert not parsed.images,case['name']
        rows.append({'case':case['name'],'pages':len(parsed.middle_json['pdf_info']),'middle_equal':True,'markdown_equal':True})
    report={'cases':rows,'scope':'installed RapidDoc, full pdf_info including indices and formula/note/link spans, Markdown'}
    (ROOT/'build/doc-stories-installed-20261101.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Installed DOC story acceptance: {len(rows)} passed')


if __name__=='__main__':main()
