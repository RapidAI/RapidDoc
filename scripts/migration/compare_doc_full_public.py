"""Compare the real DOC public result after aliasing equivalent image encodings."""
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rapid_doc import RapidDoc as PythonDoc
from rapid_doc_rust import RapidDoc as RustDoc
from compare_doc_object_images import canonical
from compare_doc_navigation_public import first_difference


def aliases(images):
    result={}
    for name,data in images.items():
        mime=mimetypes.guess_type(name)[0] or 'application/octet-stream'
        uri=f'data:{mime};base64,'+base64.b64encode(data).decode()
        value=canonical(uri)
        digest=hashlib.sha256(repr(value).encode()).hexdigest()
        result[name]='ASSET_'+digest
    return result


def replace(text,names):
    for name,alias in sorted(names.items(),key=lambda v:len(v[0]),reverse=True):
        text=text.replace(name,alias)
    return text


def main():
    path=ROOT/'demo/ms_office_docs/docx_01.doc'
    # Both engines consume one snapshot even when Word keeps the source open.
    data=path.read_bytes()
    python=PythonDoc()(data);rust=RustDoc()(data)
    expected_aliases,actual_aliases=aliases(python.images),aliases(rust.images)
    expected=json.loads(replace(json.dumps(python.middle_json['pdf_info'],ensure_ascii=False),expected_aliases))
    actual=json.loads(replace(json.dumps(rust.middle_json['pdf_info'],ensure_ascii=False),actual_aliases))
    report={'middle_equal':actual==expected,'markdown_equal':replace(rust.markdown,actual_aliases)==replace(python.markdown,expected_aliases),
            'first_difference':first_difference(actual,expected),'python_assets':len(python.images),'rust_assets':len(rust.images),
            'asset_values_equal':sorted(actual_aliases.values())==sorted(expected_aliases.values()),
            'scope':'full pdf_info including indices, Markdown and asset pixels/SVG; only equivalent asset filenames aliased'}
    (ROOT/'build/doc-full-public-comparison-20261101.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    assert report['middle_equal'] and report['markdown_equal'] and report['asset_values_equal'],report


if __name__=='__main__':main()
