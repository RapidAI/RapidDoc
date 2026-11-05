"""Compare installed Rust DOC navigation with the local Python public API."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rapid_doc import RapidDoc as PythonDoc
from rapid_doc_rust import RapidDoc as RustDoc


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != 'index'}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def navigation(middle):
    return [[clean(b) for b in page['para_blocks'] if b['type'] in ('title', 'list', 'index')]
            for page in middle['pdf_info']]


def first_difference(a, b, path=''):
    if type(a) is not type(b):
        return [path, type(a).__name__, type(b).__name__]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [path, list(a), list(b)]
        for key in a:
            diff = first_difference(a[key], b[key], f'{path}/{key}')
            if diff:
                return diff
    elif isinstance(a, list):
        if len(a) != len(b):
            return [path, len(a), len(b)]
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, f'{path}/{i}')
            if diff:
                return diff
    elif a != b:
        return [path, str(a)[:120], str(b)[:120]]
    return None


def main():
    path = ROOT / 'demo/ms_office_docs/docx_01.doc'
    python = PythonDoc()(path)
    rust = RustDoc()(path)
    actual, expected = navigation(rust.middle_json), navigation(python.middle_json)
    report = {'equal': actual == expected, 'first_difference': first_difference(actual, expected),
              'blocks_per_page': [len(p) for p in actual], 'pages': len(rust.middle_json['pdf_info']),
              'assets': len(rust.images), 'scope': 'titles, index trees, lists; excludes global numeric block indices'}
    (ROOT / 'build/doc-navigation-public-comparison-20261101.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'build/doc-navigation-installed-middle.json').write_text(
        json.dumps(rust.middle_json, ensure_ascii=False), encoding='utf-8')
    (ROOT / 'build/doc-navigation-installed.md').write_text(rust.markdown, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    assert report['equal'], report


if __name__ == '__main__':
    main()
