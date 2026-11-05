"""Export the Python DOC header/footer oracle and Unicode isdigit ranges."""

import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter


def flatten(spans, url=None):
    result = []
    for span in spans:
        if span['type'] == 'hyperlink':
            result.extend(flatten(span['content'], span['url']))
        elif span['type'] == 'text':
            result.extend(
                {'text': char, 'styles': span.get('styles', []), 'url': url}
                for char in span['content']
            )
        else:
            raise ValueError(f"Unsupported auxiliary reference span: {span['type']}")
    return result


def main():
    converter = DocConverter()
    converter.convert(io.BytesIO((ROOT / 'demo/ms_office_docs/docx_01.doc').read_bytes()))
    pages = [
        [
            {'kind': block['type'].value, 'characters': flatten(block['content'])}
            for block in page if block['type'] in ('header', 'footer')
        ]
        for page in converter.pages
    ]
    destination = ROOT / 'tests/fixtures/office_rust_reference_068/doc_auxiliary.json'
    destination.write_text(json.dumps(pages), encoding='utf-8')

    ranges = []
    for code in range(sys.maxunicode + 1):
        if chr(code).isdigit():
            if ranges and ranges[-1][1] + 1 == code:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    destination = ROOT / 'rust/rapiddoc-office/src/doc/digit_ranges.json'
    destination.write_text(json.dumps(ranges), encoding='utf-8')
    print(f'Exported {len(pages)} sections and {len(ranges)} digit ranges')


if __name__ == '__main__':
    main()
