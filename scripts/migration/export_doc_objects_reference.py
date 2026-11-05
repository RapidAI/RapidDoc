"""Freeze Python DOC object order and embedded Workbook/Graph HTML references."""
import io
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.embedded_chart import extract_embedded_chart_html


def visual_tree(block):
    kind = block['type'].value if hasattr(block['type'], 'value') else block['type']
    if kind in ('image', 'chart', 'equation'):
        return [{'type': kind, 'content': block.get('content') if kind != 'image' else None}]
    if kind == 'list':
        return [v for child in block['content'] for v in visual_tree(child)]
    return []


def record(kind, payload=b''):
    return struct.pack('<HH', kind, len(payload)) + payload


def graph(include=None, malformed=False):
    data = record(0x809, struct.pack('<HH', 0x680, 5))
    data += record(0x41e, struct.pack('<HHB', 164, 5, 0) + b'0.00%')
    data += record(0xa) + record(0x809, struct.pack('<HH', 0x680, 0x8000)) + record(0x1052)
    for row in range(3):
        for col in range(3):
            prefix = struct.pack('<HHBH', row, col, 0, 164 if col == 1 else 0)
            if col == 0:
                text = f'Row {row}'.encode()
                data += record(0x204, prefix + bytes([len(text), 0]) + text)
            elif col == 1:
                data += record(3, prefix + struct.pack('<d', .125 * (row + 1)))
            else:
                data += record(1, prefix)
    if include is not None:
        data += record(0x1053, include)
    if malformed:
        data += record(0x204, struct.pack('<HHBH', 0, 2, 0, 0) + b'\xff\x01broken')
    return data + record(0xa)


def main():
    destination = ROOT / 'tests/fixtures/office_rust_reference_068'
    source = (ROOT / 'demo/ms_office_docs/docx_01.doc').read_bytes()
    converter = DocConverter()
    converter.convert(io.BytesIO(source))
    visuals = [[v for block in page for v in visual_tree(block)] for page in converter.pages]
    (destination / 'doc_visuals.json').write_text(json.dumps(visuals, ensure_ascii=False), encoding='utf-8')
    cases = []
    with BoundedOleReader(source) as ole:
        for name in ole.stream_names(prefix='ObjectPool/'):
            if name.lower().endswith(('/workbook', '/book')):
                data = ole.read_stream(name)
                cases.append((f'doc-object-{len(cases)}.bin', data))
    for name, include, malformed in [
        ('graph-all', None, False), ('graph-selected', struct.pack('<HH', 1, 3), False),
        ('graph-invalid-range', b'\xff', False), ('graph-empty-range', struct.pack('<HH', 4, 5), False),
        ('graph-truncated-label', None, True),
    ]:
        cases.append((name + '.bin', graph(include, malformed)))
    references = []
    for name, data in cases:
        (destination / name).write_bytes(data)
        references.append({'file': name, 'html': extract_embedded_chart_html(data)})
    (destination / 'doc_embedded_charts.json').write_text(json.dumps(references, ensure_ascii=False), encoding='utf-8')
    print('Visual blocks:', [len(p) for p in visuals], 'chart streams:', len(cases))


if __name__ == '__main__':
    main()
