"""Freeze DOC navigation, bookmark and table-list behavior from local Python."""
import io
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.models import DocParagraph, DocTextRun, DocListInfo, DocTableCell, DocTable, DocTableRow
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.bookmarks import parse_bookmarks
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.records import DocBudget


def characters(spans, url=None):
    out = []
    for span in spans:
        if span['type'] == 'hyperlink':
            out.extend(characters(span['content'], span['url']))
        else:
            out.extend({'text': c, 'kind': span['type'], 'styles': span.get('styles', []), 'url': url}
                       for c in span['content'])
    return out


def navigation(block):
    kind = str(block['type'])
    fields = ['anchor', 'list_label', 'ilevel', 'attribute', 'start', 'level', 'is_numbered_style']
    result = {'kind': kind, **{k: block.get(k) for k in fields}}
    if kind in ('list', 'index'):
        result['children'] = [navigation(b) for b in block['content']]
    else:
        result['characters'] = characters(block['content'])
    return result


def main():
    directory = ROOT / 'tests/fixtures/office_rust_reference_068'
    def save(name, value):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    c = DocConverter()
    c.convert(io.BytesIO((ROOT / 'demo/ms_office_docs/docx_01.doc').read_bytes()))
    save('doc_navigation_real.json', [[navigation(b) for b in p if b['type'] in
         ('doc_title', 'paragraph_title', 'list', 'index')] for p in c.pages])
    save('doc_tables_real.json', [re.sub(r'src="[^"]*"', 'src="IMAGE"', b['content'])
         for p in c.pages for b in p if b['type'] == 'table'])

    trims = []
    for runs in [[('Title\t12', None)], [('Title', '#_Toc1'), ('\t', None), ('12', None)],
                 [('Keep 12', None)], [('Title', '#_Toc1'), ('\t １２', '#_Toc1')],
                 [('12', None)], [('Title\t', None), ('12', '#_Toc2')],
                 [('Title\t12 suffix', None)], [('Title\t\n12 ', None)]]:
        p = DocParagraph(0, 1, runs=[DocTextRun(text, hyperlink=url) for text, url in runs])
        trims.append({'runs': runs, 'characters': characters(c._rich_text(c._toc_runs(p)))})
    save('doc_toc_runs.json', trims)

    bookmarks = []
    for extended in (False, True):
        for extra in (0, 2):
            names = ['ordinary', '_Toc1', '_GoBack', 'bad-end', 'reverse', '']
            data = struct.pack('<HHH', 0xffff, len(names), extra) if extended else struct.pack('<HH', len(names), extra)
            for name in names:
                payload = name.encode('utf-16le' if extended else 'cp1252')
                data += (struct.pack('<H', len(name)) if extended else bytes([len(name)])) + payload + b'X' * extra
            starts = struct.pack('<7I', 3, 3, 4, 5, 99, 8, 100) + b''.join(struct.pack('<HH', n, 0) for n in [0, 1, 2, 99, 0, 3])
            ends = struct.pack('<5I', 10, 20, 25, 30, 100)
            for trim in (0, 1):
                name_data = data[:-trim] if trim else data
                table = name_data + starts + ends
                expected = parse_bookmarks(table, names_offset=0, names_size=len(name_data), starts_offset=len(name_data),
                    starts_size=len(starts), ends_offset=len(name_data)+len(starts), ends_size=len(ends), budget=DocBudget())
                bookmarks.append({'data': list(table), 'sizes': [len(name_data), len(starts), len(ends)], 'expected': expected})
    save('doc_bookmarks.json', bookmarks)

    cases = []
    sequences = [[(True, 0), (True, 1), (True, 1), (True, 0), None, (False, 2), (True, 2)],
                 [(False, 0), (True, 0), (False, 0)], [(True, 8), None, (True, 0)], [None, (True, 1), None]]
    for specs in sequences:
        paragraphs = []
        inputs = []
        for i, spec in enumerate(specs):
            info = DocListInfo(1, spec[1], spec[0], i+1, f'({i+1})') if spec else None
            paragraphs.append(DocParagraph(i, i+1, runs=[DocTextRun(f'item {i} <&>')], list_info=info))
            inputs.append({'text': f'item {i} <&>', 'list': None if info is None else {'ordered': info.ordered, 'level': info.level, 'label': info.label}})
        table = DocTable(0, len(specs), [DocTableRow([DocTableCell(paragraphs)])])
        cases.append({'paragraphs': inputs, 'html': c._table_html(table)})
    save('doc_cell_lists.json', cases)
    merged = [DocParagraph(i, i+1, runs=[DocTextRun(text)], list_info=DocListInfo(1, 0, True, i+1, f'{i+1}.'))
              for i, text in enumerate(['First', 'Second'])]
    save('doc_merged_cell_list.json', c._table_html(DocTable(0, 2,
         [DocTableRow([DocTableCell(merged, col_span=2)])])))
    from rapid_doc import RapidDoc
    from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
    from copy import deepcopy
    middle = RapidDoc()(ROOT / 'demo/ms_office_docs/docx_01.doc').middle_json
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != 'index'}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    pages = deepcopy(middle['pdf_info'])
    for page in pages:
        page['para_blocks'] = [b for b in page['para_blocks'] if b['type'] in ('title', 'list', 'index')]
        page['discarded_blocks'] = []
    save('doc_navigation_public.json', [clean(p['para_blocks']) for p in pages])
    save('doc_navigation_exports.json', {mode: union_make(pages, mode, 'images') for mode in
         ['mm_markdown', 'nlp_markdown', 'content_list', 'content_list_v2']})
    print('Navigation: 3 sections; bookmarks:', len(bookmarks), 'TOC runs:', len(trims), 'table lists:', len(cases))


if __name__ == '__main__':
    main()
