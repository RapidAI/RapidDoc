"""Freeze original Python table transactions without historical build fixtures."""
import copy
import gzip
import hashlib
import json

from common import ROOT
from export_native_code_containers import base
from export_native_table_page_commit import reference
from export_native_vector_formulas import base as vector_base, reference as vector_reference


def main():
    target = ROOT / 'tests/fixtures/native_table_owned_reference.json.gz'
    if target.exists():
        raise FileExistsError(target)
    raw = json.loads(gzip.decompress((ROOT / 'tests/fixtures/upstream_068/shared_supplemental_tables.json.gz').read_bytes()))
    cases = []

    def add(label, request, code=False):
        cases.append(dict(label=label, request=request, code=code, expected=reference(request, include_code=code)))

    for index, scene in enumerate(raw['cases']):
        modes = ['original', 'prior-claim', 'protected', 'all-protected', 'separators', 'excluded'] if index == 0 else ['original', 'prior-claim'] if index == 1 else ['original']
        for mode in modes:
            q = dict(source=copy.deepcopy(scene['request']['source']))
            s = q['source']
            if mode == 'prior-claim' and s['lines']:
                q['prior_claimed_line_indices'] = [s['lines'][0]['source_index']]
            if mode in ['protected', 'all-protected']:
                for line in s['lines'][:1] if mode == 'protected' else s['lines']:
                    line['semantic_type'] = 'header'
            if mode == 'separators':
                q['table_header_separator_bboxes'] = [r['bbox'] for r in s['drawing_lines']]
            if mode == 'excluded':
                q['excluded_bboxes'] = [[0., 0., *s['page_size']]]
            add(f'{index}/{mode}', q)
    for kind in ['filled', 'ruled', 'algorithm']:
        add('code/' + kind, dict(source=base(kind)['source']), code=True)
    vectors = []
    for mode in ['core', 'no-paths', 'claimed-body']:
        q = vector_base()
        if mode == 'no-paths':
            q['source']['path_infos'] = []
        if mode == 'claimed-body':
            q['claimed_line_indices'] = [0]
        vectors.append(dict(label=mode, request=q, expected=vector_reference(q)))
    paths = ['pipeline.py', 'graphics.py', 'table_detection.py', 'table_materialization.py', 'table_rules.py', 'code_blocks.py', 'formulas.py']
    sources = {}
    for name in paths:
        p = ROOT / 'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf' / name
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with target.open('wb') as out:
        out.write(gzip.compress(json.dumps(dict(sources=sources, cases=cases, vectors=vectors), ensure_ascii=False).encode(), mtime=0))
    print(len(cases), 'Python table transaction references')


if __name__ == '__main__':
    main()
