"""冻结实际旧sparse_hybrid全候选与诊断；已有参考不覆写。"""
import copy
import json
import random
from dataclasses import asdict, replace
from common import ROOT, digest, write_json


def main():
    path = ROOT / 'build/migration/native-table-sparse-hybrid.json'
    if path.exists():
        raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import sparse_hybrid as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput, NativeTableRule, NativeTableRectangle, NativeTableGlyph, NativeTableText, NativeTableTextRow, NativeTableToken
    cases = []

    def objects(r):
        v = r['table_input']
        t = NativeTableInput(tuple(v['table_bbox']), tuple(v['page_size']), v['angle'], (), tuple(NativeTableRule(**q) for q in v['drawing_lines']), tuple(NativeTableRectangle(**q) for q in v['rectangles']))
        q = r['text']
        text = NativeTableText(tuple(NativeTableGlyph(**g) for g in q['glyphs']), tuple(NativeTableTextRow(**(row | dict(tokens=tuple(NativeTableToken(**tok) for tok in row['tokens'])))) for row in q['rows']), q['median_glyph_width'], q['median_glyph_height'])
        return t, text

    def add(t, text, label):
        diagnostics = []
        result = old.build_sparse_hybrid_candidates(t, text, diagnostics)
        cases.append(dict(label=label, request=dict(table_input=asdict(t), text=asdict(text)), expected=dict(candidates=[asdict(c) for c in result], diagnostics=copy.deepcopy(diagnostics), html=[serialize_candidate_html(c) for c in result])))

    prior = json.loads((ROOT / 'build/migration/native-table-rule-band.json').read_text(encoding='utf-8'))
    for c in prior['cases']:
        add(*objects(c['request']), 'prior-rule-band')
    real = copy.deepcopy(prior['real'])
    rng = random.Random(361988)

    def make(cols=4, header=0, body=4, mode='horizontal', pattern=None):
        width = cols * 60.
        rows = [list(range(cols)) if i >= header else ([0] if i == 0 else [1, 2]) for i in range(header + body)]
        if pattern is not None:
            rows = pattern
        height = len(rows) * 24. + 6.
        gs, rs = [], []
        for i, cells in enumerate(rows):
            tokens, ids = [], []
            for col in cells:
                k = len(gs); bbox = (col * 60. + 6., i * 24. + 6., col * 60. + 22., i * 24. + 14.)
                gs.append(NativeTableGlyph(k, k, chr(65 + k % 26), bbox, i)); ids.append(k)
                tokens.append(NativeTableToken(gs[-1].text, bbox, (k,), (k,)))
            rs.append(NativeTableTextRow(i, (0., i * 24. + 6., width, i * 24. + 14.), tuple(tokens), tuple(ids)))
        horizontal = tuple(NativeTableRule((0., y, width, y + .1), .1, 'horizontal') for y in (0., height - .1)) if mode != 'none' else ()
        vertical = tuple(NativeTableRule((x * 60. - .1, 0., x * 60. + .1, height), .2, 'vertical') for x in range(1, cols)) if mode in ('vertical', 'both') else ()
        rects = tuple(NativeTableRectangle((c * 60., .1, (c + 1) * 60., 18.), 5, mode == 'fill', True) for c in range(cols)) if mode in ('rect', 'fill') else ()
        return NativeTableInput((0., 0., width, height), (1000., 1000.), 0, (), horizontal + vertical, rects), NativeTableText(tuple(gs), tuple(rs), 8., 8.)

    for cols in (2, 3, 4, 5, 7):
        for head in (0, 1, 2, 3, 4):
            for mode in ('none', 'horizontal', 'vertical', 'both', 'rect', 'fill'):
                add(*make(cols, head, mode=mode), f'grid-{cols}-{head}-{mode}')
    for n in (3, 4, 5, 6, 9, 10):
        for sparse in (0, 1, 2, 3):
            rows = [list(range(4)) if r >= sparse else [0, 1, 2] for r in range(n)]
            add(*make(pattern=rows), f'dense-run-{n}-{sparse}')
    for fraction in (.749999, .75, .750001, .98, 1.):
        t, text = make(mode='vertical')
        lines = tuple(replace(r, bbox=(r.bbox[0], 0., r.bbox[2], t.table_bbox[3] * fraction)) if r.orientation == 'vertical' else r for r in t.drawing_lines)
        add(replace(t, drawing_lines=lines), text, f'vertical-coverage-{fraction}')
    # 两层表头：第0列rowspan，后三列由上层一个居中token覆盖。
    t, text = make(header=2, body=4, mode='vertical')
    q = json.loads(json.dumps(asdict(text)))
    q['glyphs'][1]['bbox'] = [140., 6., 158., 14.]
    q['rows'][0]['tokens'].append(dict(text=q['glyphs'][1]['text'], bbox=q['glyphs'][1]['bbox'], glyph_ids=[1], source_char_indices=[1]))
    q['rows'][0]['glyph_ids'].append(1)
    # 重建ID后，两行表头与正文物理glyph引用一致。
    for row in q['rows'][1:]:
        row['tokens'] = [tok for tok in row['tokens'] if tok['glyph_ids'] != [1]]
        row['glyph_ids'] = [i for i in row['glyph_ids'] if i != 1]
    q['glyphs'][1]['visual_row'] = 0
    for coverage in (0., .2, .200001, .5, .799999, .8, 1.):
        lines = t.drawing_lines + tuple(NativeTableRule((60. * c, 21.95, 60. * c + 60. * coverage, 22.05), .1, 'horizontal') for c in range(1, 4))
        ti, tx = objects(dict(table_input=asdict(replace(t, drawing_lines=lines)), text=q))
        add(ti, tx, f'header-separator-{coverage}')
    # 物理轨道才允许粗token穿列，且不能穿glyph墨迹。
    for mode in ('horizontal', 'vertical', 'rect'):
        for ink in (False, True):
            t, tx = make(mode=mode); q = json.loads(json.dumps(asdict(tx)))
            r = q['rows'][1]; a, b = r['tokens'][:2]
            r['tokens'] = [dict(text=a['text'] + b['text'], bbox=[a['bbox'][0], a['bbox'][1], b['bbox'][2], b['bbox'][3]], glyph_ids=a['glyph_ids'] + b['glyph_ids'], source_char_indices=a['source_char_indices'] + b['source_char_indices'])] + r['tokens'][2:]
            if ink:q['glyphs'][r['glyph_ids'][0]]['bbox'][2] = 65.
            add(*objects(dict(table_input=asdict(t), text=q)), f'token-split-{mode}-{ink}')
    for _ in range(160):
        cols = rng.randrange(2, 8); head = rng.randrange(4); mode = rng.choice(['horizontal', 'vertical', 'rect'])
        t, tx = make(cols, head, body=rng.randrange(2, 9), mode=mode); q = json.loads(json.dumps(asdict(tx)))
        for g in q['glyphs']:
            dx = rng.uniform(-15, 15); b = g['bbox']; g['bbox'] = (b[0] + dx, b[1], b[2] + dx, b[3])
        add(*objects(dict(table_input=asdict(t), text=q)), 'jitter')
    for bbox in [(240., 102., 0., 0.), (0., 0., 0., 100.)]:
        t, tx = make(); add(replace(t, table_bbox=bbox), tx, 'reversed-or-empty')
    src = 'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(path, dict(sources={src + p: digest((ROOT / (src + p)).read_bytes()) for p in ['sparse_hybrid.py', 'sparse_common.py', 'candidate.py', 'geometry.py']}, cases=cases, real=real))
    print(f'Frozen {len(cases)} sparse-hybrid cases, {sum(bool(c["expected"]["candidates"]) for c in cases)} nonempty')


if __name__ == '__main__':
    main()
