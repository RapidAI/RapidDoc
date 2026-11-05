"""Sequential growth/repeated-call memory checks, with separate parity workers.

Generated cases are workloads, not additional real-document coverage. Office
uses defaults. The PDF window workload explicitly uses native-rule text parsing
without formula/table models; it does not measure the default model pipeline.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accept_doc_performance import dump, installed_binary


def generate(output):
    from accept_doc_performance import stress
    from docx import Document
    from openpyxl import Workbook
    from pptx import Presentation
    from pptx.util import Inches
    from reportlab.pdfgen.canvas import Canvas
    sys.path.insert(0, str(ROOT / 'tests/native_office'))
    import _legacy_xls_test_utils as xls
    sources = []

    def save(path, format, scale, **extra):
        sources.append(dict(format=format, scale=scale, path=str(path), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra))

    for n in [2000, 8000, 32000]:
        p = output / f'text-{n}.doc'
        p.write_bytes(stress(n)); save(p, 'doc', n, workload='Unicode paragraphs')
    for n in [200, 800, 3200]:
        p = output / f'body-{n}.docx'; document = Document()
        for i in range(n):
            paragraph = document.add_paragraph()
            paragraph.add_run(f'Paragraph {i}: 中文 English ').bold = i % 7 == 0
            paragraph.add_run('body with styles.').underline = i % 11 == 0
            if i % 20 == 0:
                table = document.add_table(rows=2, cols=3)
                for j, cell in enumerate(table._cells): cell.text = f'cell {i}/{j}'
        document.save(p); save(p, 'docx', n, workload='Paragraphs, styles and tables')
    for n in [500, 2000, 8000]:
        p = output / f'cells-{n}.xlsx'; book = Workbook(); sheet = book.active
        for i in range(n): sheet.append([f'项目 {i}', *[i * (j + 1) for j in range(5)]])
        book.save(p); book.close(); save(p, 'xlsx', n, workload='Six-column worksheet rows')
        records = []
        for i in range(n):
            records.append(xls.label_cell(i, 0, f'项目 {i}'))
            records.extend(xls.number_cell(i, j + 1, i * (j + 1)) for j in range(5))
        p = output / f'cells-{n}.xls'
        p.write_bytes(xls.build_xls([xls.SheetFixture('Data', b''.join(records))]))
        save(p, 'xls', n, workload='Six-column BIFF8 worksheet rows')
    for n in [25, 100, 400]:
        p = output / f'slides-{n}.pptx'; presentation = Presentation()
        for i in range(n):
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            slide.shapes.title.text = f'Slide {i}'
            frame = slide.placeholders[1].text_frame
            frame.text = f'正文 {i}: English and 中文'
            for j in range(4):
                paragraph = frame.add_paragraph(); paragraph.text = f'Item {i}/{j}'; paragraph.level = j % 2
            if i % 10 == 0:
                table = slide.shapes.add_table(2, 2, Inches(1), Inches(5), Inches(6), Inches(1)).table
                for r in range(2):
                    for c in range(2): table.cell(r,c).text = f'{i}/{r}/{c}'
        presentation.save(p); save(p, 'pptx', n, workload='Slide text/lists and tables')
    for n in [16, 65, 129]:
        p = output / f'windows-{n}.pdf'; canvas = Canvas(str(p), invariant=1)
        for i in range(n):
            for line in range(8): canvas.drawString(72, 750 - 24 * line, f'Page {i + 1}, line {line + 1}: native PDF text window stress.')
            canvas.showPage()
        canvas.save(); save(p, 'pdf', n, workload='Native text, default 64-page windows',
            options=dict(layout_config=dict(model_type='docvortex_rule'), formula_enable=False, table_enable=False, parse_method='txt', pdf_pages_batch=64))
    p = output / 'source.ppt'; p.write_bytes((ROOT / 'demo/ms_office_docs/pptx_01.ppt').read_bytes())
    save(p, 'ppt', 6, workload='Real six-slide PPT, repeated calls only', real=True)
    return sources


def worker(args):
    from loguru import logger
    logger.remove()
    from accept_pdf_delivery import memory
    if args.engine == 'python':
        from rapid_doc import RapidDoc
    else:
        from rapid_doc_rust import RapidDoc
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    cases = [c for c in manifest['sources'] if c['format'] == args.format]
    engine = RapidDoc(**cases[0].get('options', {}))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; report = dict(engine=args.engine, format=args.format, phase=args.phase, rows=rows, baseline=memory(), success=False)
    try:
        from accept_docx_delivery import snapshot
        for case in cases:
            data = Path(case['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == case['sha256']
            gc.collect(); before = memory(); start = time.perf_counter()
            result = engine(data)
            row = dict(scale=case['scale'], bytes=case['bytes'], seconds=time.perf_counter()-start,
                       pages=len(result.middle_json['pdf_info']), assets=len(result.images), returned=memory())
            if args.phase == 'verify':
                canonical = snapshot(result)
                dump(args.output / f"{case['scale']}.json", canonical)
                row['canonical_sha256'] = hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()
                del canonical
            del result; gc.collect(); row.update(before=before, released=memory()); rows.append(row)
            dump(args.output / 'report.json', report)
        if args.phase == 'trend':
            # Same input and same engine. Do not retain outputs or decode assets.
            for i in range(6):
                gc.collect(); before = memory(); result = engine(data); returned = memory()
                pages = len(result.middle_json['pdf_info']); del result; gc.collect()
                rows.append(dict(repeat=i+1, scale=cases[-1]['scale'], before=before,
                                 returned=returned, released=memory(), pages=pages))
                dump(args.output / 'report.json', report)
        report['success'] = True
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    dump(args.output / 'report.json', report)
    if not report['success']: raise SystemExit(1)


def summarize(args, manifest):
    results = []
    for format in sorted({c['format'] for c in manifest['sources']}):
        rows = {e:{p:json.loads((args.output/f'{format}-{p}-{e}'/'report.json').read_text(encoding='utf8'))
                   for p in ['trend','verify']} for e in ['python','rust']}
        assert all(r['success'] for phases in rows.values() for r in phases.values())
        a, b = [rows[e]['verify']['rows'] for e in ['python','rust']]
        parity = [(x['scale'],x['canonical_sha256']) for x in a] == [(x['scale'],x['canonical_sha256']) for x in b]
        item = dict(format=format, parity=parity)
        for e in ['python','rust']:
            r = rows[e]['trend']; repeats = [x for x in r['rows'] if 'repeat' in x]
            tail = repeats[-3:]
            item[e] = dict(growing=[dict(scale=x['scale'],returned_rss=x['returned']['working_set'],released_rss=x['released']['working_set'],released_private=x['released']['private_commit']) for x in r['rows'] if 'repeat' not in x],
                process_peak_rss=max(x['returned']['os_peak_working_set'] for x in r['rows']),
                repeat_released_rss=[x['released']['working_set'] for x in repeats],
                repeat_released_private=[x['released']['private_commit'] for x in repeats],
                last_three_rss_span=max(x['released']['working_set'] for x in tail)-min(x['released']['working_set'] for x in tail),
                last_three_private_span=max(x['released']['private_commit'] for x in tail)-min(x['released']['private_commit'] for x in tail))
        results.append(item)
    dump(args.output/'summary.json',dict(cases=results,all_parity=all(r['parity'] for r in results),notes=[
        'Generated inputs are scale/repetition workloads, not extra real-corpus coverage.',
        'Trend workers retain no results and do no output verification or image decoding.',
        'Parity is verified in separate processes; lifetime RSS includes imports and growing earlier inputs.',
        'PDF explicitly uses native-rule text and disabled formula/table models; default model memory is not measured here.',
        'Six same-input repeats provide observations, not proof of absence of leaks or allocator retention.',
        'Trend run latencies are diagnostic; use separate isolated performance acceptance for speed claims.']))
    if not all(r['parity'] for r in results): raise SystemExit('Stress output parity failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--engine',choices=['python','rust'])
    parser.add_argument('--format')
    parser.add_argument('--phase',choices=['trend','verify'])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.worker:return worker(args)
    path=args.output/'manifest.json'
    if args.resume:
        manifest=json.loads(path.read_text(encoding='utf8'));assert manifest['installed_rust']==installed_binary()
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        inputs=args.output/'inputs';inputs.mkdir()
        manifest=dict(installed_rust=installed_binary(),sources=generate(inputs));dump(path,manifest)
    assert all(hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256'] for c in manifest['sources']), 'Stress input changed'
    for index,format in enumerate(sorted({c['format'] for c in manifest['sources']})):
        for phase in ['trend','verify']:
            for engine in (['python','rust'] if index%2==0 else ['rust','python']):
                destination=args.output/f'{format}-{phase}-{engine}'
                if args.resume and (destination/'report.json').exists():
                    assert json.loads((destination/'report.json').read_text(encoding='utf8'))['success'];continue
                print(f'{format}: {phase} {engine}',flush=True)
                with (args.output/(destination.name+'.log')).open('w',encoding='utf8') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--engine',engine,
                        '--format',format,'--phase',phase,'--manifest',str(path),'--output',str(destination)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
    assert manifest['installed_rust']==installed_binary()
    summarize(args,manifest)


if __name__=='__main__':main()
