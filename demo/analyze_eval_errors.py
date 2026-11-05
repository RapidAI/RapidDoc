"""Rank existing OmniDocBench scores without rerunning parsing or evaluation."""
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from eval_preview import build_previews, classify

METRICS = {'text': ('文字编辑距离', False), 'formula': ('公式 CDM', True),
           'table': ('表格 TEDS', True), 'structure': ('表格结构 TEDS', True),
           'order': ('阅读顺序编辑距离', False)}


def load_result(folder):
    def read(suffix):
        return json.loads((folder/f'predictions_quick_match_{suffix}.json').read_text(encoding='utf8'))
    scores = defaultdict(dict)
    examples = defaultdict(list)
    for key, suffix in [('text', 'text_block'), ('order', 'reading_order')]:
        for name, value in read(suffix+'_per_page_edit').items():
            scores[Path(name).stem][key] = float(value)
    for suffix, keys in [('display_formula', [('formula', 'CDM')]),
                         ('table', [('table', 'TEDS'), ('structure', 'TEDS_structure_only')])]:
        grouped = defaultdict(lambda: defaultdict(list))
        for item in read(suffix+'_result'):
            page = Path(item.get('image_name') or item['img_id']).stem
            for key, metric in keys:
                value = item.get('metric', {}).get(metric)
                if value is not None:
                    grouped[page][key].append(float(value))
        for page, values in grouped.items():
            for key, samples in values.items():scores[page][key] = mean(samples)
    # Keep only the two worst matched samples per category/page for quick inspection.
    for suffix, metric, key, higher in [('text_block', 'Edit_dist', 'text', False),
                                       ('display_formula', 'CDM', 'formula', True),
                                       ('table', 'TEDS', 'table', True)]:
        grouped = defaultdict(list)
        for item in read(suffix+'_result'):
            value = item.get('metric', {}).get(metric)
            if value is None:continue
            error = 1.-float(value) if higher else float(value)
            if error <= 0:continue
            page = Path(item.get('image_name') or item['img_id']).stem
            grouped[page].append((error, item))
        for page, items in grouped.items():
            for error, item in sorted(items, key=lambda x:x[0], reverse=True)[:2]:
                def excerpt(value):
                    text = str(value or '')
                    return text[:1200] + ('…（截断；完整内容见评分 JSON）' if len(text)>1200 else '')
                examples[page].append(dict(category=key, value=1-error if higher else error,
                    gt=excerpt(item.get('gt')), pred=excerpt(item.get('pred'))))
    return dict(scores), dict(examples)


def error_value(key, value):
    return 1-value if METRICS[key][1] else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('D:/Download/OmniDocBench')
    parser.add_argument('--result', type=Path, default=root/'result_layout_v3-ocrv6_smail_onnx_tablev5_rust_orientation_on_run2')
    parser.add_argument('--baseline', type=Path, default=root/'result_layout_v3-ocrv6_smail_onnx_tablev5_rust')
    parser.add_argument('--no-baseline', action='store_true', help='Rank one run without comparing another run')
    parser.add_argument('--title', default='Rust 开启方向分类 run2 · 问题定位')
    parser.add_argument('--no-previews',action='store_true',help='Skip creating PDF and Markdown previews')
    parser.add_argument('--predictions', type=Path, default=root/'layout_v3-ocrv6_smail_onnx_tablev5_rust_orientation_on_run2')
    parser.add_argument('--pdfs', type=Path, default=root/'pdfs')
    parser.add_argument('--output', type=Path, default=root/'rust_error_analysis_orientation_on_run2')
    args = parser.parse_args()
    current, examples = load_result(args.result)
    baseline, _ = load_result(args.baseline) if args.baseline and not args.no_baseline else ({}, {})
    rows = []
    for name, values in current.items():
        errors = {key:error_value(key,value) for key,value in values.items()}
        old = baseline.get(name, {})
        deltas = {key:error_value(key,value)-error_value(key,old[key]) for key,value in values.items() if key in old}
        row = dict(file=name+'.pdf', metrics=values, baseline=old, deltas=deltas,
                   worst=max(errors.values()), worst_metric=max(errors,key=errors.get),
                   regression=max(deltas.values()) if deltas else None,
                   examples=examples.get(name, []))
        for key, path in [('pdf', args.pdfs/(name+'.pdf')), ('markdown', args.predictions/(name+'.md'))]:
            row[key] = path.resolve().as_uri() if path.is_file() else None
        rows.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    denominators={k:sum(k in r['metrics']for r in rows)for k in ('text','formula','table')}
    for row in rows:
        row['severity'],row['reasons']=classify(row['metrics'])
        row['impact']=sum(error_value(k,row['metrics'][k])*100/(3*denominators[k])for k in denominators if k in row['metrics'])
    failures=build_previews(rows,args.pdfs,args.predictions,args.output)if not args.no_previews else []
    (args.output/'preview_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf8')
    rows.sort(key=lambda row:row['worst'],reverse=True)
    def csv_file(name, items):
        with (args.output/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['文件','最大单项问题程度（非官方总分）','最差指标']+
                [v[0]+('（0–100，越高越好）' if v[1] else '（越低越好）')for v in METRICS.values()]+
                ['较基准单项最大退步（归一化差值，正数为退步）']+
                ['差值_'+v[0]+'（正数为退步）'for v in METRICS.values()]+['PDF','Markdown'])
            for row in items:
                writer.writerow([row['file'],row['worst'],METRICS[row['worst_metric']][0]]+
                    [row['metrics'].get(k)*100 if k in row['metrics'] and v[1] else row['metrics'].get(k)for k,v in METRICS.items()]+
                    [row['regression']]+[row['deltas'].get(k)for k in METRICS]+[row['pdf'],row['markdown']])
    csv_file('worst_pages.csv',rows)
    for key in METRICS:
        csv_file('worst_'+key+'.csv',sorted((r for r in rows if key in r['metrics']),key=lambda r:error_value(key,r['metrics'][key]),reverse=True))
    regressions=sorted((r for r in rows if r['regression'] is not None and r['regression']>1e-9),key=lambda r:r['regression'],reverse=True)
    csv_file('orientation_regressions.csv',regressions)
    (args.output/'pages.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    payload=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    report=HTML.replace('__DATA__',payload).replace('__METRICS__',json.dumps(METRICS,ensure_ascii=False))
    report=report.replace('__DATASET__',json.dumps(str(args.result.resolve())).replace('<','\\u003c'))
    report=report.replace('Rust 逐页问题定位',html.escape(args.title))
    if args.no_baseline:
        report=report.replace('较基准退步最多','无基准对比').replace('旋转差值仅比较两组均有的指标，正数代表退步；同一页可能有的指标改善、有的退步，差异不等于已证明由旋转导致。','本报告只展示本次评分，没有与其他运行进行比较。')
    (args.output/'index.html').write_text(report,encoding='utf8')
    summary=dict(pages=len(rows),regressed_pages=len(regressions),
                 low_score_pages=sum(r['severity']>=1 for r in rows),severe_pages=sum(r['severity']==2 for r in rows),
                 low_by_metric={k:sum(k in r['reasons']for r in rows)for k in ('text','formula','table','order')},
                 preview_failures=len(failures),
                 changed_pages=sum(any(abs(v)>1e-9 for v in r['deltas'].values())for r in rows),
                 improved_pages=sum(any(v < -1e-9 for v in r['deltas'].values())for r in rows),
                 worst=[dict(file=r['file'],metric=r['worst_metric'],value=r['metrics'][r['worst_metric']])for r in rows[:10]],
                 regressions=[dict(file=r['file'],deltas=r['deltas'])for r in regressions[:10]])
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Report:',(args.output/'index.html').resolve())


HTML=Path(__file__).with_name('eval_review.html').read_text(encoding='utf8')

if __name__=='__main__':main()
