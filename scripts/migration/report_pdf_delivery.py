"""Analyze saved PDF acceptance outputs; never invokes parsers or edits results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from accept_pdf_delivery import differences


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args(); root = args.input.resolve()
    summary = read(root/'summary.json'); results = []
    for case in summary['cases']:
        row = dict(id=case['case']['id'], source=case['case']['source'], pages=case['case']['pages'])
        engines = case['engines']
        row['complete'] = len(engines) == 2 and all(e['success'] for e in engines.values())
        if not row['complete']:
            row['errors'] = {name: e.get('error') for name,e in engines.items() if not e['success']}
            results.append(row); continue
        folder = root/row['id']
        py = read(folder/'python'/'warm.json'); ru = read(folder/'rust'/'warm.json')
        groups = Counter(); examples = []
        for d in differences(py['middle_json'], ru['middle_json']):
            path = d['path']
            if path.endswith(('.score', '.confidence', '.detection_score')): group='confidence'
            elif any(x in path for x in ['.bbox[', '.poly[', '.page_size[', '.points[']): group='geometry'
            else: group='other'
            groups[group] += 1
            if len(examples)<30: examples.append(d)
        row.update(full_page_counts=all(r['pages']==row['pages'] for e in engines.values() for r in e['runs']),
                   nonempty=all(r['nonempty'] for e in engines.values() for r in e['runs']),
                   exact_markdown=case['comparison']['markdown']['exact'],
                   exact_content_list=case['comparison']['content_list_json']['exact'],
                   exact_assets=case['comparison']['assets']['exact_pixels_and_names'],
                   exact_middle=case['comparison']['middle_json']['exact'],
                   middle_differences=dict(groups), middle_examples=examples,
                   repeat_exact={name:all(case[name+'_repeat'][field]['exact'] for field in
                                         ['markdown','content_list_json','middle_json']) and
                                         case[name+'_repeat']['assets']['exact_pixels_and_names']
                                 for name in ['python','rust']})
        row['metrics'] = {}
        for name in ['python','rust']:
            reports = [engines[name]]
            reverse = root/'reverse'/row['id']/name/'report.json'
            if reverse.exists():
                report = read(reverse)
                if report['success']:
                    reports.append(report)
                    base = read(folder/name/'warm.json'); extra = read(reverse.parent/'warm.json')
                    row.setdefault('reverse_output_exact',{})[name] = not any(differences(base,extra))
            cold=[r['runs'][0]['seconds'] for r in reports]
            warm=[r['runs'][1]['seconds'] for r in reports]
            row['metrics'][name] = dict(cold_seconds=statistics.median(cold),warm_seconds=statistics.median(warm),
                                       cold_samples=cold,warm_samples=warm,
                                       cold_peak_working_set=max(r['runs'][0]['memory']['at_return']['os_peak_working_set'] for r in reports),
                                       cold_peak_commit=max(r['runs'][0]['memory']['at_return']['os_peak_commit'] for r in reports),
                                       minimum_available_physical=min(run['memory']['minimum_available_physical'] for r in reports for run in r['runs']),
                                       maximum_memory_load_percent=max(run['memory']['maximum_system_memory_load_percent'] for r in reports for run in r['runs']))
        py,ru=(row['metrics'][name]for name in ['python','rust'])
        row['warm_rust_time_change_percent']=(ru['warm_seconds']/py['warm_seconds']-1)*100
        row['cold_rust_time_change_percent']=(ru['cold_seconds']/py['cold_seconds']-1)*100
        results.append(row)
    complete=[r for r in results if r['complete']]
    def total(name,key):return sum(r['metrics'][name][key]for r in complete)
    report=dict(planned_cases=len(results),completed_cases=len(complete),planned_pages=sum(r['pages']for r in results),
                completed_pages=sum(r['pages']for r in complete),
                visible_output_exact_cases=sum(r['exact_markdown']and r['exact_content_list']and r['exact_assets']for r in complete),
                strict_middle_exact_cases=sum(r['exact_middle']for r in complete),
                page_count_and_nonempty_cases=sum(r['full_page_counts']and r['nonempty']for r in complete),
                repeated_outputs_exact_cases=sum(all(r['repeat_exact'].values())for r in complete),
                python_warm_seconds=total('python','warm_seconds'),rust_warm_seconds=total('rust','warm_seconds'),
                python_cold_seconds=total('python','cold_seconds'),rust_cold_seconds=total('rust','cold_seconds'),
                cases=results)
    report['engine_success_counts']={name:sum(case['engines'].get(name,{}).get('success',False)
                                             for case in summary['cases'])for name in ['python','rust']}
    report['initial_cohort_totals']={}
    for label, selected in [
        ('jointly_successful', complete),
        ('visible_output_exact', [r for r in complete if r['exact_markdown']and r['exact_content_list']and r['exact_assets']]),
    ]:
        ids={r['id']for r in selected}
        cohort=[c for c in summary['cases']if c['case']['id']in ids]
        totals=dict(cases=len(selected),pages=sum(r['pages']for r in selected))
        for name in ['python','rust']:
            for index, phase in enumerate(['cold','warm']):
                totals[name+'_'+phase+'_seconds']=sum(c['engines'][name]['runs'][index]['seconds']for c in cohort)
        report['initial_cohort_totals'][label]=totals
    report['acceptance_passed']=len(complete)==len(results)and all(
        r['full_page_counts']and r['nonempty']and r['exact_markdown']and r['exact_content_list']and
        r['exact_assets']and r['exact_middle']and all(r['repeat_exact'].values())for r in complete)
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# PDF 验收实测','',f"已执行 {len(results)} 份、{report['planned_pages']} 页；Python成功 {report['engine_success_counts']['python']} 份，Rust成功 {report['engine_success_counts']['rust']} 份。双方成功配对 {len(complete)} 份、{report['completed_pages']} 页。",
           '', '验收结论：'+('通过。'if report['acceptance_passed']else'未通过，存在执行失败或输出差异。'),
           '', '耗时为正式API整篇解析及导出；首次包含模型加载。内存为首次调用返回时进程高水位，GiB单位。',
           '代表文件若有反序复测，耗时取两次中位数、内存取最大值；逐次原始数据见analysis.json。',
           '', '| 文件 | 页 | Python首次/复用秒 | Rust首次/复用秒 | Rust复用耗时变化 | 峰值工作集 Python/Rust GiB | 峰值提交 Python/Rust GiB | MD/内容/图片 | 严格中间JSON |',
           '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in results:
        if not row['complete']:
            lines.append(f"| {Path(row['source']).name} | {row['pages']} | — | — | — | — | — | 解析失败 | 解析失败 |");continue
        a,b=(row['metrics'][name]for name in ['python','rust'])
        visible='一致'if row['exact_markdown']and row['exact_content_list']and row['exact_assets']else'有差异'
        mid='一致'if row['exact_middle']else str(row['middle_differences'])
        lines.append(f"| {Path(row['source']).name} | {row['pages']} | {a['cold_seconds']:.2f}/{a['warm_seconds']:.2f} | {b['cold_seconds']:.2f}/{b['warm_seconds']:.2f} | {row['warm_rust_time_change_percent']:+.1f}% | {a['cold_peak_working_set']/2**30:.2f}/{b['cold_peak_working_set']/2**30:.2f} | {a['cold_peak_commit']/2**30:.2f}/{b['cold_peak_commit']/2**30:.2f} | {visible} | {mid} |")
    lines+=['','负百分比表示Rust耗时减少。可见结果一致不等于所有JSON数值一致，也不等于人工标注准确率。',
            '性能是本机当前环境实测，首次不等于清空磁盘缓存；单次或中位数不能证明所有文档稳定提速。',
            '失败案例不能当作通过；原始差异与错误保留在各报告中。',
            '', '## 首轮汇总（不混入反序中位数）', '',
            '| 范围 | 份/页 | Python首次/复用秒 | Rust首次/复用秒 |',
            '|---|---:|---:|---:|']
    for key,label in [('jointly_successful','双方执行成功，含输出差异'),('visible_output_exact','MD/内容/图片完全一致')]:
        t=report['initial_cohort_totals'][key]
        lines.append(f"| {label} | {t['cases']}/{t['pages']} | {t['python_cold_seconds']:.2f}/{t['python_warm_seconds']:.2f} | {t['rust_cold_seconds']:.2f}/{t['rust_warm_seconds']:.2f} |")
    lines+=['', 'analysis.json的顶层耗时是表中各文件中位数之和，混合复测与单次数据；不作为统一复测轮次的总耗时。',
            '', '## 反序复测逐次耗时', '', '| 文件/引擎 | 首次调用秒（初轮、反序） | 模型复用秒（初轮、反序） | 输出跨轮一致 |', '|---|---|---|---|']
    for row in complete:
        for name,exact in row.get('reverse_output_exact',{}).items():
            m=row['metrics'][name]
            lines.append(f"| {Path(row['source']).name}/{name} | {', '.join(f'{x:.2f}'for x in m['cold_samples'])} | {', '.join(f'{x:.2f}'for x in m['warm_samples'])} | {exact} |")
    lines+=['', '反序复测未改变算法和模型；同一实现的轮间波动不能记作代码优化收益。仅凭这些样本，不能承诺稳定提速或数倍提速。', '', '## 执行错误', '']
    for row in results:
        for name,error in row.get('errors',{}).items():
            lines.append(f"- {Path(row['source']).name}/{name}：{error}")
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items()if k!='cases'},ensure_ascii=False))


if __name__=='__main__':main()
