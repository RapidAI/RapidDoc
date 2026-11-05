"""独立进程比较已验证的换行内核；不作整文档速度或内存结论。"""
import argparse,hashlib,json,statistics,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def worker(args):
    # 复用既有跨平台进程峰值统计；测量时间不包括这些导入。
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from parse_bench import peak_rss
    if args.worker=='python':
        from rapid_doc.utils.text_boundary import merge_text_line_contents as merge
        module=sys.modules[merge.__module__].__file__
    else:
        import rapid_doc_rust
        assert Path(sys.prefix).resolve()in Path(rapid_doc_rust.__file__).resolve().parents
        merge=rapid_doc_rust.merge_text_lines;module=rapid_doc_rust.__file__
    cases=json.loads((ROOT/'build/migration/paragraphs.json').read_text())['join_cases']
    expected=[c['expected']for c in cases]
    def fingerprint(values):return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
    expected_hash=fingerprint(expected)
    del expected
    def batch():return [merge(c['lines'])for c in cases]
    for _ in range(args.warmup):
        values=batch();assert fingerprint(values)==expected_hash;del values
    ready_peak=peak_rss();runs=[]
    for _ in range(args.repeat):
        start=time.perf_counter();values=batch();runs.append(time.perf_counter()-start)
        assert fingerprint(values)==expected_hash;del values
    result=dict(engine=args.worker,python=sys.version,executable=sys.executable,module=module,
        cases=len(cases),output_sha256=expected_hash,runs_s=runs,median_s=statistics.median(runs),
        ready_process_peak_rss_bytes=ready_peak,process_lifetime_peak_rss_bytes=peak_rss())
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,default=ROOT/'.venv/Scripts/python.exe')
    parser.add_argument('--rust-python',type=Path,default=ROOT/'build/rust-wheel-env/Scripts/python.exe')
    parser.add_argument('--worker',choices=('python','rust'))
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'build/bench/text-join-comparison.json')
    args=parser.parse_args()
    if args.warmup<0 or args.repeat<1:parser.error('warmup >= 0, repeat >= 1 required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    results=[]
    for name,python in [('python',args.python),('rust',args.rust_python)]:
        target=args.output.with_name(f'text-join-{name}.json')
        command=[str(python.resolve()),'-I',str(Path(__file__).resolve()),'--worker',name,
            '--warmup',str(args.warmup),'--repeat',str(args.repeat),'--output',str(target.resolve())]
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if run.returncode:raise RuntimeError(run.stdout+run.stderr)
        results.append(json.loads(target.read_text()))
    old,new=results
    assert old['output_sha256']==new['output_sha256']and old['cases']==new['cases']
    report=dict(scope='merge_text_lines batch only, including binding/marshalling; timed imports/input loading/hash excluded; sequential isolated workers. RSS includes imports, fixture and Python host; not algorithm allocations or full parser memory.',
        full_document_quality_evaluated=False,full_document_performance_evaluated=False,
        fixture_sha256=hashlib.sha256((ROOT/'build/migration/paragraphs.json').read_bytes()).hexdigest(),
        warmup=args.warmup,repeat=args.repeat,results=results,
        sampled_batch_speed_ratio=old['median_s']/new['median_s'])
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'{old["cases"]} equal outputs; Python {old["median_s"]:.4f}s, Rust wheel {new["median_s"]:.4f}s; kernel batch only')
    print(f'Process peaks (imports/fixture included): Python {old["process_lifetime_peak_rss_bytes"]}, Rust {new["process_lifetime_peak_rss_bytes"]} bytes')


if __name__=='__main__':main()
