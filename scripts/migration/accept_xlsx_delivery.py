"""Sequential, alternating isolated Python/Rust XLSX latency and memory checks.

Reuses the accepted DOCX worker's measurement and exact image/content snapshot
rules. Does not alter the files or cell/table algorithms for benchmarking.
"""
import argparse,json,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--format',choices=['xlsx','xls','pptx'],default='xlsx')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def dump(name,value):(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=(sorted((ROOT/'demo/xlsx').glob('*.xlsx'))+[ROOT/'demo/ms_office_docs/xlsx_01.xlsx'])if args.format=='xlsx'else[ROOT/'demo/ms_office_docs/xlsx_01.xls',ROOT/'tests/migration/inputs/synthetic.xls']
    if args.format=='pptx':sources=sorted((ROOT/'demo/pptx').glob('*.pptx'))+[ROOT/'tests/fixtures/docvortex_migration/pptx_01.pptx',*sorted((ROOT/'tests/fixtures/pptx').glob('*.pptx')),ROOT/'tests/migration/inputs/synthetic.pptx']
    dump('manifest.json',dict(sources=[str(p.relative_to(ROOT))for p in sources],rounds=args.rounds,repeats=args.repeats))
    reports=[]
    for round_id in range(args.rounds):
        for index,source in enumerate(sources):
            for engine in (['python','rust']if(round_id+index)%2==0 else['rust','python']):
                destination=args.output/f'{round_id}-{index:02d}-{engine}'
                with (args.output/(destination.name+'.log')).open('w',encoding='utf-8')as log:
                    command=[sys.executable,str(ROOT/'scripts/migration/accept_docx_delivery.py'),'--worker','--engine',engine,'--source',str(source),'--output',str(destination),'--repeats',str(args.repeats)]
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=False)
                report=json.loads((destination/'report.json').read_text(encoding='utf-8'));report.update(round=round_id,case=str(source.relative_to(ROOT)))
                reports.append(report);dump('reports.json',reports)
                print(f'{report["case"]}: round={round_id+1} {engine} success={report["success"]}',flush=True)
    comparisons=[]
    for source in sources:
        selected=[r for r in reports if r['case']==str(source.relative_to(ROOT))]
        good=all(r['success']for r in selected)
        row=dict(case=str(source.relative_to(ROOT)),success=good,parity=good and len({r.get('canonical_sha256')for r in selected})==1)
        if good:
            for engine in ['python','rust']:
                rows=[r for r in selected if r['engine']==engine]
                row[engine]=dict(warm_seconds=statistics.median([s for r in rows for s in r['warm_seconds']]),cold_seconds=statistics.median(r['cold_seconds']for r in rows),process_peak_working_set=max(r['memory']['after']['os_peak_working_set']for r in rows),baseline_working_set=statistics.median(r['baseline']['working_set']for r in rows),parse_sampled_peak_working_set=max(r['memory']['sampled_peak_working_set']for r in rows))
            row['speedup']=row['python']['warm_seconds']/row['rust']['warm_seconds']
        comparisons.append(row)
    summary=dict(cases=comparisons,all_success=all(r['success']for r in comparisons),all_parity=all(r['parity']for r in comparisons),notes=[f'{len(sources)*args.rounds*2} isolated sequential workers; {args.rounds} alternating rounds','No PDF/OCR models, output disk writes or input rewriting in timed runs','Exact text/structure and pixel-verified asset filename aliases','Memory peaks include import baselines; 2ms samples may miss short spikes','One local machine/corpus, not a general performance guarantee'])
    dump('summary.json',summary)
    if not summary['all_success']or not summary['all_parity']:raise SystemExit(1)
if __name__=='__main__':main()
