"""Wait for the complete cohort, then reverse engine order on representatives."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--only',nargs='+',default=['03','49－长表格-密集型','small_ocr'])
    p.add_argument('--after-status',action='store_true')
    p.add_argument('--status-file',default='reverse-status.json')
    args=p.parse_args();root=args.input.resolve();deadline=time.monotonic()+7200
    while True:
        try:
            summary=json.loads((root/'summary.json').read_text('utf-8'))
            if all(len(c['engines'])==2 for c in summary['cases'])and(not args.after_status or(root/'reverse-status.json').exists()):break
        except (FileNotFoundError,json.JSONDecodeError):pass
        if time.monotonic()>deadline:raise TimeoutError('Initial cohort did not finish; saved reports retained')
        time.sleep(2)
    manifest=json.loads((root/'manifest.json').read_text('utf-8'));rows=[]
    for stem in args.only:
        case=next(c for c in manifest['cases']if Path(c['source']).stem==stem)
        initial=next(c for c in summary['cases']if c['case']['id']==case['id'])
        if not all(r['success']for r in initial['engines'].values()):
            rows.append(dict(case=case['id'],skipped='initial parse failed'));continue
        starts=[]
        for name in ['pdf-delivery-acceptance-20261030.log','pdf-delivery-acceptance-20261030-remaining.log']:
            starts.extend(s for s in (ROOT/'build'/name).read_text('utf-8',errors='replace').splitlines()if s.startswith('START '+case['id']+' '))
        first=starts[0].split()[-1];order=['rust','python']if first=='python'else['python','rust']
        for engine in order:
            folder=root/'reverse'/case['id']/engine;folder.parent.mkdir(parents=True,exist_ok=True)
            if (folder/'report.json').exists():continue
            print('START reverse',case['id'],engine,flush=True)
            command=[sys.executable,str(ROOT/'scripts/migration/accept_pdf_delivery.py'),'--worker',
                     '--manifest',str(root/'manifest.json'),'--case',case['id'],'--engine',engine,'--output',str(folder)]
            with (folder.parent/(engine+'.log')).open('wb')as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            rows.append(dict(case=case['id'],engine=engine,exit_code=result.returncode))
            print('DONE reverse',case['id'],engine,result.returncode,flush=True)
    (root/args.status_file).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/migration/report_pdf_delivery.py'),'--input',str(root)],check=True,cwd=ROOT)


if __name__=='__main__':main()
