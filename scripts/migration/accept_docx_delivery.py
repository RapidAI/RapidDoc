"""Sequential isolated public Python/Rust DOCX parity, latency and memory acceptance.

Timed runs exclude imports/engine construction and verification. A separate parse
samples memory; Windows lifetime peaks are reported separately from sampled peaks.
No models, algorithm switches, output files or accuracy tolerances in either engine.
"""
import argparse,gc,hashlib,io,json,os,platform,statistics,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(raw):return hashlib.sha256(raw).hexdigest()

def snapshot(result):
    from PIL import Image
    from export_office_path_pixels import normalized_svg
    aliases={};assets=[]
    for name,data in result.images.items():
        if name.endswith('.svg'):summary=dict(kind='svg',content=normalized_svg(data))
        else:
            with Image.open(io.BytesIO(data))as image:summary=dict(kind='raster',size=list(image.size),rgba_sha256=sha(image.convert('RGBA').tobytes()))
        ident=sha(json.dumps(summary,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode())
        aliases[name]='ASSET['+ident+']';assets.append(summary)
    def rename(value):
        if isinstance(value,str):
            for old,new in aliases.items():value=value.replace(old,new)
            return value
        if isinstance(value,list):return [rename(v)for v in value]
        if isinstance(value,dict):return {k:rename(v)for k,v in value.items()}
        return value
    return dict(markdown=rename(result.markdown),middle_json=rename(result.middle_json),content_list_json=rename(result.content_list_json),assets=sorted(assets,key=lambda v:json.dumps(v,sort_keys=True)))

def worker(args):
    from accept_pdf_delivery import memory
    destination=args.output;destination.mkdir(parents=True,exist_ok=False)
    report=dict(engine=args.engine,source=str(args.source.resolve()),success=False)
    started=time.perf_counter()
    try:
        from loguru import logger
        logger.remove()
        if args.engine=='python':from rapid_doc import RapidDoc
        else:from rapid_doc_rust import RapidDoc
        engine=RapidDoc()
        report['import_and_init_seconds']=time.perf_counter()-started
        report['baseline']=memory();latencies=[]
        result=None
        for _ in range(args.repeats+1):
            result=None;gc.collect();begin=time.perf_counter();result=engine(args.source);latencies.append(time.perf_counter()-begin)
        report['cold_seconds']=latencies[0];report['warm_seconds']=latencies[1:];report['warm_median_seconds']=statistics.median(latencies[1:])
        report['logical_pages']=len(result.middle_json['pdf_info']);report['assets']=len(result.images)
        # Verification and pixel decoding must not inflate the memory reading.
        result=None;gc.collect();samples=[memory()];stop=threading.Event()
        def sample():
            while not stop.wait(.002):samples.append(memory())
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        try:result=engine(args.source)
        finally:stop.set();thread.join();samples.append(memory())
        report['memory']=dict(sample_interval_seconds=.002,samples=len(samples),before=samples[0],after=samples[-1],sampled_peak_working_set=max(s['working_set']for s in samples),sampled_peak_private_commit=max(s['private_commit']for s in samples),minimum_available_physical=min(s['available_physical']for s in samples))
        canonical=snapshot(result);dump(destination/'snapshot.json',canonical)
        report['canonical_sha256']=sha(json.dumps(canonical,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode())
        report['success']=True
    except Exception as e:report.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    dump(destination/'report.json',report)
    print(json.dumps({k:report[k]for k in ['engine','source','success']},ensure_ascii=False),flush=True)
    if not report['success']:raise SystemExit(1)

def stress(source,destination,copies):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    with zipfile.ZipFile(source)as z,zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED)as out:
        for member in z.infolist():
            raw=z.read(member)
            if member.filename=='word/document.xml':
                # Repeat the unmodified body subtree with all package resources
                # preserved; keep only the final section properties at the end.
                from lxml import etree
                import copy
                root=etree.fromstring(raw);body=root.find('{'+ns+'}body');children=list(body);section=[n for n in children if n.tag=='{'+ns+'}sectPr'];blocks=[n for n in children if n.tag!='{'+ns+'}sectPr']
                for n in children:body.remove(n)
                for _ in range(copies):
                    for n in blocks:body.append(copy.deepcopy(n))
                for n in section:body.append(n)
                raw=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
            out.writestr(member,raw)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--engine',choices=['python','rust']);parser.add_argument('--source',type=Path);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--rounds',type=int,default=2);parser.add_argument('--case',action='append');parser.add_argument('--stress-copies',type=int,default=25)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.worker:return worker(args)
    assert args.repeats>0 and args.rounds>0
    args.output.mkdir(parents=True,exist_ok=False)
    sources=sorted((ROOT/'demo/docx').glob('*.docx'))+[ROOT/'demo/ms_office_docs/docx_01.docx']
    if not args.case or 'stress.docx'in args.case:
        stress_path=args.output/'stress.docx';stress(ROOT/'demo/docx/test.docx',stress_path,args.stress_copies);sources.append(stress_path)
    if args.case:sources=[p for p in sources if p.name in args.case]
    assert sources,'no cases selected'
    manifest=dict(python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),repeats=args.repeats,rounds=args.rounds,stress_copies=args.stress_copies,cases=[dict(source=str(p),sha256=sha(p.read_bytes()),bytes=p.stat().st_size)for p in sources]);dump(args.output/'manifest.json',manifest)
    reports=[]
    for round_id in range(args.rounds):
        for index,source in enumerate(sources):
            order=['python','rust']if (round_id+index)%2==0 else['rust','python']
            for engine in order:
                destination=args.output/f'{index:02d}-{source.stem}-{round_id}-{engine}'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--engine',engine,'--source',str(source),'--output',str(destination),'--repeats',str(args.repeats)]
                with (args.output/(destination.name+'.log')).open('w',encoding='utf-8')as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=False)
                report=json.loads((destination/'report.json').read_text(encoding='utf-8'));report.update(round=round_id,case=source.name);reports.append(report)
                dump(args.output/'reports.json',reports)
                print(f'{source.name}: round {round_id+1} {engine} '+(f"{report['warm_median_seconds']:.4f}s"if report['success']else report['error']),flush=True)
    comparisons=[]
    for source in sources:
        selected=[r for r in reports if r['case']==source.name];good=all(r['success']for r in selected)
        row=dict(case=source.name,success=good,parity=good and len({r.get('canonical_sha256')for r in selected})==1)
        if good:
            for engine in ['python','rust']:
                rows=[r for r in selected if r['engine']==engine];row[engine]=dict(warm_seconds=statistics.median([s for r in rows for s in r['warm_seconds']]),cold_seconds=statistics.median(r['cold_seconds']for r in rows),process_peak_working_set=max(r['memory']['after']['os_peak_working_set']for r in rows),baseline_working_set=statistics.median(r['baseline']['working_set']for r in rows),parse_sampled_peak_working_set=max(r['memory']['sampled_peak_working_set']for r in rows),parse_sampled_peak_private_commit=max(r['memory']['sampled_peak_private_commit']for r in rows))
            row['speedup']=row['python']['warm_seconds']/row['rust']['warm_seconds'];row['process_peak_rust_over_python']=row['rust']['process_peak_working_set']/row['python']['process_peak_working_set']
        comparisons.append(row)
    summary=dict(cases=comparisons,all_success=all(r['success']for r in comparisons),all_parity=all(r['parity']for r in comparisons),notes=['Independent sequential processes, alternating execution order','Public API imports/init reported separately; no PDF/OCR models preloaded','Times exclude verification/disk output; sampled-memory parse is separate from timed runs','OS lifetime peaks include imports and prior warm runs; sampled phase peaks can miss short spikes','Asset filename aliases only after exact pixel/SVG content hashing; text and structure remain strict','Generated stress document is repetitive and is not independent client-document coverage'])
    dump(args.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not summary['all_success']or not summary['all_parity']:raise SystemExit(1)
if __name__=='__main__':main()
