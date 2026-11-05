"""Current installed PDF acceptance, sequential isolated Python/Rust workers.

No production patches, model quantization, forced thread settings or old fixture
dependencies. Each worker parses one complete PDF twice using the same engine.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set', 'peak_paged', 'paged',
        'peak_nonpaged', 'nonpaged', 'commit', 'peak_commit', 'private_commit')]


class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page', 'avail_page',
                                              'total_virtual', 'avail_virtual', 'extended')]


KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
KERNEL.GetCurrentProcess.restype = wintypes.HANDLE
PSAPI = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
PSAPI.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]


def memory():
    c = Counters(); c.cb = ctypes.sizeof(c)
    if not PSAPI(KERNEL.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    m = Memory(); m.length = ctypes.sizeof(m)
    if not KERNEL.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set=c.working_set, os_peak_working_set=c.peak_working_set,
                private_commit=c.private_commit, os_peak_commit=c.peak_commit,
                available_physical=m.avail_phys, system_memory_load_percent=m.load)


class Sampling:
    def __init__(self):
        self.samples = []; self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(.1):
            self.samples.append(memory())

    def __enter__(self):
        self.samples.append(memory()); self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(); self.samples.append(memory())

    def summary(self):
        return dict(observed_peak_working_set=max(s['working_set'] for s in self.samples),
                    observed_peak_private_commit=max(s['private_commit'] for s in self.samples),
                    minimum_available_physical=min(s['available_physical'] for s in self.samples),
                    maximum_system_memory_load_percent=max(s['system_memory_load_percent'] for s in self.samples),
                    at_return=self.samples[-1])


def snapshot(result):
    from PIL import Image
    images = []
    for name, data in sorted(result.images.items()):
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            images.append(dict(name=name, mode=image.mode, size=list(image.size),
                               pixels_sha256=sha(image.tobytes()), encoded_sha256=sha(data)))
    return dict(markdown=result.markdown, content_list_json=result.content_list_json,
                middle_json=result.middle_json, assets=images)


def worker(args):
    sys.path.insert(0, str(ROOT))
    destination = args.output.resolve(); destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text('utf-8'))
    case = next(c for c in manifest['cases'] if c['id'] == args.case)
    source = ROOT/case['source']
    assert sha(source.read_bytes()) == case['sha256']
    rows = []; sessions = []
    report = dict(engine=args.engine, case=case, success=False, runs=rows, sessions=sessions)
    started = time.perf_counter()
    try:
        if args.engine == 'python':
            import onnxruntime as ort
            original = ort.InferenceSession.__init__

            def record(self, path, *pos, **kw):
                sessions.append(dict(model=str(path) if isinstance(path, (str, Path)) else '<in-memory model>',
                                     model_bytes=len(path) if isinstance(path, bytes) else None))
                return original(self, path, *pos, **kw)
            ort.InferenceSession.__init__ = record
            from rapid_doc.main import RapidDoc
        else:
            from rapid_doc_rust import RapidDoc
            from rapid_doc_rust._runtime import LazySession
            original = LazySession.descriptor

            def record(self):
                value = original(self); sessions.append(json.loads(value)); return value
            LazySession.descriptor = record
        engine = RapidDoc(formula_config={'model_dir_or_path': manifest['formula_model']},
                          checkbox_config={}, output_dir=destination/'exports')
        report.update(startup_seconds=time.perf_counter()-started,
                      implementation=sys.modules[RapidDoc.__module__].__file__,
                      configuration=dict(parse_method='auto', formula_enable=True, table_enable=True,
                                         seal_enable=True, checkbox_enable=False, pdf_pages_batch=64,
                                         formula_model=manifest['formula_model'], cpu_threads='engine defaults'))
        for label in ['cold', 'warm']:
            gc.collect()
            with Sampling() as samples:
                before_cpu = time.process_time(); before = time.perf_counter()
                result = engine(source, output_dir=destination/label/'exports')
                seconds = time.perf_counter()-before; cpu_seconds = time.process_time()-before_cpu
            row = dict(kind=label, seconds=seconds, cpu_seconds=cpu_seconds, memory=samples.summary())
            data = snapshot(result)
            row.update(pages=len(result.middle_json['pdf_info']), markdown_chars=len(result.markdown),
                       assets=len(result.images), nonempty=bool(result.markdown.strip()))
            dump(destination/(label+'.json'), data)
            rows.append(row); del result, data
            dump(destination/'report.json', report)
        report['success'] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['process_high_water_after_validation'] = memory()
        dump(destination/'report.json', report)
    return 0 if report['success'] else 1


def differences(a, b, path='$'):
    if type(a) is not type(b):
        yield dict(path=path, kind='type', expected=repr(a)[:180], actual=repr(b)[:180]); return
    if isinstance(a, dict):
        for key in sorted(a.keys() ^ b.keys()):
            yield dict(path=path+'.'+key, kind='missing_or_extra')
        for key in sorted(a.keys() & b.keys()):
            yield from differences(a[key], b[key], path+'.'+key)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield dict(path=path, kind='length', expected=len(a), actual=len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f'{path}[{i}]')
    elif a != b:
        yield dict(path=path, kind='value', expected=repr(a)[:180], actual=repr(b)[:180])


def comparison(a, b):
    output = {}
    for field in ['markdown', 'content_list_json', 'middle_json']:
        count = 0; examples = []
        for difference in differences(a[field], b[field]):
            count += 1
            if len(examples) < 30: examples.append(difference)
        output[field] = dict(exact=count == 0, differences=count, examples=examples)
    def pixels(assets):
        return [{k: v for k, v in a.items() if k != 'encoded_sha256'} for a in assets]
    output['assets'] = dict(exact_pixels_and_names=pixels(a['assets']) == pixels(b['assets']),
                            exact_encoded=a['assets'] == b['assets'], expected=a['assets'], actual=b['assets'])
    # Secondary diagnostic only: image encoding can change generated names.
    # No text whitespace, block order, geometry or unknown fields are discarded.
    def asset_markdown(data):
        text = data['markdown']
        for asset in data['assets']:
            text = text.replace(asset['name'], 'pixels:'+asset['pixels_sha256'])
        return text
    output['markdown_after_asset_identity_mapping'] = asset_markdown(a) == asset_markdown(b)
    return output


def summarize(output, manifest):
    rows = []
    for case in manifest['cases']:
        row = dict(case=case, engines={})
        for engine in ['python', 'rust']:
            path = output/case['id']/engine/'report.json'
            if path.exists(): row['engines'][engine] = json.loads(path.read_text('utf-8'))
        if len(row['engines']) == 2 and all(r['success'] for r in row['engines'].values()):
            for engine in ['python', 'rust']:
                folder = output/case['id']/engine
                a = json.loads((folder/'cold.json').read_text('utf-8')); b = json.loads((folder/'warm.json').read_text('utf-8'))
                row[engine+'_repeat'] = comparison(a, b)
            a = json.loads((output/case['id']/'python'/'warm.json').read_text('utf-8'))
            b = json.loads((output/case['id']/'rust'/'warm.json').read_text('utf-8'))
            row['comparison'] = comparison(a, b)
            py, ru = (row['engines'][e]['runs'][1] for e in ['python', 'rust'])
            row.update(warm_speedup=py['seconds']/ru['seconds'],
                       warm_memory_ratio=ru['memory']['observed_peak_working_set']/py['memory']['observed_peak_working_set'])
        rows.append(row)
    dump(output/'summary.json', dict(cases=rows, notes=[
        'Cold means first API call with lazy model loads; OS file caches are not flushed.',
        'Workers are separate and sequential; engine order alternates by document.',
        'Warm means second full parse in the same worker. Exports are enabled equally.',
        'OS high-water values are cumulative per process; sampled warm peaks are separate.',
        'No manual labels are available: parity is measured against the current original Python, not ground truth.']))


def coordinator(args):
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        import logging
        logging.getLogger('pypdf').setLevel(logging.ERROR)
        from pypdf import PdfReader
        import rapid_doc_rust
        from rapid_doc_rust._runtime import MANIFEST, resolve_model
        seen = {}; cases = []
        for path in sorted((ROOT/'demo/pdfs').glob('*.pdf')):
            digest = sha(path.read_bytes())
            if digest in seen:
                seen[digest]['duplicates'].append(str(path.relative_to(ROOT))); continue
            case = dict(id=f'{len(cases)+1:02d}-{path.stem}', source=str(path.relative_to(ROOT)), sha256=digest,
                        pages=len(PdfReader(path).pages), duplicates=[])
            seen[digest] = case; cases.append(case)
        formula = ROOT/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'
        assert sha(formula.read_bytes()) == '52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d'
        models = {role: dict(path=str(resolve_model(role)[0]), sha256=resolve_model(role)[1]) for role in MANIFEST}
        manifest = dict(cases=cases, formula_model=str(formula), models=models,
                        installed=str(Path(rapid_doc_rust.__file__).resolve()), platform=platform.platform(),
                        cpu=platform.processor(), python=sys.version, memory_before=memory(), environment={
                            k: v for k, v in os.environ.items() if k.startswith(('RAPIDDOC_', 'ORT_', 'OMP_', 'MKL_'))})
        dump(output/'manifest.json', manifest)
    manifest = json.loads((output/'manifest.json').read_text('utf-8'))
    selected = [c for c in manifest['cases'] if not args.only or Path(c['source']).stem in args.only]
    for i, case in enumerate(selected):
        order = ['python', 'rust'] if i % 2 == 0 else ['rust', 'python']
        for engine in order:
            folder = output/case['id']/engine
            if args.resume and (folder/'report.json').exists(): continue
            folder.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--engine', engine,
                       '--case', case['id'], '--manifest', str(output/'manifest.json'), '--output', str(folder)]
            print(f"START {case['id']} {engine}", flush=True)
            with (folder.parent/(engine+'.log')).open('wb') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"DONE {case['id']} {engine} exit={result.returncode}", flush=True)
            summarize(output, manifest)
    summarize(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--engine', choices=['python', 'rust'])
    parser.add_argument('--case')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    return worker(args) if args.worker else coordinator(args)


if __name__ == '__main__':
    raise SystemExit(main())
