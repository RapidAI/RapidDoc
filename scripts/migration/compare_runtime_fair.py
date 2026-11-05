"""Sequential Python/Rust PDF controls with equal exports and Windows peak memory."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import sys
import time
import os

root = Path(__file__).resolve().parents[2]
mode, namespace = sys.argv[1:3]
assert mode in ('python', 'installed', 'dev')
assert Path(namespace).name == namespace
stems = sys.argv[3:] or ['比亚迪财报', 'small_ocr', '03']
destination = root / 'build' / namespace
destination.mkdir(exist_ok=False)
telemetry = None
if os.environ.get('RAPIDDOC_BENCH_TELEMETRY') == '1':
    import atexit
    from runtime_telemetry import Telemetry
    telemetry = Telemetry(destination/'telemetry.jsonl')
    atexit.register(telemetry.close)
fixture = json.loads(Path(os.environ.get('RAPIDDOC_BENCH_REFERENCE',str(root / 'build/migration/user-seven-pdfs-20261029.json'))).read_text(encoding='utf-8'))
prior = {c['source']: c for c in json.loads(Path(os.environ.get('RAPIDDOC_BENCH_PRIOR',str(root / 'build/migration/user-seven-rust-before-phase-20261029.json'))).read_text(encoding='utf-8'))['cases']}
formula_override = os.environ.get('RAPIDDOC_BENCH_FORMULA_MODEL')
if formula_override is not None:
    formula_override = str(Path(formula_override).resolve(strict=True))

def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ('peak_working_set', 'working_set', 'peak_paged', 'paged',
            'peak_nonpaged', 'nonpaged', 'commit', 'peak_commit', 'private_commit')]
    values = Counters(); values.cb = ctypes.sizeof(values)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    assert api(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb)
    return {key: getattr(values, key) for key in ('peak_working_set', 'working_set', 'commit', 'peak_commit', 'private_commit')}

if mode == 'python':
    sys.path.insert(0, str(root))
    arena_mode = os.environ.get('RAPIDDOC_BENCH_ARENA_MODE')
    if arena_mode is not None:
        assert arena_mode in ('separate', 'shared')
        import onnxruntime as ort
        if arena_mode == 'shared':
            ort.create_and_register_allocator_v2('CPUExecutionProvider',
                ort.OrtMemoryInfo('Cpu', ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT),
                {}, ort.OrtArenaCfg(0, 0, -1, -1))
        arena_session_init = ort.InferenceSession.__init__
        def configured_arena_session(self, path_or_bytes, sess_options=None, **kwargs):
            options = sess_options if sess_options is not None else ort.SessionOptions()
            options.enable_cpu_mem_arena = True
            options.add_session_config_entry('session.use_env_allocators', '1' if arena_mode=='shared' else '0')
            return arena_session_init(self, path_or_bytes, sess_options=options, **kwargs)
        ort.InferenceSession.__init__ = configured_arena_session
    if os.environ.get('RAPIDDOC_BENCH_FORMULA_DUPLICATES'):
        from rapid_doc.model.formula.rapid_formula_self.inference_engine.onnxruntime.main import OrtInferSession
        original_formula_run = OrtInferSession.__call__
        formula_calls = []
        def measured_formula_run(self, tensor):
            hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in tensor]
            result = original_formula_run(self, tensor)
            formula_calls.extend(hashes)
            (destination/'formula-input-hashes.json').write_text(json.dumps(formula_calls), encoding='utf-8')
            return result
        OrtInferSession.__call__ = measured_formula_run
    stop_spinning = os.environ.get('RAPIDDOC_BENCH_FORCE_SPIN_STOP')
    if stop_spinning is not None:
        import onnxruntime as ort
        original_session_init = ort.InferenceSession.__init__
        def configured_session_init(self, path_or_bytes, sess_options=None, **kwargs):
            options = sess_options if sess_options is not None else ort.SessionOptions()
            options.add_session_config_entry('session.force_spinning_stop', stop_spinning)
            return original_session_init(self, path_or_bytes, sess_options=options, **kwargs)
        ort.InferenceSession.__init__ = configured_session_init
    sorted_batch = os.environ.get('RAPIDDOC_BENCH_FORMULA_SORTED_BATCH')
    if sorted_batch is not None:
        from rapid_doc.model.formula.rapid_formula_model import RapidFormulaModel
        original_formula_batch = RapidFormulaModel.batch_predict
        def scheduled_formula_batch(self, images, batch_size):
            order = sorted(range(len(images)), key=lambda i: images[i].shape[0] * images[i].shape[1])
            scheduled = original_formula_batch(self, [images[i] for i in order], int(sorted_batch))
            assert len(scheduled) == len(images)
            restored = [None] * len(images)
            for index, text in zip(order, scheduled):
                restored[index] = text
            return restored
        RapidFormulaModel.batch_predict = scheduled_formula_batch
    import demo
    original_config = demo._build_config
    def common_config():
        configs = list(original_config())
        configs[1] = {**configs[1], 'seal_enable': False}
        formula_model = root / 'build/migration/models/pp_formulanet_plus_m.onnx'
        if not formula_model.is_file():formula_model=root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'
        if formula_override is None:
            assert hashlib.sha256(formula_model.read_bytes()).hexdigest() == '52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d'
        else:
            formula_model = Path(formula_override)
        configs[2] = {**configs[2], 'model_dir_or_path': str(formula_model)}
        return tuple(configs)
    demo._build_config = common_config
else:
    if mode == 'dev':
        sys.path.insert(0, os.environ.get('RAPIDDOC_BENCH_SDK_ROOT', str(root / 'build/api-dev-sdk')))
    import rapid_doc_rust as sdk
    batch_override = os.environ.get('RAPIDDOC_BENCH_FORMULA_BATCH')
    formula_options = {} if batch_override is None else {'batch_num': int(batch_override)}
    if formula_override is not None:
        formula_options['model_dir_or_path'] = formula_override
    cpu_options={}
    spinning=os.environ.get('RAPIDDOC_BENCH_CPU_SPINNING')
    dynamic=os.environ.get('RAPIDDOC_BENCH_CPU_DYNAMIC')
    if spinning is not None:
        assert spinning in ('0','1')
        cpu_options['cpu_spinning']=spinning=='1'
    if dynamic is not None:cpu_options['cpu_dynamic_block_base']=int(dynamic)
    grouped={'layout':{},'ocr':{'seal_enable':False},'formula':formula_options,'table':{}}
    shared=os.environ.get('RAPIDDOC_BENCH_SHARED_ARENA')
    if shared is not None:
        assert shared in ('0','1')
        for config in grouped.values():config.setdefault('engine_cfg',{})['cpu_shared_arena']=shared=='1'
    stop=os.environ.get('RAPIDDOC_BENCH_FORCE_SPIN_STOP')
    if stop is not None:
        assert stop in ('0','1')
        for config in grouped.values():config.setdefault('engine_cfg',{})['cpu_force_spinning_stop']=stop=='1'
    groups=os.environ.get('RAPIDDOC_BENCH_NO_SPIN_GROUPS','').split(',')
    for group in groups:
        if group:grouped[group].setdefault('engine_cfg',{})['cpu_spinning']=False
    threads=os.environ.get('RAPIDDOC_BENCH_CPU_THREADS')
    if threads is not None:
        for group in ['layout','ocr','formula']:grouped[group].setdefault('engine_cfg',{})['intra_op_num_threads']=int(threads)
    engine = sdk.RapidDoc(output_dir=destination, layout_config=grouped['layout'], ocr_config=grouped['ocr'], formula_config=grouped['formula'], table_config=grouped['table'], **cpu_options)
    (destination/'configuration.json').write_text(json.dumps(dict(module=sdk.__file__,formula_options=formula_options,cpu_options=cpu_options,model_configs=grouped)),encoding='utf-8')

from PIL import Image
summaries = []
for stem in stems:
    if telemetry is not None:telemetry.document = stem
    case = next(c for c in fixture['cases'] if Path(c['source']).stem == stem)
    source = root / case['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case['sha256']
    started_cpu = time.process_time()
    started = time.perf_counter()
    if mode == 'python':
        demo.do_parse(str(destination), [stem], [source.read_bytes()],
            f_draw_layout_bbox=False, f_draw_span_bbox=False, f_dump_orig_pdf=False,
            f_dump_model_output=False, f_dump_md_html=False, f_dump_md_docx=False)
    else:
        result = engine(source)
    folder = destination / stem / 'auto'
    actual = dict(markdown=(folder / (stem + '.md')).read_text(encoding='utf-8'),
        middle_json=json.loads((folder / (stem + '_middle.json')).read_text(encoding='utf-8')),
        content_list_json=json.loads((folder / (stem + '_content_list.json')).read_text(encoding='utf-8')))
    seconds = time.perf_counter() - started
    cpu_seconds = time.process_time() - started_cpu
    usage = memory()
    assert actual['markdown'] == case['expected']['markdown'], stem
    assert actual['content_list_json'] == case['expected']['content_list_json'], stem
    expected_middle = case['expected']['middle_json'] if mode == 'python' else prior[case['source']]['expected']['middle_json']
    assert actual['middle_json'] == expected_middle, stem
    assets = []
    for item in case['assets']:
        data = (folder / 'images' / item['name']).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            assets.append(dict(name=item['name'], mode=im.mode, size=list(im.size),
                               pixels_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    assert assets == case['assets'], stem
    summaries.append(dict(mode=mode, source=case['source'], seconds=seconds, cpu_seconds=cpu_seconds, memory_bytes=usage,
        output_matches_reference=True, models_and_first_load_included=True,
        export_scope='Equal Markdown/middle/content/images exports and JSON readback; no visual overlays or original-PDF copies; seal disabled in both'))
    (destination / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Fair runtime control:', json.dumps(summaries[-1], ensure_ascii=False), flush=True)
