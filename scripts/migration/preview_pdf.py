"""用已安装的 Rust wheel 和本地已验证 CPU 模型转换真实 PDF，不导入旧解析器。

在项目根目录运行（模型路径/哈希来自 build/migration 的既有模型清单）：
build/rust-wheel-env/Scripts/python.exe -I scripts/migration/preview_pdf.py demo/pdfs/example.pdf
"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

from rapid_doc_rust import OnnxSession, parse_pdf_models


ROOT = Path(__file__).resolve().parents[2]


def manifest(name):
    return json.loads((ROOT / 'build/migration' / name).read_text(encoding='utf-8'))


def load_models(runtime, threads, *, native_only=False):
    config = manifest('model-document-default.json')['models']
    layout = manifest('legacy-model.json')
    formula = manifest('formula-decode.json')['model']
    routing = manifest('legacy-table-routing.json')
    cls = routing['classifiers']

    def session(path, sha):
        print(f'Loading CPU model: {path}', flush=True)
        return OnnxSession(runtime, ROOT / path, sha, threads=threads)

    sessions = {
        key: session(config[key + '_model'], config[key + '_sha256'])
        for key in ('det', 'rec', 'orientation', 'table')
    }
    return dict(
        layout_model=None if native_only else session(layout['model'], layout['model_sha256']),
        detector=sessions['det'], recognizer=sessions['rec'],
        orientation=sessions['orientation'], table_model=sessions['table'],
        formula_model=session(formula['path'], formula['sha256']),
        table_variant='unet-slanet-plus',
        wired_model=session(routing['unet_model'], routing['unet_sha256']),
        classifier=session(cls['model'], cls['sha256']),
        secondary_classifier=session(cls['qmodel'], cls['qsha256']),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdfs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, help='新输出目录；已有目录不覆盖')
    parser.add_argument('--runtime', type=Path, default=ROOT / '.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll')
    parser.add_argument('--start-page', type=int, default=0, help='从0开始')
    parser.add_argument('--end-page', type=int, help='不包含此页；默认到文档末尾')
    parser.add_argument('--dpi', type=float, default=144.)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--docvortex-mode', choices=['off','rule'], default='off')
    parser.add_argument('--native-only', action='store_true', help='原生版面；空页保留为空，不执行模型版面回退')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if args.native_only and args.docvortex_mode=='off':parser.error('--native-only requires --docvortex-mode rule')
    if not args.runtime.is_file():
        parser.error(f'ONNX Runtime not found: {args.runtime}')
    inputs = [p.resolve() for p in args.pdfs]
    if any(not p.is_file() for p in inputs):
        parser.error('one or more input PDFs do not exist')
    if len({p.stem.casefold() for p in inputs}) != len(inputs):
        parser.error('input PDF stems must be unique')
    output = (args.output or ROOT / 'build/rust-pdf-preview' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if output.exists():
        parser.error(f'output directory already exists: {output}')
    # 只读取模型清单中的路径和哈希；所有页面版面/公式均真实推理，不读取参考响应。
    models = load_models(args.runtime.resolve(), args.threads,native_only=args.native_only)
    output.mkdir(parents=True)
    summary = []
    for source in inputs:
        destination = output / source.stem
        destination.mkdir()
        start = time.perf_counter()
        print(f'Converting: {source}', flush=True)
        try:
            result = parse_pdf_models(
                source, **models, native_text=True, formula_enabled=True,
                start_page=args.start_page, end_page=args.end_page, dpi=args.dpi,
                docvortex_mode=args.docvortex_mode,native_only=args.native_only,
            )
            (destination / 'document.md').write_text(result['markdown'], encoding='utf-8')
            for name, data in result['assets'].items():
                path = destination / name
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('asset path escapes document directory')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            metadata = {k: v for k, v in result.items() if k != 'assets'}
            (destination / 'result.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            record = dict(source=str(source), status='ok', source_pages=result['source_page_count'],
                          page_range=result['page_range'], markdown_characters=len(result['markdown']),
                          assets=len(result['assets']), seconds=round(time.perf_counter() - start, 3),
                          markdown=str(destination / 'document.md'))
        except Exception as error:
            (destination / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            record = dict(source=str(source), status='error', error=str(error),
                          seconds=round(time.perf_counter() - start, 3))
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output: {output}', flush=True)
    return int(any(r['status'] != 'ok' for r in summary))


if __name__ == '__main__':
    raise SystemExit(main())
