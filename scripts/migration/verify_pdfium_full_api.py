"""Compare each engine's isolated output with its own mixed-process output.

This is PDFium coexistence acceptance, not a Python/Rust quality/speed benchmark.
"""
import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ['示例1-论文模板', 'small_ocr']


def sha(value):
    return hashlib.sha256(value).hexdigest()


def result_summary(result):
    from PIL import Image
    images = []
    for name, data in sorted(result.images.items()):
        with Image.open(io.BytesIO(data)) as image:
            images.append(dict(name=name, mode=image.mode, size=list(image.size), pixels=sha(image.tobytes())))
    metadata = dict(markdown=result.markdown, middle=result.middle_json, content=result.content_list_json)
    return dict(markdown_chars=len(result.markdown), images=images,
                metadata_sha256=sha(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode('utf-8')))


def worker(mode, destination, sdk_root):
    if sdk_root:
        sys.path.insert(0, str(sdk_root))
    # ROOT is needed only for the original Python engine, never to replace the SDK.
    sys.path.insert(0, str(ROOT))
    destination.mkdir(parents=True, exist_ok=False)
    models = dict(layout_config={'model_type': 'pp_doclayoutv3'},
                  formula_config={'model_type': 'pp_formulanet_plus_m', 'model_dir_or_path':
                    str(ROOT/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx')},
                  ocr_config={'seal_enable': False})
    summaries = {}
    if mode in ('rust-only', 'mixed'):
        from rapid_doc_rust import RapidDoc as RustDoc
        rust_engine = RustDoc(**models)
    if mode in ('python-only', 'mixed'):
        from rapid_doc.main import RapidDoc as PythonDoc
        from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType as LayoutType
        from rapid_doc.model.formula.rapid_formula_self.utils.typings import ModelType as FormulaType
        python_models = {key: dict(value) for key, value in models.items()}
        python_models['layout_config']['model_type'] = LayoutType.PP_DOCLAYOUTV3
        python_models['formula_config']['model_type'] = FormulaType.PP_FORMULANET_PLUS_M
        python_engine = PythonDoc(**python_models)
    def parse(engine, label):
        rows = []
        for name in SOURCES:
            source = ROOT/'demo/pdfs'/(name+'.pdf')
            result = engine(source, output_dir=destination/label, start_page_id=0, end_page_id=0)
            assert result.markdown.strip(), (label, name, 'empty output')
            rows.append(dict(source=name, **result_summary(result)))
        summaries[label] = rows
    if mode == 'rust-only':
        parse(rust_engine, 'rust')
        assert 'pypdfium2' not in sys.modules
    elif mode == 'python-only':
        parse(python_engine, 'python')
    else:
        import pypdfium2 as pdfium
        source = ROOT/'demo/pdfs/示例1-论文模板.pdf'
        with pdfium.PdfDocument(source) as document:
            page = document[0]
            text = page.get_textpage()
            before = text.get_text_range()
            parse(rust_engine, 'rust-before')
            assert text.get_text_range() == before
            parse(python_engine, 'python')
            assert text.get_text_range() == before
            parse(rust_engine, 'rust-after')
            assert text.get_text_range() == before
            text.close()
            page.close()
        assert summaries['rust-before'] == summaries['rust-after']
    (destination/'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk-root', type=Path)
    parser.add_argument('--worker', choices=['rust-only', 'python-only', 'mixed'])
    args = parser.parse_args()
    destination = args.output.resolve()
    assert not destination.exists()
    if args.worker:
        worker(args.worker, destination, args.sdk_root.resolve() if args.sdk_root else None)
        return
    destination.mkdir(parents=True, exist_ok=False)
    reports = {}
    for mode in ['rust-only', 'python-only', 'mixed']:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', mode, '--output', str(destination/mode)]
        if args.sdk_root:
            command.extend(['--sdk-root', str(args.sdk_root.resolve())])
        with (destination/(mode+'.log')).open('wb') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
        reports[mode] = json.loads((destination/mode/'summary.json').read_text(encoding='utf-8'))
        print('Passed full API:', mode, flush=True)
    assert reports['mixed']['rust-before'] == reports['rust-only']['rust']
    assert reports['mixed']['rust-after'] == reports['rust-only']['rust']
    assert reports['mixed']['python'] == reports['python-only']['python']
    (destination/'acceptance.json').write_text(json.dumps(dict(all_passed=True, sources=SOURCES,
        rust_mixed_matches_isolated=True, python_mixed_matches_isolated=True,
        held_python_textpage_unchanged=True, stamp_enabled=False, benchmark=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
