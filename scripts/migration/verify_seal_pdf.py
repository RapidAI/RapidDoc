"""Seal-only delivery acceptance on the existing four-page stamp PDF.

Compare emitted seal text with the original Python pipeline; verify the Rust
default and explicit disable switch. This is not the final performance benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def seals(value, page=None):
    result = []
    if isinstance(value, dict):
        page = value.get('page_idx', page)
        if value.get('original_label') == 'seal' and 'content' in value:
            content = value['content']
            result.append(dict(page=page, text=content if isinstance(content, list) else [content]))
        for child in value.values():
            result.extend(seals(child, page))
    elif isinstance(value, list):
        for child in value:
            result.extend(seals(child, page))
    return result


def worker(args):
    if args.sdk_root:
        sys.path.insert(0, str(args.sdk_root.resolve()))
    sys.path.insert(0, str(ROOT))
    roles = []
    if args.engine == 'rust':
        from rapid_doc_rust import RapidDoc
        from rapid_doc_rust._runtime import LazySession
        original = LazySession.descriptor

        def record(self):
            roles.append(self.role)
            return original(self)
        LazySession.descriptor = record
        model_type = 'pp_doclayoutv3'
    else:
        from rapid_doc.main import RapidDoc
        from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType
        model_type = ModelType.PP_DOCLAYOUTV3
    ocr = {'EngineConfig.onnxruntime.intra_op_num_threads': 1,
           'EngineConfig.onnxruntime.inter_op_num_threads': 1}
    if args.disable:
        ocr['seal_enable'] = False
    engine = RapidDoc(layout_config={'model_type': model_type, 'conf_thresh': .5,
                                    'engine_cfg': {'intra_op_num_threads': 1}},
                      ocr_config=ocr, formula_enable=False, table_enable=False)
    result = engine(ROOT/'demo/pdfs/含印章.pdf', output_dir=args.output_dir,
                    parse_method='ocr', formula_enable=False, table_enable=False)
    middle = json.loads(result.middle_json) if isinstance(result.middle_json, str) else result.middle_json
    summary = dict(engine=args.engine, disabled=args.disable, seals=seals(middle), resolved_models=roles,
                   pages=len(middle['pdf_info']), markdown_chars=len(result.markdown))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir/'document.md').write_text(result.markdown, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--engine', choices=['rust', 'python'])
    parser.add_argument('--disable', action='store_true')
    parser.add_argument('--validate-existing', action='store_true', help='Validate saved worker reports without repeating model inference')
    args = parser.parse_args()
    if args.engine:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=args.validate_existing)
    rows = {}
    for label, engine, disable in [('python', 'python', False), ('rust', 'rust', False), ('rust-disabled', 'rust', True)]:
        destination = args.output_dir/label
        if args.validate_existing:
            rows[label] = json.loads((destination/'summary.json').read_text('utf-8'))
            assert rows[label]['engine'] == engine and rows[label]['disabled'] == disable
            continue
        command = [sys.executable, __file__, '--engine', engine, '--output-dir', str(destination.resolve())]
        if args.sdk_root:
            command += ['--sdk-root', str(args.sdk_root.resolve())]
        if disable:
            command += ['--disable']
        with (args.output_dir/(label+'.log')).open('wb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        rows[label] = json.loads((destination/'summary.json').read_text('utf-8'))
    assert rows['python']['seals'], 'The reference must contain detected seals'
    assert any(s['text'] for s in rows['python']['seals']), 'The reference must contain recognized seal text'
    assert rows['python']['seals'] == rows['rust']['seals'], 'Python/Rust seal text differs'
    assert 'seal_det' in rows['rust']['resolved_models'], 'Default seal recognition was not requested'
    assert 'seal_det' not in rows['rust-disabled']['resolved_models'], 'Disabled seal detector was loaded'
    assert not any(isinstance(text,str) and text.strip() for s in rows['rust-disabled']['seals'] for text in s['text']), 'Disabled seals produced text'
    (args.output_dir/'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(pages=rows['rust']['pages'], seal_entries=len(rows['rust']['seals']), passed=True)))


if __name__ == '__main__':
    main()
