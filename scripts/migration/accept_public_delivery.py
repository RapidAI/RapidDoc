"""Installed Python/Rust public API acceptance for the seven supported formats.

Workers are isolated and sequential. Office covers full documents and API variants;
PDF covers explicit page ranges with default recognition models, not the full corpus.
No timing claim is made from these correctness runs.
"""
import argparse
import ast
import base64
import io
import inspect
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def public_signatures():
    """Read the Python declaration without loading its PDFium into this process."""
    from rapid_doc_rust import RapidDoc
    original = next(n for n in ast.parse((ROOT / 'rapid_doc/main.py').read_text(encoding='utf8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'RapidDoc')
    rows = []
    for name in ['__init__', '__call__', 'warmup']:
        node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        expected = [a.arg for a in node.args.args]
        signature = inspect.signature(getattr(RapidDoc, name))
        assert [p for p in signature.parameters if p in expected] == expected, name
        extras = [p for p in signature.parameters if p not in expected]
        assert all(signature.parameters[p].default is not inspect.Parameter.empty for p in extras)
        for argument, value in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
            default = 'mm_markdown' if isinstance(value, ast.Attribute) and value.attr == 'MM_MD' else ast.literal_eval(value)
            assert signature.parameters[argument.arg].default == default, (name, argument.arg)
        rows.append(dict(method=name, common_parameters=expected, optional_rust_extensions=extras, common_defaults_equal=True))
    return rows


class Writer:
    def __init__(self):
        self.data = {}

    def write(self, path, data):
        self.data[path] = data

    def write_string(self, path, text):
        self.write(path, text.encode('utf8'))


def canonical(result):
    from accept_docx_delivery import snapshot
    # Compare data URI pixels by the same strict identity rule as file assets.
    markdown = result.markdown
    if isinstance(markdown, str):
        for name, data in result.images.items():
            mime = mimetypes.guess_type(name)[0] or 'image/png'
            uri = 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')
            markdown = markdown.replace(uri, 'assets/' + name)
    return snapshot(SimpleNamespace(markdown=markdown, images=result.images,
                    middle_json=result.middle_json, content_list_json=result.content_list_json))


def worker(args):
    from loguru import logger
    logger.remove()
    if args.engine == 'python':
        from rapid_doc import RapidDoc
    else:
        from rapid_doc_rust import RapidDoc
    source = args.source.resolve()
    data = source.read_bytes()
    office = source.suffix.lower() != '.pdf'
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    observed_errors = []
    engine = RapidDoc()
    defaults = {key: getattr(engine, key) for key in ['parse_method','formula_enable','table_enable','lang','image_dir_name','image_output_mode','pdf_pages_batch']}
    if office and args.engine == 'rust':
        def forbidden(*a, **kw):
            raise AssertionError('Office parsing must not load recognition models')
        engine._get_models = forbidden

    def check(label, operation):
        try:
            result, details = operation()
            if result is not None:
                results = result if isinstance(result, list) else [result]
                for item in results:
                    assert isinstance(item.images, dict)
                    assert isinstance(item.middle_json['pdf_info'], list)
                    assert isinstance(item.content_list_json, list)
                    markdown, images = item
                    assert markdown == item.markdown and images is item.images
                snapshots = [canonical(item) for item in results]
                dump(args.output / (label + '.json'), snapshots if isinstance(result, list) else snapshots[0])
            else:
                dump(args.output / (label + '.json'), details)
            rows.append(dict(check=label, success=True, **details))
        except Exception as error:
            rows.append(dict(check=label, success=False, error=str(error), traceback=traceback.format_exc()))
        dump(args.output / 'report.json', dict(engine=args.engine, source=str(source), checks=rows,
             defaults=defaults, observed_errors=observed_errors, success=all(r['success'] for r in rows)))

    if args.errors_only:
        def probe(item, must_reject=True, **kwargs):
            try:
                result = engine(item, **kwargs)
            except Exception as error:
                observed_errors.append(dict(type=type(error).__name__,code=getattr(error,'code',None),message=str(error)))
                return None, dict(rejected=True)
            assert not must_reject, 'Invalid input was silently accepted'
            return result, dict(rejected=False)
        missing = args.output / ('missing' + source.suffix)
        check('missing-file', lambda: probe(missing))
        check('lang-count', lambda: probe([source], lang=['ch', 'en']))
        empty = args.output / ('empty' + source.suffix)
        empty.write_bytes(b'')
        check('empty-file', lambda: probe(empty, must_reject=False))
        corrupt = args.output / ('corrupt' + source.suffix)
        corrupt.write_bytes((b'%PDF-1.7\ninvalid' if not office else
                            b'PK\x03\x04invalid' if source.suffix.endswith('x') else b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1invalid'))
        check('corrupt-file', lambda: probe(corrupt))
        return

    if not office:
        for label, first, last, window in [('selected-pages', 0, 1, 64), ('single-page-window', 1, 1, 1)]:
            def pdf(first=first, last=last, window=window):
                parser = RapidDoc(pdf_pages_batch=window)
                result = parser(source, start_page_id=first, end_page_id=last)
                assert len(result.middle_json['pdf_info']) == last - first + 1
                assert result.markdown.strip(), 'Selected PDF pages returned no content'
                return result, dict(start_page_id=first, end_page_id=last, pdf_pages_batch=window)
            check(label, pdf)
        return

    def parse(item, **kwargs):
        result = engine(item, **kwargs)
        return result, dict(pages=len(result.middle_json['pdf_info']), assets=len(result.images))

    check('path', lambda: parse(source))
    check('bytes', lambda: parse(data))
    check('stream', lambda: parse(io.BytesIO(data)))

    def batch():
        results = engine([source, data], lang=['ch', 'ch'])
        assert len(results) == 2 and canonical(results[0]) == canonical(results[1])
        return results[0], dict(count=len(results))
    check('batch', batch)
    check('prefix', lambda: parse(source, image_dir_name='assets'))
    check('data-uri', lambda: parse(source, image_dir_name='assets', image_output_mode='data_uri'))

    def writers():
        image_writer, md_writer = Writer(), Writer()
        result = engine(source, image_writer=image_writer, md_writer=md_writer,
                        f_dump_middle_json=False, f_dump_content_list=False)
        assert image_writer.data == result.images
        assert md_writer.data == {source.stem + '.md': result.markdown.encode('utf8')}
        return result, dict(written_images=len(image_writer.data), written_texts=sorted(md_writer.data))
    check('writers', writers)

    def disk():
        destination = args.output / 'exports'
        result = engine(source, output_dir=destination)
        folder = destination / source.stem / 'office'
        assert (folder / (source.stem + '.md')).read_text(encoding='utf8') == result.markdown
        assert json.loads((folder / (source.stem + '_middle.json')).read_text(encoding='utf8')) == result.middle_json
        assert json.loads((folder / (source.stem + '_content_list.json')).read_text(encoding='utf8')) == result.content_list_json
        assert {p.name: p.read_bytes() for p in (folder / 'images').iterdir()} == result.images
        return result, dict(files=3 + len(result.images))
    check('disk', disk)
    for mode in ['nlp_markdown', 'content_list', 'content_list_v2']:
        check(mode, lambda mode=mode: (RapidDoc(make_md_mode=mode)(source), dict(mode=mode)))
    if source.suffix == '.doc':
        def mixed():
            paths = sorted(p for p in source.parent.iterdir() if p.suffix in {'.doc','.docx','.xls','.xlsx','.ppt','.pptx'} and not p.name.startswith('~$'))
            results = engine(paths, lang=['ch'] * len(paths))
            assert len(results) == len(paths)
            return results, dict(sources=[p.name for p in paths])
        check('mixed-office-batch', mixed)


def coordinator(args):
    from accept_pdf_delivery import differences
    from accept_doc_performance import installed_binary
    import hashlib
    sources = list((ROOT / 'demo/ms_office_docs').glob('*'))
    sources = sorted(p for p in sources if p.suffix.lower() in {'.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'} and not p.name.startswith('~$'))
    sources += [ROOT / 'demo/pdfs' / (name + '.pdf') for name in ['small_ocr', '示例1-论文模板', '含印章']]
    if args.formats:
        sources = [p for p in sources if p.suffix[1:] in args.formats]
    assert sources
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / 'manifest.json', dict(installed_rust=installed_binary(), public_signatures=public_signatures(), cases=[
            dict(source=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]))
    else:
        manifest = json.loads((args.output / 'manifest.json').read_text(encoding='utf8'))
        assert manifest['installed_rust'] == installed_binary(), 'Installed binary changed; use a new output directory'
        expected = {c['source']: c['sha256'] for c in manifest['cases']}
        assert set(expected) == {str(p.relative_to(ROOT)) for p in sources}, 'Resume must retain the original case selection'
        assert all(expected[str(p.relative_to(ROOT))] == hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)
    rows = []
    for index, source in enumerate(sources):
        case = source.stem + '-' + source.suffix[1:]
        for engine in (['python', 'rust'] if index % 2 == 0 else ['rust', 'python']):
            destination = args.output / case / engine
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not (args.resume and (destination / 'report.json').exists()):
                print(f'{case}: {engine}', flush=True)
                with (destination.parent / (engine + '.log')).open('w', encoding='utf8') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--engine', engine,
                        '--source', str(source), '--output', str(destination), *(['--errors-only'] if args.errors_only else [])], cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, timeout=900, check=True)
        py, ru = [json.loads((args.output / case / e / 'report.json').read_text(encoding='utf8')) for e in ['python', 'rust']]
        assert [r['check'] for r in py['checks']] == [r['check'] for r in ru['checks']], case
        assert py['defaults'] == ru['defaults'], (case, py['defaults'], ru['defaults'])
        for a, b in zip(py['checks'], ru['checks']):
            assert a['check'] == b['check']
            row = dict(case=case, check=a['check'], python_success=a['success'], rust_success=b['success'])
            if a['success'] and b['success']:
                snapshots = [json.loads((args.output / case / e / (a['check'] + '.json')).read_text(encoding='utf8')) for e in ['python', 'rust']]
                diffs = list(differences(*snapshots))
                row.update(exact=not diffs, differences=len(diffs), examples=diffs[:12])
            else:
                row.update(exact=False, python_error=a.get('error'), rust_error=b.get('error'))
            rows.append(row)
        dump(args.output / 'summary.json', dict(checks=rows, all_passed=all(r['exact'] for r in rows)))
    if not all(r['exact'] for r in rows):
        raise SystemExit('Public acceptance found differences; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--engine', choices=['python', 'rust'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--formats', nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--errors-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    return worker(args) if args.worker else coordinator(args)


if __name__ == '__main__':
    main()
