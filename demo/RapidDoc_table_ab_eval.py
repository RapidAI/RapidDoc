"""Run the agreed A/B/C table comparison on a fixed PDF subset, sequentially.

Each group gets a separate process and output directory. No changes to the
existing full-corpus end-to-end script or parser defaults.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GROUPS = {
    'A': dict(model_type='unet_slanet_plus', rule_table_engine='docvortex', fallback_model_type=None),
    'B': dict(model_type='unet_slanet_plus', rule_table_engine='docvortex', fallback_model_type='table_transformer'),
    'C': dict(model_type='unet_slanet_plus', rule_table_engine='docvortex_ocr', fallback_model_type='table_transformer'),
}


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', type=Path, required=True)
    parser.add_argument('--pdfs', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=GROUPS)
    args = parser.parse_args()
    names = [line.strip().replace('\\', '/').rsplit('/', 1)[-1]
             for line in args.list.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not names or len(names) != len(set(names)) or any(Path(n).suffix.lower() != '.pdf' for n in names):
        raise ValueError('Expected nonempty, unique PDF filenames')
    files = [args.pdfs / name for name in names]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    args.output.mkdir(parents=True, exist_ok=True)
    common = dict(formula_enable=True, table_enable=True, parse_method='auto',
                  doc_orientation_classify=False, fallback_to_text=False, fallback_text_loss_threshold=.2)
    manifest = dict(files=names, file_list_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                    common=common, groups=GROUPS)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Output directory belongs to a different comparison')
    dump(manifest_path, manifest)
    if args.group is None:
        results = []
        for group in GROUPS:
            with (args.output / f'{group}.console.log').open('a', encoding='utf-8') as stream:
                command = [sys.executable, str(Path(__file__).resolve()), '--list', str(args.list),
                           '--pdfs', str(args.pdfs), '--models', str(args.models), '--output', str(args.output),
                           '--group', group]
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                results.append(dict(group=group, exit_code=result.returncode))
            dump(args.output / 'status.json', dict(groups=results, complete=len(results) == len(GROUPS)))
        return int(any(r['exit_code'] for r in results))

    os.environ.update(MINERU_DEVICE_MODE='cuda', RAPID_MODELS_DIR=str(args.models),
                      USE_DOC_ORIENTATION_CLASSIFY='false', PYTHONUNBUFFERED='1')
    from loguru import logger
    from rapid_doc_rust import RapidDoc
    output = args.output / args.group
    output.mkdir(exist_ok=True)
    images = output / 'images'
    images.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level='INFO', backtrace=False, diagnose=False)
    logger.add(args.output / f'{args.group}.errors.log', level='ERROR', encoding='utf-8',
               backtrace=False, diagnose=False)
    config = dict(GROUPS[args.group], fallback_to_text=False, fallback_text_loss_threshold=.2)
    engine = RapidDoc(parse_method='auto', formula_enable=True, table_enable=True, table_config=config)
    logger.info('Group {}: files={}, formula=True, document_orientation=False, table={}', args.group, len(files), config)
    started = time.perf_counter()
    completed = skipped = failed = 0
    for index, source in enumerate(files, 1):
        target = output / (source.stem + '.md')
        if target.exists():
            skipped += 1
            continue
        begin = time.perf_counter()
        record = dict(file=source.name)
        logger.info('[{}/{}] {}', index, len(files), source.name)
        try:
            result = engine(source, image_dir_name='images', f_dump_middle_json=False, f_dump_content_list=False)
            for name, data in result.images.items():
                if Path(name).name != name or name in ('.', '..'):
                    raise ValueError('Invalid image asset path')
                asset = images / name
                if asset.exists() and asset.read_bytes() != data:
                    raise ValueError('Conflicting image asset')
                asset.write_bytes(data)
            temporary = target.with_suffix('.md.part')
            temporary.write_text(result.markdown, encoding='utf-8')
            temporary.replace(target)
            completed += 1
            record.update(ok=True, characters=len(result.markdown), assets=len(result.images))
        except Exception as error:
            failed += 1
            record.update(ok=False, error_type=type(error).__name__, error=str(error))
            logger.exception('Failed {}', source.name)
        record['seconds'] = time.perf_counter() - begin
        with (args.output / f'{args.group}.files.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        dump(args.output / f'{args.group}.summary.json', dict(group=args.group, total=len(files),
             completed=completed, skipped=skipped, failed=failed, seconds=time.perf_counter()-started))
    logger.info('Done {}: completed={}, skipped={}, failed={}, seconds={:.1f}',
                args.group, completed, skipped, failed, time.perf_counter()-started)
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
