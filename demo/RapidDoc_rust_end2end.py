"""手动运行生成 OmniDocBench 预测；评分沿用原流程，替换预测目录即可。"""
import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 修改这些路径后运行；输出目录与 Python 版本分开。
FILES_DIR = Path('/web/hzkitty/OmniDocBenchFiles/pdfs')
OUTPUT_DIR = Path('/web/hzkitty/OmniDocBenchFiles/rust_v3_unet_slanet_tatr_fallback_orientation_off_full_20261103')
MODELS_DIR = '/web/hzkitty/models'
DEVICE = 'cuda'  # 本机无 GPU 时改为 cpu。

os.environ['MINERU_DEVICE_MODE'] = DEVICE
os.environ['RAPID_MODELS_DIR'] = MODELS_DIR
os.environ['USE_DOC_ORIENTATION_CLASSIFY'] = 'false'

from loguru import logger
from rapid_doc_rust import RapidDoc


def main():
    files = sorted(p for p in FILES_DIR.iterdir() if p.is_file() and p.suffix.lower() == '.pdf')
    if not files:
        raise FileNotFoundError(f'No PDF files in {FILES_DIR}')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Duplicate PDF stems would overwrite predictions')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT_DIR / 'images'
    images_dir.mkdir(exist_ok=True)
    log_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + '_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    logger.remove()
    logger.add(sys.stderr, level='INFO', backtrace=False, diagnose=False)
    logger.add(log_dir / f'{run_id}.log', encoding='utf-8', level='DEBUG',
               enqueue=False, backtrace=False, diagnose=False)
    logger.add(log_dir / f'{run_id}.errors.log', encoding='utf-8', level='ERROR',
               enqueue=False, backtrace=False, diagnose=False)
    failures_path = log_dir / f'{run_id}.failures.jsonl'
    logger.info('Logs: {}', log_dir)
    logger.info('Document orientation classification: {}; output: {}',
                os.environ['USE_DOC_ORIENTATION_CLASSIFY'], OUTPUT_DIR)

    table_config = {
        "model_type": "unet_slanet_plus",
        "rule_table_engine": "docvortex",
        "fallback_model_type": "table_transformer",
        "fallback_to_text": False,
        "fallback_text_loss_threshold": 0.2,
    }
    engine = RapidDoc(parse_method='auto', formula_enable=True, table_config=table_config)
    logger.info('Table configuration: {}', table_config)
    started = time.perf_counter()
    completed = skipped = failed = 0
    for index, source in enumerate(files, 1):
        destination = OUTPUT_DIR / f'{source.stem}.md'
        if destination.exists():
            skipped += 1
            logger.info('[{}/{}] Skip {}', index, len(files), source.name)
            continue
        logger.info('[{}/{}] Parse {}', index, len(files), source.name)
        try:
            result = engine(source, image_dir_name='images',
                            f_dump_middle_json=False, f_dump_content_list=False)
            for name, data in result.images.items():
                if Path(name).name != name or name in ('.', '..'):
                    raise ValueError(f'Invalid image name: {name}')
                target = images_dir / name
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'Conflicting image asset: {name}')
                target.write_bytes(data)
            # Markdown 最后原子写入；中断后重跑只跳过已完成的文件。
            temporary = destination.with_suffix('.md.part')
            temporary.write_text(result.markdown, encoding='utf-8')
            temporary.replace(destination)
            completed += 1
        except Exception as error:
            failed += 1
            with failures_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(file=str(source), error_type=type(error).__name__,
                    error=str(error)), ensure_ascii=False) + '\n')
            logger.exception('Failed: {}', source.name)
    logger.info('Done: completed={}, skipped={}, failed={}, seconds={:.1f}',
                completed, skipped, failed, time.perf_counter() - started)
    logger.complete()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
