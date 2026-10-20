# Copyright (c) Opendatalab. All rights reserved.
import time
from io import BytesIO

from loguru import logger
from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes, reject_encrypted_ooxml

def office_analyze(
        file_bytes,
        image_writer=None,
        file_type=None,
):
    infer_start = time.time()
    reject_encrypted_ooxml(file_bytes)
    file_type = str(file_type).lower().lstrip(".") if file_type else guess_suffix_by_bytes(file_bytes)
    file_stream = BytesIO(file_bytes)

    from rapid_doc._vendor.docvortex_core.analyzers.native import (
        DocModel, DocxModel, XlsModel, XlsxModel, PptModel, PptxModel,
    )
    models = {
        "doc": DocModel, "docx": DocxModel,
        "xls": XlsModel, "xlsx": XlsxModel, "xlsm": XlsxModel,
        "ppt": PptModel, "pptx": PptxModel,
    }
    if file_type not in models:
        raise ValueError(f"Unsupported or unknown office file type: {file_type}")
    results = models[file_type]().predict(file_stream)

    infer_time = round(time.time() - infer_start, 2)
    safe_time = max(infer_time, 0.01)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / safe_time, 3)} page/s")

    middle_json = result_to_middle_json(
        results,
        image_writer,
    )

    return middle_json, results

if __name__ == '__main__':
    # Resolve a default docx file relative to this script so the example
    # works no matter what the current working directory is when the
    # module is executed.  Allow the user to override the path via a
    # command-line argument for even greater flexibility.
    from pathlib import Path
    import argparse

    script_root = Path(__file__).resolve().parent.parent.parent.parent
    default_docx = script_root / "demo" / "docx" / "demo1.docx"

    parser = argparse.ArgumentParser(
        description="Quick demo runner for office_analyze"
    )
    parser.add_argument(
        "docx",
        nargs="?",
        default=str(default_docx),
        help="path to docx file (defaults to demo/docx/demo1.docx relative to project root)"
    )
    parser.add_argument(
        "--output-images",
        help="directory to write image outputs",
        default="./output_images"
    )
    args = parser.parse_args()

    docx_path = Path(args.docx)
    from rapid_doc.data.data_reader_writer import FileBasedDataWriter

    with open(docx_path, 'rb') as f:
        file_bytes = f.read()
    image_writer = FileBasedDataWriter(args.output_images)
    middle_json, results = office_analyze(
        file_bytes,
        image_writer=image_writer,
    )

    import json
    logger.info(json.dumps(middle_json, indent=2, ensure_ascii=False))
