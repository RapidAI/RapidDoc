# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import time
from pathlib import Path
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
# os.environ['MINERU_TOOLS_CONFIG_JSON'] = os.path.abspath('magic.json') # 配置文件绝对路径
# C:\Users\huazhen\.cache\modelscope\hub\models\OpenDataLab\PDF-Extract-Kit-1.0
from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
# from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

from rapidocr import EngineType as OcrEngineType
from kitty_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType, EngineType
from kitty_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese) ['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    p_formula_enable=True,  # Enable formula parsing
    p_table_enable=True,  # Enable table parsing
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        layout_config = {
            "model_type": LayoutModelType.PP_DOCLAYOUT_L,
            "engine_type": EngineType.ONNXRUNTIME,
            "conf_thresh": 0.5,
            "model_dir_or_path": "C:\ocr\models\ppmodel\layout\PP-DocLayout-L\pp_doclayout_l.onnx"
        }

        ocr_config = {
            "Det.engine_type": OcrEngineType.OPENVINO,
            "Cls.engine_type": OcrEngineType.OPENVINO,
            "Rec.engine_type": OcrEngineType.OPENVINO,

            # "Cls.cls_batch_num": 1,
            # "Rec.rec_batch_num": 1,

            # "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.xml",
            # "Cls.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_ppocr_mobile_v2.0_cls_infer\openvino\ch_ppocr_mobile_v2.0_cls.xml",
            # "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.xml",
        }
        # 27
        # 50

        # 25
        # 42

        formula_config = {
            "model_type": FormulaModelType.PP_FORMULANET_PLUS_L,
            "conf_thresh": 0.5,
            "formula_level": 1, # 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别
            "model_dir_or_path": r"C:\ocr\models\ppmodel\formula\PP-FormulaNet_plus-L\pp_formulanet_plus_l.onnx"
        }

        # os.environ['MINERU_MODEL_SOURCE'] = 'local'


        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes
        # 记录开始时间
        start_time = time.time()
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=p_formula_enable,table_enable=p_table_enable)

        for idx, model_list in enumerate(infer_results):

            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, p_formula_enable)
            # 计算总运行时间（单位：秒）
            print(f"运行时间: {time.time() - start_time}秒")
            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")

def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # args Table Predict: 100%|██████████| 9/9 [00:37<00:00,  4.14s/it]
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output_plus_mineru")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = [
        "D:\\file\\text-pdf\\示例1-论文模板.pdf",
        "D:\\file\\text-pdf\\示例1-论文模板.pdf",
        # "D:\\file\\text-pdf\\示例7-研究报告.pdf",
        # "D:\\file\\text-pdf\\示例7-研究报告.pdf"
        # "D:\\file\\text-pdf\\demo1.pdf",
        # "C:\ocr\img\pageddd_5.png",
        # "C:\ocr\img\pages_50_02.png",
        # "C:\ocr\img\pages_50_04.png",
        # "C:\ocr\img\pages_50_49.png",
        "D:\\file\\text-pdf\\demo1.pdf",
        "C:\\ocr\\pdf\\pages_50.pdf"
        # Formula Predict: 100%|██████████| 192/192 [00:45<00:00,  4.22it/s]
    ]
    # """Use pipeline mode if your environment does not support VLM"""
    # # 记录开始时间
    for doc_path in doc_path_list:
        start_time = time.time()
        parse_doc([doc_path], output_dir, backend="pipeline")
        # 计算总运行时间（单位：秒）
        print(f"总运行时间: {time.time() - start_time}秒")
