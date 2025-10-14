# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import time
from pathlib import Path

# # 使用默认 GPU（cuda:0）
# os.environ['MINERU_DEVICE_MODE'] = "cuda"
# # 或指定 GPU 编号，例如使用第二块 GPU（cuda:1）
# os.environ['MINERU_DEVICE_MODE'] = "cuda:1"

from loguru import logger

from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
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
    layout_config = {
        # "model_type": LayoutModelType.PP_DOCLAYOUT_PLUS_L,
        # "conf_thresh": 0.4,
        # "batch_num": 1,
        # "model_dir_or_path": "C:\ocr\models\ppmodel\layout\PP-DocLayout-L\pp_doclayout_l.onnx",
        # "model_dir_or_path": r"C:\ocr\models\ppmodel\layout\PP-DocLayout_plus-L\pp_doclayout_plus_l.onnx",
    }

    ocr_config = {
        # "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.onnx",
        # "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.onnx",
        # "Rec.rec_batch_num": 1,

        # "Det.ocr_version": OCRVersion.PPOCRV5,
        # "Rec.ocr_version": OCRVersion.PPOCRV5,
        # "Det.model_type": ModelType.SERVER,
        # "Rec.model_type": ModelType.SERVER,

        # 新增的自定义参数
        # "engine_type": OCREngineType.TORCH, # 统一设置推理引擎
        # "Det.rec_batch_num": 1, # Det批处理大小

        # 是否使用ocr的Det定位文本行，默认False，直接使用pdf里的文本bbox，当parse_method="ocr"或parse_method="auto"自动判断为需要ocr时，use_det_bbox会自动变为True
        # "use_det_bbox": False
    }

    formula_config = {
        # "model_type": FormulaModelType.PP_FORMULANET_PLUS_S,
        # "formula_level": 1, # 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别
        # "batch_num": 1,
        # "model_dir_or_path": r"C:\ocr\models\ppmodel\formula\PP-FormulaNet_plus-S\pp_formulanet_plus_s.onnx",
    }

    # os.environ['MINERU_MODEL_SOURCE'] = 'local'

    table_config = {
        # "force_ocr": False, # 表格文字，是否强制使用ocr，默认 False 根据 parse_method 来判断是否需要ocr还是从pdf中直接提取文本
        # "model_type": TableModelType.UNET_SLANET_PLUS,  # （默认） 有线表格使用unet，无线表格使用slanet_plus
        # "model_type": TableModelType.UNET_UNITABLE, # 有线表格使用unet，无线表格使用unitable
        # "model_type": TableModelType.SLANEXT,  # 有线表格使用slanext_wired，无线表格使用slanext_wireless

        # "model_dir_or_path": "", #单个模型使用。如SLANET_PLUS、UNITABLE

        # "cls.model_dir_or_path": "", # 表格分类模型地址

        # "unet.model_dir_or_path": "", # UNET表格模型地址

        # "unitable.model_dir_or_path": "", # UNITABLE表格模型地址
        # "slanet_plus.model_dir_or_path": "", # SLANET_PLUS表格模型地址

        # "wired_cell.model_dir_or_path": "", # 有线单元格模型地址，配置SLANEXT时使用
        # "wireless_cell.model_dir_or_path": "", # 无线单元格模型地址，配置SLANEXT时使用
        # "wired_table.model_dir_or_path": "", # 有线表结构模型地址，配置SLANEXT时使用
        # "wireless_table.model_dir_or_path": "", # 无线表结构模型地址，配置SLANEXT时使用,
    }

    checkbox_config = {
        # "checkbox_enable": False, # 是否识别复选框，默认不识别，基于opencv，有可能会误检
    }


    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes
    # 记录开始时间
    start_time = time.time()
    infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, parse_method=parse_method, formula_enable=p_formula_enable,table_enable=p_table_enable,
                                                                                                     layout_config=layout_config, ocr_config=ocr_config, formula_config=formula_config, table_config=table_config, checkbox_config=checkbox_config)

    for idx, model_list in enumerate(infer_results):

        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_dict= all_page_dicts[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable, p_formula_enable, ocr_config=ocr_config)
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
        method="auto",
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            parse_method=method,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(__dir__, "output")

    doc_path_list = [
        "D:\\file\\text-pdf\\示例1-论文模板.pdf",
    ]
    for doc_path in doc_path_list:
        start_time = time.time()
        parse_doc([doc_path], output_dir)
        print(f"总运行时间: {time.time() - start_time}秒")
