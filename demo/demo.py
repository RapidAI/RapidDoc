import copy
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# ============== 设备配置 ==============
# 使用默认 GPU（cuda:0）
# os.environ['MINERU_DEVICE_MODE'] = "cuda"
# # 或指定 GPU 编号，例如使用第二块 GPU（cuda:1）
# os.environ['MINERU_DEVICE_MODE'] = "cuda:1"
# os.environ['MINERU_LAYOUT_ORIGINAL_IMAGE'] = "true"
# 是否启用图片方向矫正，开启后，可以自动识别并矫正 90°、270°的图片
# os.environ['USE_DOC_ORIENTATION_CLASSIFY'] = "true"
# # 模型文件存储目录
# os.environ['RAPID_MODELS_DIR'] = r'D:\CodeProjects\doc\RapidAI\models' #模型文件存储目录，如果不设置会默认下载到rapid_doc项目里面
from loguru import logger

from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.utils.config_reader import get_processing_window_size
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path
from rapid_doc.utils.office_converter import convert_legacy_office_to_modern
from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn, office_suffixes, old_office_suffixes
from rapid_doc.backend.office.office_analyze import office_analyze
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make as office_union_make
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

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
    f_dump_md_html=False,  # Whether to convert markdown to HTML
    f_dump_md_docx=False,  # Whether to convert markdown to docx (via Pandoc)
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config = _build_config()
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
        f_dump_md_html=f_dump_md_html,
        f_dump_md_docx=f_dump_md_docx,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        if start_page_id !=0 or end_page_id is not None:
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes
    pdf_pages_batch = get_processing_window_size(default=64)

    _process_pipeline_docs_in_batches(
        output_dir=output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        parse_method=parse_method,
        p_formula_enable=p_formula_enable,
        p_table_enable=p_table_enable,
        f_draw_layout_bbox=f_draw_layout_bbox,
        f_draw_span_bbox=f_draw_span_bbox,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_pdf=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_dump_md_html=f_dump_md_html,
        f_dump_md_docx=f_dump_md_docx,
        f_make_md_mode=f_make_md_mode,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config=checkbox_config,
        image_config=image_config,
        pdf_pages_batch=pdf_pages_batch,
    )


def _process_pipeline_docs_in_batches(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_dump_md_html,
        f_dump_md_docx,
        f_make_md_mode,
        layout_config,
        ocr_config,
        formula_config,
        table_config,
        checkbox_config,
        image_config,
        pdf_pages_batch,
):
    local_image_dirs = []
    local_md_dirs = []
    image_writers = []
    md_writers = []
    for pdf_file_name in pdf_file_names:
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        local_image_dirs.append(local_image_dir)
        local_md_dirs.append(local_md_dir)
        image_writers.append(FileBasedDataWriter(local_image_dir))
        md_writers.append(FileBasedDataWriter(local_md_dir))

    tmp_start_page_id = 0
    batch_idx = 0
    middle_json_list = [None] * len(pdf_bytes_list)
    model_json_list = [[] if f_dump_model_output else None for _ in pdf_bytes_list]
    finished = [False] * len(pdf_bytes_list)

    while not all(finished):
        active_indexes = [idx for idx, is_finished in enumerate(finished) if not is_finished]
        active_pdf_bytes_list = [pdf_bytes_list[idx] for idx in active_indexes]
        infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list, file_end_list = pipeline_doc_analyze(active_pdf_bytes_list, parse_method=parse_method, formula_enable=p_formula_enable,table_enable=p_table_enable,
                                                                                                         layout_config=layout_config, ocr_config=ocr_config, formula_config=formula_config, table_config=table_config, checkbox_config=checkbox_config,
                                                                                                         start_page_id=tmp_start_page_id, end_page_id=None, pdf_pages_batch=pdf_pages_batch)

        for active_idx, model_list in enumerate(infer_results):
            original_idx = active_indexes[active_idx]
            if f_dump_model_output:
                model_json_list[original_idx].extend(copy.deepcopy(model_list))

            tmp_middle_json = pipeline_result_to_middle_json(model_list, all_image_lists[active_idx], all_page_dicts[active_idx], image_writers[original_idx], lang_list[active_idx]
                                                         , ocr_enabled_list[active_idx], p_formula_enable, ocr_config=ocr_config, image_config=image_config
                                                         , batch_idx=batch_idx, pdf_pages_batch = pdf_pages_batch)
            if middle_json_list[original_idx] is None:
                middle_json_list[original_idx] = tmp_middle_json
            else:
                middle_json_list[original_idx]["pdf_info"].extend(tmp_middle_json["pdf_info"])

            if file_end_list[active_idx]:
                pdf_file_name = pdf_file_names[original_idx]
                _process_output(
                    middle_json_list[original_idx]["pdf_info"], pdf_bytes_list[original_idx], pdf_file_name, local_md_dirs[original_idx], local_image_dirs[original_idx],
                    md_writers[original_idx], f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                    f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                    f_make_md_mode, middle_json_list[original_idx], model_json_list[original_idx], process_mode="pipeline", f_dump_md_html=f_dump_md_html,
                    f_dump_md_docx=f_dump_md_docx
                )
                finished[original_idx] = True
            elif not model_list:
                logger.warning(f"No pages parsed for {pdf_file_names[original_idx]}, stop batch processing.")
                finished[original_idx] = True

        tmp_start_page_id += pdf_pages_batch
        batch_idx += 1



def _build_config():
    from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType, EngineType as FormulaEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType, EngineType as TableEngineType
    from rapid_doc.model.custom.paddleocr_vl.paddleocr_vl import PaddleOCRVLTableModel, PaddleOCRVLOCRModel, PaddleOCRVLFormulaModel
    layout_config = {
        # "model_type": LayoutModelType.PP_DOCLAYOUTV2,
        # "conf_thresh": 0.4,
        # "batch_num": 1,
        # "model_dir_or_path": r"C:\ocr\models\ppmodel\layout\PP-DocLayoutV2\pp_doclayoutv2.onnx",
        # "markdown_ignore_labels": ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text",],
    }

    ocr_config = {
        # "custom_model": PaddleOCRVLOCRModel(),
        # "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.onnx",
        # "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.onnx",
        # "Rec.rec_batch_num": 1,

        # "Det.ocr_version": OCRVersion.PPOCRV5,
        # "Rec.ocr_version": OCRVersion.PPOCRV5,
        # "Det.model_type": OCRModelType.SERVER,
        # "Rec.model_type": OCRModelType.SERVER,

        # 新增的自定义参数
        # "engine_type": OCREngineType.TORCH, # 统一设置推理引擎
        # "Det.rec_batch_num": 8, # Det批处理大小

        # 文本检测框模式：auto（默认）、txt、ocr
        # "use_det_mode": 'auto' #（1、txt只会从pypdfium2获取文本框，保留pdf中的图片，2、ocr只会从OCR-det获取文本框，3、auto先从pypdfium2获取文本框，提取不到再使用OCR-det提取）
    }

    formula_config = {
        # "custom_model": PaddleOCRVLFormulaModel(),
        # "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
        # "engine_type": FormulaEngineType.TORCH,
        # "formula_level": 1, # 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别
        # "batch_num": 1,
        # "model_dir_or_path": r"C:\ocr\models\ppmodel\formula\PP-FormulaNet_plus-S\pp_formulanet_plus_s.onnx",
        # "dict_keys_path": "D:\CodeProjects\doc\RapidAI\model\pp_formulanet_plus_m_inference.yml", #yml字典路径（torch使用）
    }

    table_config = {
        # "custom_model": PaddleOCRVLTableModel(),
        # "force_ocr": False, # 表格文字，是否强制使用ocr，默认 False 根据 parse_method 来判断是否需要ocr还是从pdf中直接提取文本
        # 注：文字版pdf可以使用pypdfium2提取到表格内图片，扫描版或图片需要使用PP_DOCLAYOUT_PLUS_L/PP_DOCLAYOUTV2版面识别模型，才能识别到表格内的图片
        # "skip_text_in_image": True, # 是否跳过表格里图片中的文字（如表格单元格中嵌入的图片、图标、扫描底图等）
        # "use_img2table": False, # 是否优先使用img2table库提取表格，需要手动安装（pip install img2table），基于opencv识别准确度不如使用模型，但是速度很快，默认关闭

        # "model_type": TableModelType.SLANETPLUS,
        # "model_type": TableModelType.UNET_SLANET_PLUS,  # （默认） 有线表格使用unet，无线表格使用slanet_plus
        # "model_type": TableModelType.UNET_UNITABLE, # 有线表格使用unet，无线表格使用unitable
        # "model_type": TableModelType.UNITABLE,
        # "model_dir_or_path": "", #单个模型使用。如SLANET_PLUS、UNITABLE

        # "use_word_box": False, # 使用单字坐标匹配单元格，默认 False
        # "use_compare_table": False,  # 启用表格结果比较（同时跑有线/无线并比对），默认 False
        # "table_formula_enable": False, # 表格内公式识别
        # "table_image_enable": False, # 表格内图片识别
        # "extract_original_image": False # 是否提取表格内原始图片，默认 False
        # "cls.model_type": TableModelType.PADDLE_Q_CLS, # 表格分类模型
        # "cls.model_dir_or_path": "", # 表格分类模型地址
        # "unet.model_dir_or_path": "", # UNET表格模型地址
        # "unitable.model_dir_or_path": "", # UNITABLE表格模型地址
        # "slanet_plus.model_dir_or_path": "", # SLANET_PLUS表格模型地址

        # "engine_type": TableEngineType.ONNXRUNTIME,  # 统一设置推理引擎
    }

    checkbox_config = {
        # "checkbox_enable": True, # 是否识别复选框，默认不识别，基于opencv，有可能会误检
    }

    # 版面识别元素为图片的配置
    image_config = {
        # "extract_original_image": True, # 是否提取原始图片（使用 pypdfium2 提取原始图片。截图可能导致清晰度降低和边界丢失，默认关闭）
        # "extract_original_image_iou_thresh": 0.5, # 是否提取原始图片和版面识别的图片，bbox重叠度，默认0.9
    }
    return layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config

def _process_office_doc(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_file=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        f_dump_md_html=False,
        f_dump_md_docx=False,
):
    need_remove_index = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[i]
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix in office_suffixes:

            need_remove_index.append(i)

            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"office")
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = office_analyze(
                file_bytes,
                image_writer=image_writer,
            )

            f_draw_layout_bbox = False
            f_draw_span_bbox = False
            pdf_info = middle_json["pdf_info"]

            _process_output(
                pdf_info, file_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_file,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, infer_result, process_mode=file_suffix, f_dump_md_html=f_dump_md_html, f_dump_md_docx=f_dump_md_docx
            )

    return need_remove_index


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        process_mode="pipeline",
        f_dump_md_html=False,
        f_dump_md_docx=False,
):

    if process_mode == "pipeline":
        make_func = pipeline_union_make
    elif process_mode in office_suffixes:
        make_func = office_union_make
    else:
        raise Exception(f"Unknown process_mode: {process_mode}")

    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        if process_mode in ["pipeline", "vlm"]:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )
        elif process_mode in office_suffixes:
            md_writer.write(
                f"{pdf_file_name}_origin.{process_mode}",
                pdf_bytes,
            )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

        # ===================== Markdown 转 HTML =====================
        if f_dump_md_html and md_content_str:
            try:
                from rapid_doc.utils.markdown_to_html import markdown_to_html
                html_path = os.path.join(local_md_dir, f"{pdf_file_name}.html")
                markdown_to_html(
                    md_content_str,
                    output_path=html_path,
                    title=pdf_file_name,
                    image_base_path=local_md_dir,  # 图片相对于md目录
                    embed_images=False,  # 不嵌入图片，保持引用
                )
            except ImportError as e:
                logger.warning(f"Markdown转HTML失败: {e}")
            except Exception as e:
                logger.error(f"Markdown转HTML失败: {e}")

        # ===================== Markdown 转 docx (via Pandoc) =====================
        if f_dump_md_docx and md_content_str:
            try:
                from rapid_doc.utils.markdown_to_word import markdown_to_docx
                md_docx_path = os.path.join(local_md_dir, f"{pdf_file_name}_md.docx")
                markdown_to_docx(
                    md_content_str,
                    output_path=md_docx_path,
                    image_base_path=local_md_dir,  # 图片相对于md目录
                )
            except ImportError as e:
                logger.warning(f"Markdown转docx失败: {e}")
            except Exception as e:
                logger.error(f"Markdown转docx失败: {e}")

    if f_dump_content_list:
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if process_mode != "pipeline":
        content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list_v2.json",
            json.dumps(content_list_v2, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
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
            file_suffix = guess_suffix_by_path(path)
            if file_suffix in old_office_suffixes:
                path = convert_legacy_office_to_modern(path)
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

def collect_files(directory: Path, suffixes: set[str]) -> list[Path]:
    """收集目录下指定后缀的文件，递归查找。"""
    if not directory.exists() or not directory.is_dir():
        return []

    return [
        file_path
        for file_path in directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in suffixes
    ]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    dir_suffix_map = {
        base_dir / "pdfs": {".pdf"},
        base_dir / "docx": {".docx"},
        base_dir / "pptx": {".pptx"},
        base_dir / "xlsx": {".xlsx"},
        base_dir / "images": {".png", ".jpeg", ".jpg"},
    }

    doc_path_list = []
    for directory, suffixes in dir_suffix_map.items():
        doc_path_list.extend(collect_files(directory, suffixes))

    parse_doc(doc_path_list, output_dir)