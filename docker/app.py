import os
import gc
import json
import re
import uuid
import zipfile
import tempfile, shutil
import glob
from base64 import b64encode
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from loguru import logger

from file_converter import ensure_pdf, OFFICE_EXTENSIONS
from rapid_doc.cli.common import aio_do_parse
from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from rapid_doc.version import __version__

app = FastAPI(
    title="RapidDoc Web API",
    description="Compatible with RapidDoc official API - Parse documents using RapidDoc",
    version=__version__
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": __version__,
        "api": "RapidDoc Web API",
        "compatible": "Official RapidDoc API"
    }

# 支持的文件扩展名 - 与官方 API 保持一致
pdf_suffixes = [".pdf"]
# office_suffixes = [".ppt", ".pptx", ".doc", ".docx"]
office_suffixes = list(OFFICE_EXTENSIONS)
image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]


def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._-
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'

def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None

def _convert_value_to_enum(config):
    """
      自动将配置 dict 中的字符串枚举（如 "OCRVersion.PPOCRV5"）转换为实际枚举对象。
      """
    # ⚡ 如果 config 是空字典，直接返回
    if not config:
        return config
    from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType, LangDet, LangRec
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType, EngineType as FormulaEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    # 可识别的枚举类映射表（可扩展）
    enum_map = {
        "OCREngineType": OCREngineType,
        "OCRVersion": OCRVersion,
        "OCRModelType": OCRModelType,
        "LangDet": LangDet,
        "LangRec": LangRec,
        "LayoutModelType": LayoutModelType,
        "FormulaModelType": FormulaModelType,
        "FormulaEngineType": FormulaEngineType,
        "TableModelType": TableModelType,
    }

    def resolve_enum_value(value):
        """递归解析单个值"""
        if isinstance(value, str) and "." in value:
            prefix, member = value.split(".", 1)
            if prefix in enum_map and hasattr(enum_map[prefix], member):
                return getattr(enum_map[prefix], member)
        elif isinstance(value, dict):
            return {k: resolve_enum_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_enum_value(v) for v in value]
        return value

    return {k: resolve_enum_value(v) for k, v in config.items()}


@app.post(
    "/file_parse",
    tags=["projects"],
    summary="Parse files using RapidDoc - Compatible with official API",
)
async def file_parse(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output"),
    clear_output_file: bool = Form(True),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    layout_config: Optional[str] = Form("{}"),
    ocr_config: Optional[str] = Form("{}"),
    formula_config: Optional[str] = Form("{}"),
    table_config: Optional[str] = Form("{}"),
    checkbox_config: Optional[str] = Form("{}"),
    image_config: Optional[str] = Form("{}"),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(False),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
    response_format_zip: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
):
    """
    Parse files using RapidDoc - Compatible with official API
    
    Args:
        files: List of files to be parsed (PDF, images)
        output_dir: Output directory for results
        lang_list: List of languages for parsing (e.g., ['ch', 'en'])
        backend: Parsing backend (pipeline)
        parse_method: Parsing method (auto, ocr, txt)
        formula_enable: Whether to enable formula parsing
        table_enable: Whether to enable table parsing
        return_md: Whether to return markdown content
        return_middle_json: Whether to return middle JSON
        return_model_output: Whether to return model output
        return_content_list: Whether to return content list
        return_images: Whether to return images as base64
        start_page_id: Start page ID for parsing
        end_page_id: End page ID for parsing
    """
    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        logger.info(f"Created unique output directory: {unique_dir}")

        # 把 JSON 字符串转成 dict，默认空 dict
        layout_config = _convert_value_to_enum(json.loads(layout_config or "{}"))
        ocr_config = _convert_value_to_enum(json.loads(ocr_config or "{}"))
        formula_config = _convert_value_to_enum(json.loads(formula_config or "{}"))
        table_config = _convert_value_to_enum(json.loads(table_config or "{}"))
        checkbox_config = json.loads(checkbox_config or "{}")
        image_config = json.loads(image_config or "{}")

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []
        # 读取文件内容
        for file in files:
            # 验证文件扩展名
            file_suffix = Path(file.filename).suffix.lower()
            if file_suffix not in pdf_suffixes + office_suffixes + image_suffixes:
                return JSONResponse(
                    content={"error": f"File type {file_suffix} is not supported for {file.filename}"},
                    status_code=400,
                )
            if file_suffix in pdf_suffixes + image_suffixes:
                content = await file.read()
                if file_suffix in image_suffixes:
                    content = images_bytes_to_pdf_bytes(content)
            else:
                # 创建临时目录用于文档转换
                temp_dir = tempfile.mkdtemp(prefix="fastapi_adapter_")
                # 调用 ensure_pdf 进行文档转换，保存上传的文件到临时目录
                temp_file_path = os.path.join(temp_dir, file.filename)
                with open(temp_file_path, "wb") as tmp_f:
                    tmp_f.write(await file.read())
                pdf_to_process, temp_pdf_to_delete = ensure_pdf(temp_file_path, temp_dir)
                if not pdf_to_process:
                    return JSONResponse(
                        content={"error": f"Failed to convert Office/document to PDF: {file.filename}"},
                        status_code=500,
                    )
                # 从磁盘读字节
                with open(pdf_to_process, "rb") as f:
                    content = f.read()
                shutil.rmtree(temp_dir, ignore_errors=True)
            file_name = Path(file.filename).stem  # 去掉扩展名
            pdf_bytes_list.append(content)
            pdf_file_names.append(file_name)

        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 调用异步处理函数
        await aio_do_parse(
                output_dir=unique_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=actual_lang_list,
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=return_md,
                f_dump_middle_json=return_middle_json,
                f_dump_model_output=return_model_output,
                f_dump_orig_pdf=False,
                f_dump_content_list=return_content_list,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                layout_config = layout_config, ocr_config = ocr_config, formula_config = formula_config,
                table_config = table_config, checkbox_config = checkbox_config, image_config = image_config,
            )

        # 根据 response_format_zip 决定返回类型
        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="rapid_doc_results_")
            os.close(zip_fd)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_name in pdf_file_names:
                    safe_pdf_name = sanitize_filename(pdf_name)
                    if backend.startswith("pipeline"):
                        parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    else:
                        parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                    if not os.path.exists(parse_dir):
                        continue

                    # 写入文本类结果
                    if return_md:
                        path = os.path.join(parse_dir, f"{pdf_name}.md")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))

                    if return_middle_json:
                        path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_middle.json"))

                    if return_model_output:
                        path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, os.path.basename(path)))

                    if return_content_list:
                        path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_content_list.json"))

                    # 写入图片
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                        for image_path in image_paths:
                            zf.write(image_path,
                                     arcname=os.path.join(safe_pdf_name, "images", os.path.basename(image_path)))
            # 是否清理文件
            if clear_output_file:
                shutil.rmtree(unique_dir, ignore_errors=True)
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="results.zip",
                background=BackgroundTask(cleanup_file, zip_path)
            )
        else:
            # 构建 JSON 结果
            result_dict = {}
            for pdf_name in pdf_file_names:
                result_dict[pdf_name] = {}
                data = result_dict[pdf_name]

                if backend.startswith("pipeline"):
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                else:
                    parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                if os.path.exists(parse_dir):
                    if return_md:
                        data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                    if return_middle_json:
                        data["middle_json"] = json.loads(get_infer_result("_middle.json", pdf_name, parse_dir))
                    if return_model_output:
                        data["model_output"] = json.loads(get_infer_result("_model.json", pdf_name, parse_dir))
                    if return_content_list:
                        data["content_list"] = json.loads(get_infer_result("_content_list.json", pdf_name, parse_dir))
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                        image_paths = glob.glob(safe_pattern)
                        data["images"] = {
                            os.path.basename(
                                image_path
                            ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                            for image_path in image_paths
                        }
            # 是否清理文件
            if clear_output_file:
                shutil.rmtree(unique_dir, ignore_errors=True)
            return JSONResponse(
                status_code=200,
                content={
                    "backend": backend,
                    "version": __version__,
                    "results": result_dict
                }
            )

    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )
    finally:
        gc.collect()


if __name__ == "__main__":
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    uvicorn.run(app, host="0.0.0.0", port=8888)
