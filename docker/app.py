import gc
import json
import os
import traceback
import tempfile, shutil
from glob import glob
from base64 import b64encode
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from file_converter import ensure_pdf, OFFICE_EXTENSIONS
from rapid_doc.cli.common import aio_do_parse
from rapid_doc.utils.language import remove_invalid_surrogates
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

# Removed clean_data_for_json function - now using official API

# Removed init_writers function - now using simpler file handling


# Removed old processing functions - now using official API's aio_do_parse


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, file_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果 - 支持嵌套目录查找"""
    import glob
    
    # 首先尝试官方路径格式
    result_file_path = os.path.join(parse_dir, f"{file_name}{file_suffix_identifier}")
    logger.info(f"Looking for result file: {result_file_path}")
    
    if os.path.exists(result_file_path):
        logger.info(f"Found result file: {result_file_path}")
        try:
            with open(result_file_path, "r", encoding="utf-8") as fp:
                content = fp.read()
                logger.info(f"Read {len(content)} characters from {result_file_path}")
                return content
        except Exception as e:
            logger.error(f"Error reading file {result_file_path}: {e}")
            return None
    
    # 如果官方路径不存在，尝试递归查找
    logger.warning(f"Result file not found at official path: {result_file_path}")
    logger.info("Searching recursively in subdirectories...")
    
    # 构建文件名模式
    file_pattern = f"*{file_suffix_identifier}"
    logger.info(f"Searching for pattern: {file_pattern}")
    
    # 递归查找文件
    found_files = []
    for root, dirs, files in os.walk(parse_dir):
        for file in files:
            if file.endswith(file_suffix_identifier):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
                logger.info(f"Found potential result file: {full_path}")
    
    if found_files:
        # 优先选择包含文件名的文件
        for file_path in found_files:
            if file_name in file_path:
                logger.info(f"Selected result file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        logger.info(f"Read {len(content)} characters from {file_path}")
                        return content
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
        
        # 如果没有找到包含文件名的文件，使用第一个找到的文件
        if found_files:
            selected_file = found_files[0]
            logger.info(f"Using first found file: {selected_file}")
            try:
                with open(selected_file, "r", encoding="utf-8") as fp:
                    content = fp.read()
                    logger.info(f"Read {len(content)} characters from {selected_file}")
                    return content
            except Exception as e:
                logger.error(f"Error reading file {selected_file}: {e}")
                return None
    
    # 如果仍然没有找到，列出目录内容帮助调试
    logger.warning(f"No result file found for pattern: {file_pattern}")
    if os.path.exists(parse_dir):
        logger.info(f"Listing all files in {parse_dir}:")
        for root, dirs, files in os.walk(parse_dir):
            for file in files:
                logger.info(f"  {os.path.join(root, file)}")
    else:
        logger.warning(f"Parse directory does not exist: {parse_dir}")
    
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
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
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
    clear_output_file: bool = Form(False),
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
        # 验证后端类型
        supported_backends = ["pipeline"]
        if backend not in supported_backends:
            return JSONResponse(
                content={"error": f"Unsupported backend: {backend}. Supported: {supported_backends}"},
                status_code=400,
            )

        # 验证文件
        if not files:
            return JSONResponse(
                content={"error": "No files provided"},
                status_code=400,
            )

        # 把 JSON 字符串转成 dict，默认空 dict
        layout_config = _convert_value_to_enum(json.loads(layout_config or "{}"))
        ocr_config = _convert_value_to_enum(json.loads(ocr_config or "{}"))
        formula_config = _convert_value_to_enum(json.loads(formula_config or "{}"))
        table_config = _convert_value_to_enum(json.loads(table_config or "{}"))
        checkbox_config = json.loads(checkbox_config or "{}")
        image_config = json.loads(image_config or "{}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
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

            else:
                # 创建临时目录用于文档转换
                temp_dir = tempfile.mkdtemp(prefix="fastapi_adapter_")
                print(temp_dir)
                # 调用 ensure_pdf 进行文档转换
                # 保存上传的文件到临时目录
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
            # 读取文件内容
            file_name = Path(file.filename).stem  # 去掉扩展名
            # 修正：使用完整文件名匹配RapidDoc的输出文件命名格式
            file_basename = file.filename  # 保留完整文件名

            
            # 使用官方 API 的 aio_do_parse 函数
            try:
                logger.info(f"Starting to parse {file.filename} using backend {backend}")
                logger.info(f"Output directory: {output_dir}")
                logger.info(f"File name stem: {file_name}")
                
                await aio_do_parse(
                    output_dir=output_dir,
                    pdf_file_names=[file.filename],
                    pdf_bytes_list=[content],
                    p_lang_list=lang_list,
                    backend=backend,
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    layout_config=layout_config, ocr_config=ocr_config, formula_config=formula_config,
                    table_config=table_config, checkbox_config=checkbox_config, image_config=image_config,
                )
                
                logger.info(f"Parse completed for {file.filename}")
                # 列出输出目录中的所有文件
                if os.path.exists(output_dir):
                    all_files = []
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            all_files.append(os.path.join(root, f))
                    logger.info(f"Files created in output directory: {all_files}")
                
                # 收集结果
                file_result = {"filename": file.filename}
                logger.info(f"Collecting results for {file.filename}")
                
                if return_md:
                    md_content = get_infer_result(".md", file_basename, output_dir)
                    file_result["md_content"] = md_content
                
                if return_middle_json:
                    middle_json_content = get_infer_result("_middle.json", file_basename, output_dir)
                    if middle_json_content:
                        file_result["middle_json"] = json.loads(middle_json_content)
                
                if return_model_output:
                    model_json_content = get_infer_result("_model.json", file_basename, output_dir)
                    if model_json_content:
                        file_result["model_output"] = json.loads(model_json_content)
                
                if return_content_list:
                    content_list_content = get_infer_result("_content_list.json", file_basename, output_dir)
                    if content_list_content:
                        file_result["content_list"] = json.loads(content_list_content)
                
                if return_images:
                    # 在输出目录中查找图像文件 - 支持多种目录结构
                    image_dirs = [
                        os.path.join(output_dir, file_basename, "images"),  # 原始路径
                        os.path.join(output_dir, file_basename, "auto", "images"),  # auto 子目录
                    ]
                    
                    found_images = {}
                    for image_dir in image_dirs:
                        if os.path.exists(image_dir):
                            logger.info(f"Found image directory: {image_dir}")
                            # 支持多种图片格式
                            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]:
                                image_paths = glob(os.path.join(image_dir, ext))
                                for image_path in image_paths:
                                    image_name = os.path.basename(image_path)
                                    if image_name not in found_images:  # 避免重复
                                        found_images[image_name] = f"data:image/jpeg;base64,{encode_image(image_path)}"
                                        logger.info(f"Added image: {image_name}")
                    
                    file_result["images"] = found_images
                    logger.info(f"Total images found: {len(found_images)}")
                
                file_result["backend"] = backend
                results.append(file_result)
                
            except Exception as parse_error:
                # logger.error(f"Error parsing {file.filename}: {str(parse_error)}")
                tb_str = traceback.format_exc()  # 获取完整 traceback 信息
                logger.error(f"Error parsing {file.filename}:\n{tb_str}")

                results.append({
                    "filename": file.filename,
                    "error": str(parse_error)
                })
            
            # 是否清理文件
            if clear_output_file:
                shutil.rmtree(os.path.join(output_dir, file_basename), ignore_errors=True)

        # 返回结果
        response_data = {
            "results": results,
            "total_files": len(files),
            "successful_files": len([r for r in results if "error" not in r])
        }
        
        return JSONResponse(response_data, status_code=200)

    except Exception as e:
        error_message = remove_invalid_surrogates(str(e))
        logger.error(f"API error: {error_message}")
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


if __name__ == "__main__":
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    uvicorn.run(app, host="0.0.0.0", port=8888)
