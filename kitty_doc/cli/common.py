# Copyright (c) Opendatalab. All rights reserved.
import io
import os
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger

from kitty_doc.utils import PyPDFium2Parser
from kitty_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes

pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg", ".webp", ".gif"]


def read_fn(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if path.suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif path.suffix in pdf_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {path.suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    with PyPDFium2Parser.lock:
        # 从字节数据加载PDF
        pdf = pdfium.PdfDocument(pdf_bytes)

        # 确定结束页
        end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf) - 1
        if end_page_id > len(pdf) - 1:
            logger.warning("end_page_id is out of range, use pdf_docs length")
            end_page_id = len(pdf) - 1

        # 创建一个新的PDF文档
        output_pdf = pdfium.PdfDocument.new()

        # 选择要导入的页面索引
        page_indices = list(range(start_page_id, end_page_id + 1))

        # 从原PDF导入页面到新PDF
        output_pdf.import_pages(pdf, page_indices)

        # 将新PDF保存到内存缓冲区
        output_buffer = io.BytesIO()
        output_pdf.save(output_buffer)

        # 获取字节数据
        output_bytes = output_buffer.getvalue()

        pdf.close()  # 关闭原PDF文档以释放资源
        output_pdf.close()  # 关闭新PDF文档以释放资源

    return output_bytes
