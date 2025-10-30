# Copyright (c) Opendatalab. All rights reserved.
import os
import uuid
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from loguru import logger
from PIL import Image

from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.pdf_reader import image_to_b64str, image_to_bytes, page_to_image
from . import PyPDFium2Parser
from .boxbase import calculate_iou
from .enum_class import ImageType, CategoryId, ContentType
from .hash_utils import str_sha256, bytes_md5


def pdf_page_to_image(page: pdfium.PdfPage, dpi=200, image_type=ImageType.PIL) -> dict:
    """Convert pdfium.PdfDocument to image, Then convert the image to base64.

    Args:
        page (_type_): pdfium.PdfPage
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.
        image_type (ImageType, optional): The type of image to return. Defaults to ImageType.PIL.

    Returns:
        dict:  {'img_base64': str, 'img_pil': pil_img, 'scale': float }
    """
    pil_img, scale = page_to_image(page, dpi=dpi)
    image_dict = {
        "scale": scale,
    }
    if image_type == ImageType.BASE64:
        image_dict["img_base64"] = image_to_b64str(pil_img)
    else:
        image_dict["img_pil"] = pil_img

    return image_dict


def load_images_from_pdf(
    pdf_bytes: bytes,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,  # PIL or BASE64
):
    images_list = []
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    pdf_page_num = len(pdf_doc)
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if end_page_id > pdf_page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = pdf_page_num - 1

    for index in range(0, pdf_page_num):
        if start_page_id <= index <= end_page_id:
            page = pdf_doc[index]
            image_dict = pdf_page_to_image(page, dpi=dpi, image_type=image_type)
            images_list.append(image_dict)

    return images_list, pdf_doc


def cut_image(span, ori_image_list, extract_original_image, extract_original_image_iou_thresh, page_num: int, page_pil_img, return_path, image_writer: FileBasedDataWriter, scale=2):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""
    bbox = span['bbox']
    crop_img = None
    if extract_original_image and span['type'] in [ContentType.IMAGE]:
        # 判断是否可以提取原始图片
        for ori_image in ori_image_list:
            if calculate_iou(bbox, ori_image['bbox']) >= extract_original_image_iou_thresh:
                crop_img = ori_image['pil_image']

    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 老版本返回不带bucket的路径
    img_path = f"{return_path}_{filename}" if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f"{str_sha256(img_path)}.jpg"
    # img_hash256_path = f'{img_path}.jpg'
    if not crop_img:
        crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    image_writer.write(img_hash256_path, img_bytes)
    return img_hash256_path


def get_crop_img(bbox: tuple, pil_img, scale=2):
    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )
    return pil_img.crop(scale_bbox)


def get_crop_np_img(bbox: tuple, input_img, scale=2):

    if isinstance(input_img, Image.Image):
        np_img = np.asarray(input_img)
    elif isinstance(input_img, np.ndarray):
        np_img = input_img
    else:
        raise ValueError("Input must be a pillow object or a numpy array.")

    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )

    return np_img[scale_bbox[1]:scale_bbox[3], scale_bbox[0]:scale_bbox[2]]

def images_bytes_to_pdf_bytes(image_bytes):
    # 内存缓冲区
    pdf_buffer = BytesIO()

    # 载入并转换所有图像为 RGB 模式
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 第一张图保存为 PDF，其余追加
    image.save(pdf_buffer, format="PDF", save_all=True)

    # 获取 PDF bytes 并重置指针（可选）
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes

def get_ori_image(
        page: pdfium.PdfPage,
        max_depth: int = 15,
        render: bool = False,
        scale_to_original: bool = True,
) -> list:
    """
    从 PDF 中提取所有原始图片

    参数:
        max_depth: 搜索嵌套图像的最大层数
        render: 是否渲染图像（考虑 mask / 变换矩阵）
        scale_to_original: 渲染时是否缩放到原始分辨率
    """
    with PyPDFium2Parser.lock:
        images = list(
            page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=max_depth)
        )
    images_list = []
    for image in images:
        # === 获取 bbox ===
        bbox, pil_image = None, None
        try:
            # PDF页面坐标系 左下角原点坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = image.get_pos()
            page_width, page_height = page.get_size()

            # 转换为左上角原点坐标
            new_x1 = x1
            new_x2 = x2
            new_y1 = page_height - y2
            new_y2 = page_height - y1

            bbox = [new_x1, new_y1, new_x2, new_y2]
        except Exception:
            pass
        try:
            # ❶ 检查是否支持 scale_to_original 参数
            from inspect import signature
            sig = signature(image.get_bitmap)
            kwargs = {}
            if "render" in sig.parameters:
                kwargs["render"] = render
            if "scale_to_original" in sig.parameters:
                kwargs["scale_to_original"] = scale_to_original
            pil_image = image.get_bitmap(**kwargs).to_pil()
        except Exception:
            pass
        finally:
            image.close()
        if bbox and pil_image:
            images_list.append({
                "uuid": str(uuid.uuid4()),
                "bbox": bbox,
                "pil_image": pil_image,
            })
    return images_list

def save_table_fill_image(layout_dets: list[dict], table_fill_image_list: list[dict], page_img_md5, page_num, image_writer: FileBasedDataWriter):
    if not table_fill_image_list:
        return
    """保存表格里的图片，图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""

    def return_path(path_type):
        return f"{path_type}/{page_img_md5}"

    try:
        for layout_det in layout_dets:
            if layout_det['category_id'] != CategoryId.TableBody or not layout_det.get('html'):
                continue
            for fill_image in table_fill_image_list:
                if fill_image['uuid'] not in layout_det['html']:
                    continue
                bbox = fill_image['bbox']
                pil_image = fill_image['pil_image']
                # 拼接文件名
                filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

                # 老版本返回不带bucket的路径
                img_path = f"{return_path}_{filename}" if return_path is not None else None

                # 新版本生成平铺路径
                img_hash256_path = f"{str_sha256(img_path)}.jpg"
                # img_hash256_path = f'{img_path}.jpg'
                img_bytes = image_to_bytes(pil_image, image_format="JPEG")
                image_writer.write(img_hash256_path, img_bytes)

                image_dir = str(os.path.basename(image_writer._parent_dir))
                image_path = f"{image_dir}/{img_hash256_path}"
                format_image = '<img src="{}" alt="Image" />'.format(image_path)

                layout_det['html'] = layout_det['html'].replace(fill_image['uuid'], format_image)
    except Exception as e:
        logger.exception(e)
