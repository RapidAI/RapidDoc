import math

from loguru import logger

from .pdf_image_tools import cut_image


def cut_image_and_table(span, ori_image_list, extract_original_image, extract_original_image_iou_thresh, page_pil_img, page_img_md5, page_id, image_writer, scale=2):

    def return_path(path_type):
        return f"{path_type}/{page_img_md5}"

    span_type = span["type"]

    bbox = span["bbox"]
    context = f"page={page_id}, type={span_type}, bbox={bbox}"
    if not image_writer:
        span["image_path"] = ""
    elif not check_img_bbox(bbox):
        logger.warning(f"PDF visual asset has invalid crop geometry: {context}")
        span["image_path"] = ""
    else:
        try:
            span["image_path"] = cut_image(
                span, ori_image_list, extract_original_image, extract_original_image_iou_thresh,
                page_id, page_pil_img, return_path=return_path(span_type),
                image_writer=image_writer, scale=scale,
            )
        except Exception as exc:
            raise ValueError(f"PDF visual asset generation failed: {context}: {exc}") from exc

    return span


def check_img_bbox(bbox) -> bool:
    return (
        len(bbox) == 4
        and all(math.isfinite(value) for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )
