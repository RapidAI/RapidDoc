# Copyright (c) RapidAI. All rights reserved.
"""
VL OCR 模块 - 支持 PaddleOCR-VL 模型
"""
import os
import threading

from loguru import logger
from tqdm import tqdm

from rapid_doc.model.custom import CustomBaseModel
from rapid_doc.model.custom.paddleocr_vl.genai import GenAIConfig
from rapid_doc.model.custom.paddleocr_vl.predictor import DocVLMPredictor
from rapid_doc.model.custom.paddleocr_vl.uilts import convert_otsl_to_html, crop_margin, tokenize_figure_of_table

class VLModelPool:
    _vl_model = None
    _lock = threading.Lock()

    @classmethod
    def get_vl_model(cls):

        if cls._vl_model is None:
            with cls._lock:
                if cls._vl_model is None:
                    backend = os.getenv('PADDLEOCRVL_VL_REC_BACKEND', 'vllm-server')
                    server_url = os.getenv('PADDLEOCRVL_VL_VL_REC_SERVER_URL')
                    paddleocrvl_version = os.getenv('PADDLEOCRVL_VERSION')
                    # 环境变量校验
                    if not paddleocrvl_version:
                        raise RuntimeError(
                            "PADDLEOCRVL_VERSION not set — VL OCR is disabled. "
                            "Please set environment variable PADDLEOCRVL_VERSION "
                            "(e.g. v1 or v1.5)."
                        )
                    if not server_url:
                        raise RuntimeError(
                            "PADDLEOCRVL_VL_VL_REC_SERVER_URL not set — "
                            "VL backend server url is required."
                        )
                    genai_config = GenAIConfig(
                        backend=backend,
                        server_url=server_url
                    )

                    if not paddleocrvl_version:
                        logger.warning("PADDLEOCRVL_VERSION not set, VL OCR disabled")
                        return
                    if paddleocrvl_version == "v1":
                        model_name = "PaddleOCR-VL-0.9B"
                    elif paddleocrvl_version == "v1.5":
                        model_name = "PaddleOCR-VL-1.5-0.9B"
                    else:
                        raise ValueError(
                            f"environment PADDLEOCRVL_VERSION Unsupported: {paddleocrvl_version}. "
                            "Supported versions: v1, v1.5"
                        )
                    cls._vl_model = DocVLMPredictor(
                        model_name=model_name,
                        genai_config=genai_config,
                    )

        return cls._vl_model


class PaddleOCRVLOCRModel(CustomBaseModel):

    def batch_predict(self, image_list: list, **kwargs) -> list[str]:
        result_res = []
        vl_rec_model = VLModelPool.get_vl_model()
        with tqdm(total=len(image_list), desc="OCR Predict") as pbar:
            data = [{"image": img, "query": "OCR:"} for img in image_list]
            preds = vl_rec_model._genai_client_process(data)

            for result_str in preds:
                if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                ):
                    result_str = result_str.replace("$", "")

                    result_str = (
                        result_str.replace("\\(", " $")
                        .replace("\\)", "$")
                        .replace("\\[\\[", "\\[")
                        .replace("\\]\\]", "\\]")
                        .replace("\\[", " $$ ")
                        .replace("\\]", " $$ ")
                    )
                result_res.append(result_str)
            pbar.update(len(image_list))
        return result_res

class PaddleOCRVLFormulaModel(CustomBaseModel):

    def batch_predict(self, image_list: list, **kwargs) -> list[str]:
        result_res = []
        vl_rec_model = VLModelPool.get_vl_model()
        with tqdm(total=len(image_list), desc="Formula Predict") as pbar:
            processed_images = []
            for block_img in image_list:
                crop_img = crop_margin(block_img)
                w, h, _ = crop_img.shape
                if w > 2 and h > 2:
                    block_img = crop_img
                processed_images.append(block_img)

            data = [{"image": img, "query": "Formula Recognition:"} for img in processed_images]
            preds = vl_rec_model._genai_client_process(data)

            for result_str in preds:
                if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                ):
                    # result_str = result_str.replace("$", "")
                    result_str = (
                        result_str.replace("\\(", "  ")
                        .replace("\\)", " ")
                        .replace("\\[\\[", "\\[")
                        .replace("\\]\\]", "\\]")
                        .replace("\\[", "  ")
                        .replace("\\]", "  ")
                    )
                result_res.append(result_str)
            pbar.update(len(image_list))
        return result_res

class PaddleOCRVLTableModel(CustomBaseModel):

    def batch_predict(self, image_list: list, **kwargs) -> list[str]:
        fill_image_res_list = kwargs.get('fill_image_res_list')
        result_res = []
        vl_rec_model = VLModelPool.get_vl_model()
        with tqdm(total=len(image_list), desc="Table Predict") as pbar:
            processed_images = []
            figure_token_map_list = []
            for i, block_img in enumerate(image_list):
                block_img, figure_token_map, drop_figures = (
                    tokenize_figure_of_table(
                        block_img, fill_image_res_list[i]
                    )
                )
                processed_images.append(block_img)
                figure_token_map_list.append(figure_token_map)
            data = [{"image": img, "query": "Table Recognition:"} for img in processed_images]
            preds = vl_rec_model._genai_client_process(data)

            for idx, result_str in enumerate(preds):
                if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                ):
                    result_str = result_str.replace("$", "")

                    result_str = (
                        result_str.replace("\\(", " $ ")
                        .replace("\\)", " $")
                        .replace("\\[\\[", "\\[")
                        .replace("\\]\\]", "\\]")
                        .replace("\\[", " $$ ")
                        .replace("\\]", " $$ ")
                    )
                html_str = convert_otsl_to_html(result_str)
                figure_token_map = figure_token_map_list[idx]
                for token, html in figure_token_map.items():
                    html_str = html_str.replace(token, html)
                result_res.append(html_str)
            pbar.update(len(image_list))
        return result_res
