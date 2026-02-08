# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from .analyze_utils import _extract_text_from_pdf, _run_ocr_det_batch, _process_single_table, _run_ocr_rec_postprocess
from .model_init import AtomModelSingleton
from ..utils import remove_layout_in_ori_images, filter_overlap_boxes
from ...model.custom import CustomBaseModel
from ...utils.checkbox_det_cls import checkbox_predict
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.span_pre_proc import extract_table_fill_image


class BatchAnalyze:
    """批量分析处理器"""
    
    def __init__(
        self,
        model_manager,
        batch_ratio: int,
        formula_enable: bool,
        table_enable: bool,
        layout_config: Optional[Dict] = None,
        ocr_config: Optional[Dict] = None,
        formula_config: Optional[Dict] = None,
        table_config: Optional[Dict] = None,
        checkbox_config: Optional[Dict] = None,
    ):
        self.model_manager = model_manager
        self.batch_ratio = batch_ratio
        
        # 功能开关
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.checkbox_enable = checkbox_config.get("checkbox_enable", False) if checkbox_config else False
        
        # 配置项
        self.layout_config = layout_config or {}
        self.ocr_config = ocr_config or {}
        self.formula_config = formula_config or {}
        self.table_config = table_config or {}
        
        # OCR 配置
        self.use_det_mode = self.ocr_config.get("use_det_mode", "auto")
        self.ocr_det_base_batch_size = self.ocr_config.get("Det.rec_batch_num", 1)
        self.use_custom_ocr = False
        
        # 版面配置
        self.layout_base_batch_size = self.layout_config.get("batch_num", 1)
        
        # 公式配置
        self.formula_level = self.formula_config.get("formula_level", 0)
        self.formula_base_batch_size = self.formula_config.get("batch_num", 1)
        
        # 表格配置
        self.table_force_ocr = self.table_config.get("force_ocr", False)
        self.skip_text_in_image = self.table_config.get("skip_text_in_image", True)
        self.use_img2table = self.table_config.get("use_img2table", False)
        self.table_use_word_box = self.table_config.get("use_word_box", True)
        self.table_formula_enable = self.table_config.get("table_formula_enable", True)
        self.table_image_enable = self.table_config.get("table_image_enable", True)
        self.table_extract_original_image = self.table_config.get("extract_original_image", False)
    
    def __call__(
        self,
        images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict]]
    ) -> List[List[Dict]]:
        """
        执行批量分析
        
        Args:
            images_with_extra_info: [(PIL图像, 缩放比例, ocr_enable, 语言, pdf_dict), ...]
            
        Returns:
            每页的版面检测结果列表
        """
        if not images_with_extra_info:
            return []
        
        # 初始化模型
        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )
        self.use_custom_ocr = isinstance(self.model.ocr_model, CustomBaseModel)
        
        # 预处理数据
        pdf_dict_list = [pdf_dict for _, _, _, _, pdf_dict in images_with_extra_info]
        np_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image, _, _, _, _ in images_with_extra_info]
        scale_list = [scale for _, scale, _, _, _ in images_with_extra_info]
        
        # 1. 版面识别
        images_layout_res = self._run_layout_detection(np_images, pdf_dict_list, scale_list)
        # 2. 收集各类型检测区域
        ocr_res_all_page, table_res_all_page, formula_res_all_page = self._collect_detection_regions(
            images_layout_res, np_images, images_with_extra_info
        )
        # 3. 公式识别
        if self.formula_enable:
            self._run_formula_recognition(formula_res_all_page)
        # 4. OCR 识别 (根据模式选择不同的处理方式)
        if isinstance(self.model.ocr_model, CustomBaseModel):
            self._run_custom_ocr(ocr_res_all_page)
        else:
            self._run_traditional_ocr(
                ocr_res_all_page, pdf_dict_list, scale_list
            )
        # 5. 表格识别
        if self.table_enable:
            self._run_table_recognition(table_res_all_page, pdf_dict_list, scale_list)
        # 6. 后处理 OCR rec 结果
        _run_ocr_rec_postprocess(images_layout_res, self.ocr_config)

        clean_vram(self.model.device, vram_threshold=8)
        return images_layout_res
    
    def _run_layout_detection(
        self,
        np_images: List[np.ndarray],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ) -> List[List[Dict]]:
        """执行版面检测"""
        images_layout_res = self.model.layout_model.batch_predict(
            np_images, self.layout_base_batch_size
        )
        images_layout_res = [filter_overlap_boxes(item, self.use_custom_ocr) for item in images_layout_res]
        # 如果是 txt 模式，移除原始图片中的版面元素
        if self.use_det_mode == 'txt':
            images_layout_res = remove_layout_in_ori_images(images_layout_res, pdf_dict_list, scale_list)
        
        # 公式等级过滤
        if self.formula_enable and self.formula_level == 1:
            images_layout_res = [
                [item for item in page if item["category_id"] != CategoryId.InlineEquation]
                for page in images_layout_res
            ]
        
        return images_layout_res
    
    def _collect_detection_regions(
        self,
        images_layout_res: List[List[Dict]],
        np_images: List[np.ndarray],
        images_with_extra_info: List[Tuple]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """收集各类型检测区域"""
        ocr_res_all_page = []
        table_res_all_page = []
        formula_res_all_page = []
        
        for index, np_img in enumerate(np_images):
            _, _, ocr_enable, _lang, _ = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            
            ocr_res_list, table_res_list, formula_res_list = get_res_list_from_layout_res(layout_res, np_img)
            
            # 复选框检测
            checkbox_res = []
            if self.checkbox_enable:
                checkbox_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                checkbox_res = checkbox_predict(checkbox_img)
                for res in checkbox_res:
                    poly = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                            res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
                    layout_res.append({
                        'bbox': res['bbox'], 'poly': poly,
                        'category_id': CategoryId.CheckBox,
                        'checkbox': res['text'], 'score': 0.9
                    })

            # OCR 区域
            ocr_res_all_page.append({
                'ocr_res_list': ocr_res_list,
                'lang': _lang,
                'ocr_enable': ocr_enable,
                'np_img': np_img,
                'single_page_mfdetrec_res': formula_res_list,
                'checkbox_res': checkbox_res,
                'layout_res': layout_res,
                'page_idx': index,
            })

            # 表格区域
            for table_res in table_res_list:
                table_img, useful_list = crop_img(table_res, np_img)
                rect_table_img, _ = crop_img(table_res, np_img, layout_shape_mode="rect")
                table_res_all_page.append({
                    'table_res': table_res,
                    'lang': _lang,
                    'table_img': table_img, #矩形框/异型框的表格
                    'rect_table_img': rect_table_img, #矩形框的表格
                    'single_page_mfdetrec_res': formula_res_list,
                    'checkbox_res': checkbox_res,
                    'useful_list': useful_list,
                    'ocr_enable': ocr_enable,
                    'page_idx': index,
                })

            # 公式区域
            for formula_res in formula_res_list:
                formula_img, _ = crop_img(formula_res, np_img)
                formula_res_all_page.append({
                    'formula_res': formula_res,
                    'lang': _lang,
                    'formula_img': formula_img,
                })

        return ocr_res_all_page, table_res_all_page, formula_res_all_page

    def _run_formula_recognition(self, formula_res_all_page: List[Dict]):
        """执行公式识别"""
        formula_imgs = [d['formula_img'] for d in formula_res_all_page]
        if not formula_imgs:
            return
        formula_results = self.model.formula_model.batch_predict(
            formula_imgs, batch_size=self.formula_base_batch_size
        )
        for d, res in zip(formula_res_all_page, formula_results):
            if res:
                d['formula_res']['latex'] = res
            else:
                logger.warning('latex recognition processing fails')

    def _run_custom_ocr(self, ocr_res_all_page: List[Dict]):
        """
        使用 VL 模型进行 OCR 识别

        VL OCR 直接识别整个文本区域，返回多行文本，不产生单行 span
        """
        # 收集所有需要 OCR 的区域
        all_ocr_regions = []

        for ocr_res_dict in ocr_res_all_page:
            for res in ocr_res_dict['ocr_res_list']:
                # 裁剪图像
                new_image, useful_list = crop_img(res, ocr_res_dict['np_img'])
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                all_ocr_regions.append({
                    'image': bgr_image,
                    'res': res,
                    'layout_res': ocr_res_dict['layout_res'],
                    'useful_list': useful_list,
                })

        if not all_ocr_regions:
            return

        # 批量 VL OCR
        images = [r['image'] for r in all_ocr_regions]
        ocr_texts = self.model.ocr_model.batch_predict(
            images, batch_size=self.ocr_det_base_batch_size
        )

        # 将结果填充到 layout_res
        for region, text in zip(all_ocr_regions, ocr_texts):
            res = region['res']

            # 构建 VL OCR 结果，使用特殊的 category_id 标识
            vl_ocr_result = {
                'poly': res['poly'],
                'category_id': CategoryId.OcrText,
                'score': 0.95,  # VL 模型置信度较高
                'text': text.strip() if text else '',
                'vl_ocr': True,  # 标识这是 VL OCR 结果
                'original_label': res.get('original_label'),
                'original_order': res.get('original_order'),
                'polygon_points': res.get('polygon_points'),
            }

            region['layout_res'].append(vl_ocr_result)

    def _run_traditional_ocr(
        self,
        ocr_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float],
    ):
        """传统 OCR 处理流程 (det + rec)"""
        atom_model_manager = AtomModelSingleton()

        # PDF 文本提取模式
        if self.use_det_mode != 'ocr':
            _extract_text_from_pdf(ocr_res_all_page, pdf_dict_list, scale_list)

        # OCR 检测处理
        _run_ocr_det_batch(ocr_res_all_page, atom_model_manager, self.ocr_config)

    def _run_table_recognition(
        self,
        table_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ):
        """执行表格识别"""
        if isinstance(self.model.table_model, CustomBaseModel):
            table_imgs = []
            fill_image_res_list = []
            for table_res_dict in table_res_all_page:
                page_idx = table_res_dict['page_idx']
                page_dict = pdf_dict_list[page_idx]
                scale = scale_list[page_idx]
                fill_image_res = []
                if self.table_image_enable:
                    fill_image_res = extract_table_fill_image(
                        page_dict, table_res_dict, scale, self.table_extract_original_image
                    )
                table_imgs.append(table_res_dict["table_img"])
                fill_image_res_list.append(fill_image_res)

            if table_imgs:
                table_results = self.model.table_model.batch_predict(
                    table_imgs, fill_image_res_list=fill_image_res_list
                )
                for table_res_dict, table_result in zip(table_res_all_page, table_results):
                    table_res_dict['table_res'].pop('layout_image_list', None)
                    if table_result:
                        table_res_dict['table_res']['html'] = table_result
        else:
            # 传统模式表格识别
            self._run_traditional_table_recognition(table_res_all_page, pdf_dict_list, scale_list)

    def _run_traditional_table_recognition(
        self,
        table_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ):
        """传统表格识别处理"""
        atom_model_manager = AtomModelSingleton()

        table_res_grouped = {}
        for x in table_res_all_page:
            table_res_grouped.setdefault(x["page_idx"], []).append(x)

        total_tables = sum(len(tables) for tables in table_res_grouped.values())

        with tqdm(total=total_tables, desc="Table Predict") as pbar:
            for page_idx, table_list in table_res_grouped.items():
                page_dict = pdf_dict_list[page_idx]
                scale = scale_list[page_idx]

                for table_res_dict in table_list:
                    table_res_dict["table_img"] = table_res_dict["rect_table_img"]
                    _process_single_table(
                        table_res_dict, page_dict, scale, atom_model_manager,
                        self.table_config,
                        self.ocr_config,
                    )
                    pbar.update(1)

