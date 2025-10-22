import time
from PIL import Image
from typing import List, Tuple

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.boxbase import rotate_image_and_boxes
from ...utils.checkbox_det_cls import checkbox_predict
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img, get_res_list_from_layout_res
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_ocr_result_list_table
from ...utils.span_pre_proc import txt_spans_extract, txt_spans_bbox_extract, txt_spans_bbox_extract_table, \
    txt_most_angle_extract_table, extract_table_fill_image


# YOLO_LAYOUT_BASE_BATCH_SIZE = 8
# MFR_BASE_BATCH_SIZE = 16
# OCR_DET_BASE_BATCH_SIZE = 16

# LAYOUT_BASE_BATCH_SIZE = 1
# FORMULA_BASE_BATCH_SIZE = 1
# OCR_DET_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, formula_enable, table_enable, enable_ocr_det_batch: bool = False,
                layout_config=None,
                ocr_config=None,
                formula_config=None,
                table_config=None,
                checkbox_config=None):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.formula_level = formula_config.get("formula_level", 0) if formula_config else 0
        self.table_enable = get_table_enable(table_enable)
        self.table_force_ocr = table_config.get("force_ocr", False) if table_config else False
        self.skip_text_in_image = table_config.get("skip_text_in_image", True) if table_config else True
        self.use_img2table = table_config.get("use_img2table", False) if table_config else False
        self.checkbox_enable = checkbox_config.get("checkbox_enable", False) if checkbox_config else False
        self.layout_config = layout_config
        self.ocr_config = ocr_config
        self.formula_config = formula_config
        self.table_config = table_config
        self.model_manager = model_manager
        self.enable_ocr_det_batch = ocr_config.get("Det.rec_batch_num", 1) > 1 if ocr_config else False
        self.ocr_det_base_batch_size = ocr_config.get("Det.rec_batch_num", 1) if ocr_config else 1 #16
        self.layout_base_batch_size = layout_config.get("batch_num", 1) if layout_config else 1 #8
        self.formula_base_batch_size = formula_config.get("batch_num", 1) if formula_config else 1 #16
        self.use_det_mode = ocr_config.get("use_det_mode", 'auto') if ocr_config else 'auto'

    def __call__(self, images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict]]) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )
        atom_model_manager = AtomModelSingleton()

        pdf_dict_list = [pdf_dict for _, _, _, _, pdf_dict in images_with_extra_info]
        np_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image, _, _, _, _ in images_with_extra_info]
        scale_list = [scale for _, scale, _, _, _ in images_with_extra_info]


        # 版面识别
        images_layout_res += self.model.layout_model.batch_predict(
            np_images, self.layout_base_batch_size
        )

        # formula_level：公式识别等级，默认为0，全识别。
        # 如果公式识别等级为 1（只保留行间公式），过滤掉 category_id == 13（inline_formula）
        if self.formula_enable and self.formula_level == 1:
            images_layout_res = [
                [item for item in page if item["category_id"] != 13]
                for page in images_layout_res
            ]

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        latex_res_list_all_page = []
        for index in range(len(np_images)):
            _, _, ocr_enable, _lang, _ = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res, np_img)
            )

            # 复选框检测
            checkbox_res = []
            if self.checkbox_enable:
                checkbox_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                checkbox_res = checkbox_predict(checkbox_img)
                for res in checkbox_res:
                    poly = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                            res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
                    layout_res.append({'bbox': res['bbox'], 'poly': poly, 'category_id': CategoryId.CheckBox,
                                       'checkbox': res['text'], 'score': 0.9})

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'checkbox_res': checkbox_res,
                                          'layout_res':layout_res,
                                          'page_idx': index,
                                          })

            for table_res in table_res_list:
                table_img, useful_list = crop_img(table_res, np_img)
                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':table_img,
                                                'single_page_mfdetrec_res': single_page_mfdetrec_res,
                                                'checkbox_res': checkbox_res,
                                                'useful_list': useful_list,
                                                'ocr_enable': ocr_enable,
                                                'page_idx': index,
                                              })
            for latex_res in single_page_mfdetrec_res:
                latex_img, _ = crop_img(latex_res, np_img)
                latex_res_list_all_page.append({'latex_res': latex_res,
                                                'lang': _lang,
                                                'latex_img': latex_img,
                                                })

        if self.formula_enable:
            # 公式检测
            latex_imgs = [d['latex_img'] for d in latex_res_list_all_page]
            latex_results = self.model.formula_model.batch_predict(latex_imgs, batch_size=self.formula_base_batch_size)
            for d, res in zip(latex_res_list_all_page, latex_results):
                if res:
                    d['latex_res']['latex'] = res
                else:
                    logger.warning('latex recognition processing fails, not get latex return')

        # 清理显存
        # clean_vram(self.model.device, vram_threshold=8)

        if self.use_det_mode != 'ocr':
            # 分页分组
            ocr_res_list_grouped_page = {}
            for x in ocr_res_list_all_page:
                ocr_res_list_grouped_page.setdefault(x["page_idx"], []).append(x)
            # 计算总数
            total_texts = sum(len(texts) for texts in ocr_res_list_grouped_page.values())
            with tqdm(total=total_texts, desc="PDF-det Predict") as pbar:
                for page_idx, text_list in ocr_res_list_grouped_page.items():
                    if text_list:
                        page_dict = pdf_dict_list[page_idx]
                        scale = scale_list[page_idx]
                    for ocr_res_list_dict in text_list:
                        _lang = ocr_res_list_dict['lang']
                        if ocr_res_list_dict['ocr_enable']:
                            # 需要进行ocr的这里跳过
                            continue
                        # 从pdf中获取文本行点位
                        for res in ocr_res_list_dict['ocr_res_list']:
                            # res 点位信息
                            new_image, useful_list = crop_img(
                                res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                            )
                            # 跳过公式和复选框
                            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                                ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'],
                                useful_list
                            )
                            # PDF-det
                            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                            ocr_res = txt_spans_bbox_extract(page_dict, res, mfd_res=adjusted_mfdetrec_res, scale=scale, useful_list=useful_list) # 从pdf中获取文本行点位
                            # Integration results
                            if ocr_res:
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang, res['original_label']
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                        pbar.update(1)  # 每处理一个更新一次


        # OCR检测处理
        if self.enable_ocr_det_batch:
            # 批处理模式 - 按语言和分辨率分组
            # 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page:
                _lang = ocr_res_list_dict['lang']
                for res in ocr_res_list_dict['ocr_res_list']:
                    # 仅当整页OCR未启用时，判断是否跳过
                    if not ocr_res_list_dict['ocr_enable']:
                        if (
                                self.use_det_mode == 'txt' or
                                (self.use_det_mode != 'ocr' and not ocr_res_list_dict['ocr_enable'] and not res.get('need_ocr_det'))
                        ):
                            # 从 PDF 中直接提取文本框，无需 OCR，且已提取到框，则跳过
                            continue
                    res.pop('need_ocr_det', None)
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    # OCR检测，跳过公式和复选框
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'], useful_list
                    )

                    # BGR转换
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                    all_cropped_images_info.append((
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang
                    ))

            # 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang,
                    ocr_config=self.ocr_config,
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                RESOLUTION_GROUP_STRIDE = 64  # 定义分辨率分组的步进值

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[0]
                    h, w = cropped_img.shape[:2]
                    # 使用更大的分组容差，减少分组数量
                    # 将尺寸标准化到32的倍数
                    normalized_h = ((h + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE  # 向上取整到32的倍数
                    normalized_w = ((w + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (normalized_h, normalized_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for group_key, group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):

                    # 计算目标尺寸（组内最大尺寸，向上取整到32的倍数）
                    max_h = max(crop_info[0].shape[0] for crop_info in group_crops)
                    max_w = max(crop_info[0].shape[1] for crop_info in group_crops)
                    target_h = ((max_h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((max_w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE

                    # 对所有图像进行padding到统一尺寸
                    batch_images = []
                    for crop_info in group_crops:
                        img = crop_info[0]
                        h, w = img.shape[:2]
                        # 创建目标尺寸的白色背景
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        # 将原图像粘贴到左上角
                        padded_img[:h, :w] = img
                        batch_images.append(padded_img)

                    # 批处理检测
                    det_batch_size = min(len(batch_images), self.batch_ratio * self.ocr_det_base_batch_size)  # 增加批处理大小
                    # logger.debug(f"OCR-det batch: {det_batch_size} images, target size: {target_h}x{target_w}")
                    # batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)
                    batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

                    # 处理批处理结果
                    for i, (crop_info, (dt_boxes, elapse)) in enumerate(zip(group_crops, batch_results)):
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info

                        if dt_boxes is not None and len(dt_boxes) > 0:
                            # 直接应用原始OCR流程中的关键处理步骤

                            # 1. 排序检测框
                            if len(dt_boxes) > 0:
                                dt_boxes_sorted = sorted_boxes(dt_boxes)
                            else:
                                dt_boxes_sorted = []

                            # 2. 合并相邻检测框
                            if dt_boxes_sorted:
                                dt_boxes_merged = merge_det_boxes(dt_boxes_sorted)
                            else:
                                dt_boxes_merged = []

                            # 3. 根据公式位置更新检测框（关键步骤！）
                            if dt_boxes_merged and adjusted_mfdetrec_res:
                                dt_boxes_final = update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                            else:
                                dt_boxes_final = dt_boxes_merged

                            # 构造OCR结果格式
                            ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]

                            if ocr_res:
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang, res['original_label']
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang,
                    ocr_config=self.ocr_config,
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    # 仅当整页OCR未启用时，判断是否跳过
                    if not ocr_res_list_dict['ocr_enable']:
                        if (
                                self.use_det_mode == 'txt' or
                                (self.use_det_mode != 'ocr' and not ocr_res_list_dict['ocr_enable'] and not res.get('need_ocr_det'))
                        ):
                            # 从 PDF 中直接提取文本框，无需 OCR，且已提取到框，则跳过
                            continue
                    res.pop('need_ocr_det', None)
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    # OCR检测，跳过公式和复选框
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'], useful_list
                    )
                    # OCR-det
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(
                        bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_list_dict['ocr_enable'],bgr_image, _lang, res['original_label']
                        )
                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # 表格识别 table recognition
        if self.table_enable:
            # 分页分组
            table_res_list_grouped_page = {}
            for x in table_res_list_all_page:
                table_res_list_grouped_page.setdefault(x["page_idx"], []).append(x)
            # 计算总表格数
            total_tables = sum(len(tables) for tables in table_res_list_grouped_page.values())
            with tqdm(total=total_tables, desc="Table Predict") as pbar:
                for page_idx, table_list in table_res_list_grouped_page.items():
                    page_dict = pdf_dict_list[page_idx]
                    scale = scale_list[page_idx]
                    for table_res_dict in table_list:
                        _lang = table_res_dict['lang']
                        useful_list = table_res_dict['useful_list']
                        # OCR检测，跳过公式和复选框
                        adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                            table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'],
                            useful_list, return_text=True
                        )

                        ocr_result = None
                        if not self.table_force_ocr and not table_res_dict['ocr_enable']:
                            ocr_res = []
                            # if self.use_det_mode != 'ocr':
                            #     # 从pdf中直接提取文本块点位（部分表格效果较差暂不考虑），并支持270和90度的表格
                            #     ocr_res, most_angle = txt_spans_bbox_extract_table(page_dict, table_res_dict, scale=scale)  # 从pdf中获取文本行点位
                            if not ocr_res:
                                # RapidTable非OCR文本提取 OcrText
                                # 进行 OCR-det 识别文字框
                                ocr_model = atom_model_manager.get_atom_model(
                                    atom_model_name=AtomicModel.OCR,
                                    ocr_show_log=False,
                                    det_db_box_thresh=0.3,
                                    lang=_lang,
                                    ocr_config=self.ocr_config,
                                    enable_merge_det_boxes=False,
                                )
                                new_table_image = cv2.cvtColor(table_res_dict['table_img'], cv2.COLOR_RGB2BGR)
                                ocr_res = ocr_model.ocr(new_table_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]
                                most_angle = txt_most_angle_extract_table(page_dict, table_res_dict, scale=scale) # 从pdf中获取文本行 角度投票
                                if most_angle in [90, 270]:
                                    table_res_dict['table_img'], ocr_res = rotate_image_and_boxes(
                                        np.asarray(table_res_dict["table_img"]),
                                        ocr_res,
                                        most_angle
                                    )
                            if ocr_res:
                                ocr_spans = get_ocr_result_list_table(ocr_res, useful_list, scale)
                                poly = table_res_dict['table_res']['poly']
                                table_bboxes = [[int(poly[0]/scale), int(poly[1]/scale), int(poly[4]/scale), int(poly[5]/scale)
                                                    , None, None, None,'text', None, None, None, None, 1]]
                                # 从pdf中提取表格的文本
                                txt_spans_extract(page_dict, ocr_spans, table_res_dict['table_img'], scale, table_bboxes,[])
                                ocr_result = [list(x) for x in zip(*[[item['ori_bbox'], item['content'], item['score']] for item in ocr_spans])]
                        table_model = atom_model_manager.get_atom_model(
                            atom_model_name='table',
                            lang=_lang,
                            ocr_config=self.ocr_config,
                            table_config=self.table_config,
                        )
                        # 从pdf里提取表格里的图片
                        fill_image_res = extract_table_fill_image(page_dict, table_res_dict, scale=scale)
                        html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(table_res_dict['table_img'], ocr_result
                                                                                                 , fill_image_res, adjusted_mfdetrec_res, self.skip_text_in_image, self.use_img2table)
                        # 判断是否返回正常
                        if html_code:
                            # 检查html_code是否包含'<table>'和'</table>'
                            if '<table>' in html_code and '</table>' in html_code:
                                # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                                start_index = html_code.find('<table>')
                                end_index = html_code.rfind('</table>') + len('</table>')
                                table_res_dict['table_res']['html'] = html_code[start_index:end_index]
                                # 把图片里的公式和图片放到span，方便layout画图
                                latex_boxes = [t["bbox"] for t in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'] if "bbox" in t]
                                if latex_boxes:
                                    table_res_dict['table_res']['latex_boxes'] = [[int(coord / scale) for coord in bbox] for bbox in latex_boxes]
                                img_boxes = [t["ori_bbox"] for t in fill_image_res if "bbox" in t]
                                if img_boxes:
                                    table_res_dict['table_res']['img_boxes'] = [[int(coord / scale) for coord in bbox] for bbox in img_boxes]
                            else:
                                logger.warning(
                                    'table recognition processing fails, not found expected HTML table end'
                                )
                        else:
                            logger.warning(
                                'table recognition processing fails, not get html return'
                            )
                        pbar.update(1)  # 每处理一个表格更新一次

        # OCR rec
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')

        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    start = time.perf_counter()
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        det_db_box_thresh=0.3,
                        lang=lang,
                        ocr_config=self.ocr_config,
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        if ocr_score < OcrConfidence.min_confidence:
                            layout_res_item['category_id'] = 16
                        else:
                            layout_res_bbox = [layout_res_item['poly'][0], layout_res_item['poly'][1],
                                               layout_res_item['poly'][4], layout_res_item['poly'][5]]
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if ocr_text in ['（204号', '（20', '（2', '（2号', '（20号', '号', '（204'] and ocr_score < 0.8 and layout_res_width < layout_res_height:
                                layout_res_item['category_id'] = 16

                    total_processed += len(img_crop_list)

        return images_layout_res
